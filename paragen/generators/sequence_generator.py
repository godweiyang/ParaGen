from typing import Dict
import logging
logger = logging.getLogger(__name__)

import torch

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.encoders import AbstractEncoder
from paragen.modules.search import create_search, AbstractSearch
from paragen.utils.runtime import Environment
from paragen.utils.io import UniIO, mkdir, cp
from paragen.utils.tensor import to_device


@register_generator
class SequenceGenerator(AbstractGenerator):
    """
    SequenceGenerator is combination of a model and search algorithm.
    It processes in a multi-step fashion while model processes only one step.
    It is usually separated into encoder and search with decoder, and is
    exported and load with encoder and search module.

    Args:
        search: search configs
        path: path to export or load generator
    """

    def __init__(self,
                 search: Dict=None,
                 path=None):
        super().__init__(path)
        self._search_configs = search

        self._model = None
        self._encoder, self._search = None, None
        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._env = None

    def build_from_model(self, model, src_special_tokens, tgt_special_tokens):
        """
        Build generator from model and search.

        Args:
            model (paragen.models.EncoderDecoder): an encoder-decoder model to be wrapped
            src_special_tokens (dict): source special token dict
            tgt_special_tokens (dict): target special token dict
        """
        self._model = model
        self._encoder = model.encoder
        self._src_special_tokens, self._tgt_special_tokens = src_special_tokens, tgt_special_tokens

        self._search = create_search(self._search_configs)
        self._search.build(decoder=model.decoder,
                           bos=self._tgt_special_tokens['bos'],
                           eos=self._tgt_special_tokens['eos'],
                           pad=self._tgt_special_tokens['pad'])
        self._env = Environment()

    def _forward(self, encoder, decoder, search=None):
        """
        Infer a sample as model in evaluation mode.
        Compute encoder output first and decode results with search module

        Args:
            encoder (tuple): encoder inputs
            decoder (tuple): decoder inputs
            search (tuple): search states

        Returns:
            decoder_output: results inferred by search algorithm on decoder
        """
        if not search:
            search = tuple()
        encoder_output = self._encoder(*encoder)
        _, decoder_output = self._search(*decoder, *encoder_output, *search)
        return decoder_output

    def export(self, path, net_input, *args, **kwargs):
        self.eval()
        self.reset('infer')
        net_input = to_device(net_input, device=self._env.device)
        with torch.no_grad():
            logger.info(f'trace encoder {self._encoder.__class__.__name__}')
            encoder = torch.jit.trace_module(self._encoder, {'forward': net_input['encoder']})
            mkdir(path)
        logger.info(f'save encoder to {path}/encoder')
        with UniIO(f'{path}/encoder', 'wb') as fout:
            torch.jit.save(encoder, fout)

        if 'use_onnx' in kwargs and kwargs['use_onnx']:
            logger.info('exporting onnx model')
            opset_version = 14
            kwargs = dict(opset_version=opset_version,
                        do_constant_folding=True,
                        strip_doc_string=True)
            input_names = ['prev_tokens', 'memory', 'memory_padding_mask']
            output_names = ['scores', 'output_tokens']
            dynamic_axes = {'prev_tokens': {0: 'batch_size', 1: 'start_flag'},
                            'memory': {0: 'sequence_length', 1: 'batch_size', 2: 'embedding_dimension'},
                            'memory_padding_mask': {0: 'batch_size', 1: 'sequence_length'},
                            'scores': {0: 'batch_size'},
                            'output_tokens': {0: 'batch_size', 1: 'trg_seq_len'}}
            kwargs['dynamic_axes'] = dynamic_axes
            kwargs['input_names'] = input_names
            kwargs['output_names'] = output_names
            torch.manual_seed(666)
            prev_tokens = torch.zeros([16, 1], dtype=torch.int64)
            memory = torch.rand([16, 16, 64])
            memory_padding_mask = torch.zeros([16, 16])
            
            torch.set_printoptions(profile='full')
            _, torch_decoder_output = self._search(prev_tokens, memory, memory_padding_mask)
            print('torch_decoder_output: \n', torch_decoder_output)
            
            self._search.trace_decoder()
            with torch.no_grad():
                logger.info(f'script search {self._search.__class__.__name__}')
                search = torch.jit.script(self._search)
            
            torch.onnx.export(search, (prev_tokens, memory, memory_padding_mask),
                            f'{path}/greedy_search.onnx', **kwargs)

             # export the encoder
            encoder_input_names = ['encoder_input']
            encoder_output_names = ['memory', 'memory_padding_mask']
            dynamic_axes = {'encoder_input': {0: 'batch_size', 1: 'source_sequence_length'},
                            'memory': {0: 'sequence_length', 1: 'batch_size', 2: 'embedding_dimension'},
                            'memory_padding_mask': {0: 'batch_size', 1: 'sequence_length'}}
            kwargs['dynamic_axes'] = dynamic_axes
            kwargs['input_names'] = encoder_input_names
            kwargs['output_names'] = encoder_output_names
            torch.onnx.export(self._encoder, (net_input['encoder']),
                              f'{path}/encoder.onnx', **kwargs)
            import onnxruntime
            import onnx
            _, script_decoder_output = search(prev_tokens, memory, memory_padding_mask)
            print('script_decoder_output: \n', script_decoder_output)
            input_dict = {'prev_tokens': prev_tokens.numpy(),
                        'memory': memory.numpy(),
                        'memory_padding_mask': memory_padding_mask.numpy()}
            ort_session = onnxruntime.InferenceSession(f'{path}/greedy_search.onnx')
            _, onnx_decoder_output = ort_session.run(output_names, input_dict)
            print('onnx_decoder_output: \n', onnx_decoder_output)
            print('\n ============================== \n')

    def load(self):
        """
        Load generator (encoder & search) from path
        """
        logger.info('load encoder from {}/encoder'.format(self._path))
        with UniIO('{}/encoder'.format(self._path), 'rb') as fin:
            self._encoder = torch.jit.load(fin)
        logger.info('load search from {}/search'.format(self._path))
        with UniIO('{}/search'.format(self._path), 'rb') as fin:
            self._search = torch.jit.load(fin)

    @property
    def encoder(self):
        return self._encoder

    @property
    def search(self):
        return self._search

    def reset(self, mode):
        """
        Reset generator states.

        Args:
            mode: running mode
        """
        self.eval()
        self._mode = mode
        if self._traced_model is None:
            if isinstance(self._encoder, AbstractEncoder):
                self._encoder.reset(mode)
            if isinstance(self._search, AbstractSearch):
                self._search.reset(mode)
            if self._env.device == 'cuda':
                torch.cuda.empty_cache()
