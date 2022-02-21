from typing import Optional

import torch
from torch import Tensor

from paragen.modules.search import AbstractSearch


class SequenceSearch(AbstractSearch):
    """
    SequenceSearch algorithms are used to generate a complete sequence with strategies.
    It usually built from a one-step neural model and fledges the model to a full-step generation.
    """

    def __init__(self):
        super().__init__()

        self._decoder = None
        self._bos, self._eos, self._pad = None, None, None

    def build(self, decoder, bos, eos, pad, *args, **kwargs):
        self._decoder = decoder
        self.num_decoder_layers = self._decoder._num_layers
        self._bos, self._eos, self._pad = bos, eos, pad

    def trace_decoder(self):
        tgt = torch.zeros([8, 1], dtype=torch.int64)
        memory = torch.zeros([16, 8, 64])
        memory_padding_mask = torch.zeros([8, 16])
        # prevs_layers = [torch.zeros([1, 1, 1]) for _ in range(self.num_decoder_layers)]
        prevs_layers = torch.zeros([self.num_decoder_layers, 1, 1, 1])
        self._decoder.eval()
        self._decoder.reset('infer')
        with torch.no_grad():
            self._decoder = torch.jit.trace(self._decoder, (tgt, memory, memory_padding_mask, prevs_layers), check_trace=False)

    def forward(self,
                prev_tokens: Tensor,
                memory: Tensor,
                memory_padding_mask: Tensor,
                target_mask: Optional[Tensor] = None,
                prev_scores: Optional[Tensor] = None):
        """
        Decoding full-step sequence

        Args:
            prev_tokens: previous tokens or prefix of sequence
            memory: memory for attention.
              :math:`(M, N, E)`, where M is the memory sequence length, N is the batch size,
            memory_padding_mask: memory sequence padding mask.
              :math:`(N, M)` where M is the memory sequence length, N is the batch size.
            target_mask: target mask indicating blacklist tokens
              :math:`(B, V)` where B is batch size and V is vocab size
            prev_scores: scores of previous tokens
              :math:`(B)` where B is batch size

        Returns:
            - log probability of generated sequence
            - generated sequence
        """
        raise NotImplementedError

    def reset(self, mode):
        """
        Reset encoder and switch running mode

        Args:
            mode: running mode in [train, valid, infer]
        """
        self._mode = mode
        self._decoder.reset(mode)

