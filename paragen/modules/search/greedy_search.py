from typing import Optional
import torch

from paragen.modules.search import register_search
from paragen.modules.search.sequence_search import SequenceSearch
from paragen.modules.utils import create_init_scores


@register_search
class GreedySearch(SequenceSearch):
    """
    GreedySearch is greedy search on sequence generation.

    Args:
        maxlen_coef (a, b): maxlen computation coefficient.
            The max length is computed as `(S * a + b)`, where S is source sequence length.
    """

    def __init__(self, maxlen_coef=(1.2, 10)):
        super().__init__()

        self._maxlen_a, self._maxlen_b = maxlen_coef

    def forward(self,
                prev_tokens,
                memory,
                memory_padding_mask):
        batch_size = prev_tokens.size(0)
        scores = create_init_scores(prev_tokens, memory)
        prevs_layers = torch.zeros(self.num_decoder_layers, 1, 1, 1)
        for k in range(int(memory.size(0) * self._maxlen_a + self._maxlen_b)):
            logits, caches = self._decoder(prev_tokens, memory, memory_padding_mask, prevs_layers)
            logits = logits[:, -1, :]
            next_word_scores, words = logits.max(dim=-1)
            eos_mask = words.eq(self._eos)
            if eos_mask.long().sum() == batch_size:
                break
            prev_sequence_length = prevs_layers.size(1)
            feature_dim = caches.size(3)
            expanded_prevs_layers = prevs_layers.expand(self.num_decoder_layers, prev_sequence_length, batch_size, feature_dim)
            prevs_layers = torch.cat([expanded_prevs_layers, caches], dim=1)

            scores = scores + next_word_scores.masked_fill_(eos_mask, torch.tensor(0).float()).view(-1)
            prev_tokens = torch.cat([prev_tokens, words.unsqueeze(dim=-1)], dim=-1)
        return scores, prev_tokens
