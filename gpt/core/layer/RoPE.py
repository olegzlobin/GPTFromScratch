import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_length=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        inv_freq = inv_freq.to(torch.get_default_dtype())
        t = torch.arange(max_seq_length, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, x, seq_len):
        cos = self.freqs_cos[:seq_len].view(1, 1, seq_len, self.dim//2)
        sin = self.freqs_sin[:seq_len].view(1, 1, seq_len, self.dim//2)

        cos = cos.expand(-1, x.size(1), -1, -1)
        sin = sin.expand(-1, x.size(1), -1, -1)

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        return x * cos + self._rotate_half(x) * sin