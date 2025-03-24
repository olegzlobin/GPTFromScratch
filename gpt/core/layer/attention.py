import torch
import torch.nn as nn

from gpt.core.layer.RoPE import RotaryPositionalEmbedding


class AttentionMask(nn.Module):
    def __init__(self, max_seq_length=1024):
        super().__init__()
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, max_seq_length, max_seq_length)))

    def forward(self, attn_weights, seq_length):
        causal_mask = self.bias[:, :, :seq_length, :seq_length].bool()
        return attn_weights.masked_fill(~causal_mask, float('-inf'))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, context, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.rotary = RotaryPositionalEmbedding(self.head_dim, context)

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.mask = AttentionMask(context)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary.apply_rotary(q, T)
        k = self.rotary.apply_rotary(k, T)

        att = q @ k.transpose(-2, -1)
        att = self.mask(att, T)
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))

        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, context, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.mask = AttentionMask(context)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]

        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float, device=x.device))
        att = (q @ k.transpose(-2, -1)) / scale

        att = self.mask(att, T)
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))

        return y