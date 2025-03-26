import torch
import torch.nn as nn

from gpt.core.layer.attention import CausalSelfAttention


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):

        token_embeddings = self.tok_emb(input_ids)

        x = token_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = torch.matmul(x, self.tok_emb.weight.transpose(0, 1))
        return logits


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x