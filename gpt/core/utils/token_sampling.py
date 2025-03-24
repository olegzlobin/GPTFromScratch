import torch

def sample_token(next_token_logits: torch.Tensor,
                 temperature: float = 1.0,
                 top_k: int = None
                 ) -> torch.Tensor:
    scaled_logits = next_token_logits / temperature

    if top_k is not None and top_k > 0:
        top_k_values = torch.topk(scaled_logits, top_k, dim=-1)
        indices_to_remove = scaled_logits < top_k_values.values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.softmax(scaled_logits, dim=-1)
    return probs