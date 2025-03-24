import torch

from gpt.core.utils.token_sampling import sample_token


def _base_autoregressive_loop(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_length: int,
    temperature: float,
    top_k: int,
    device: str
):
    generated_ids = input_ids.clone()

    for _ in range(max_length - input_ids.size(1)):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs["logits"][:, -1, :]

        probs = sample_token(next_token_logits, temperature, top_k)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        next_token_id = next_token.squeeze().item()
        yield next_token_id, generated_ids

        if next_token_id == tokenizer.eos_token_id:
            break

def autoregressive_generate(
    model,
    tokenizer,
    prompt: str = "",
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for token_id, generated_ids in _base_autoregressive_loop(
        model, input_ids, tokenizer, max_length, temperature, top_k, device
    ):
        pass

    return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

def autoregressive_print(
    model,
    tokenizer,
    prompt: str = "",
    max_length: int = 500,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(prompt, end="", flush=True)
    generated_ids = input_ids.clone()

    for token_id, ids in _base_autoregressive_loop(
        model, input_ids, tokenizer, max_length, temperature, top_k, device
    ):
        generated_ids = ids
        new_text = tokenizer.decode([token_id], skip_special_tokens=True)
        print(new_text, end="", flush=True)

    print()
    return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)