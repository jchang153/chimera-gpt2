import torch

def generate_text(model, tokenizer, prompt, device, max_new_tokens, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            topk_logits, topk_indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(topk_logits, dim=-1)
            next_token = topk_indices[0][torch.multinomial(probs, num_samples=1)]
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        print(tokenizer.decode(next_token), end="", flush=True)

    print("\n---\nFull output:")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def evaluate_c(loader, model, controller, device, max_batches=None):
    controller.set_active("ALL")
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            out = model(input_ids=x, labels=y)
            total_loss += out.loss.item() * x.numel()
            total_tokens += x.numel()

            if max_batches is not None and step >= max_batches:
                break

    model.train()
    return total_loss / max(1, total_tokens)   # per-token average loss

def evaluate(loader, model, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            out = model(input_ids=x, labels=y)
            total_loss += out.loss.item() * x.numel()
            total_tokens += x.numel()

            if max_batches is not None and step >= max_batches:
                break

    model.train()
    return total_loss / max(1, total_tokens)   # per-token average loss