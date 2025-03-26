import torch.nn as nn


class HuggingfaceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def get_loss(self, logits, labels, attention_mask):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_mask = attention_mask[..., 1:].contiguous()

        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        active_loss = shift_mask.view(-1) == 1
        active_loss = active_loss.float().sum()
        return (loss * active_loss) / active_loss if active_loss > 0 else loss * 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        if labels is not None:
            loss = self.get_loss(logits, labels, attention_mask)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}