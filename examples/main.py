from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

from gpt.core.model.gpt2 import GPT
from gpt.core.utils.show_params import print_model_parameters

block_size = 1024


dataset = load_dataset("IgorVolochay/russian_jokes")


tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruGPT-3.5-13B")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        padding=False
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

n_layer = 1
n_embd = 16
n_head = 2
context = 1024

model = GPT(
    vocab_size=tokenizer.vocab_size,
    block_size=context,
    n_layer=n_layer,
    n_embd=n_embd,
    n_head=n_head
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print_model_parameters(model)


training_args = TrainingArguments(
    output_dir="./results",
    logging_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=True,
    report_to="none",
    save_strategy="no",
    optim="adamw_torch",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)


trainer.train()