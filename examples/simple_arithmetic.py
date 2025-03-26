from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

from gpt.core.model.gpt2 import GPT
from gpt.core.model.hf_wrapper import HuggingfaceWrapper
from gpt.core.utils.autoregressive_inference import autoregressive_print
from gpt.core.utils.show_params import print_model_parameters
from gpt.data.dataloaders.simple_math import get_math_dataset

tokenizer = AutoTokenizer.from_pretrained("fhswf/BPE_GPT2_TinyStoriesV2_cleaned_1024")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


dataset = get_math_dataset(tokenizer)
dataset = dataset.train_test_split(
    test_size=0.001,
    shuffle=True,
    seed=42
)

n_layer = 4
n_embd = 16
n_head = 4
context = 256

model = GPT(
    vocab_size=tokenizer.vocab_size,
    block_size=context,
    n_layer=n_layer,
    n_embd=n_embd,
    n_head=n_head
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = HuggingfaceWrapper(model)
print_model_parameters(model)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,

    learning_rate=3e-4,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    remove_unused_columns=True,
    report_to="tensorboard",
    optim="adamw_torch",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)


trainer.train()

autoregressive_print()
