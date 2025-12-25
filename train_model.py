import config
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )

df = pd.read_csv(config.DATA_PATH).head(100)

texts = []
for _, row in df.iterrows():
    title = str(row["title"]).strip()
    text = str(row["text"]).strip()
    full_text = f"{title}\n{text}"
    texts.append(full_text)

dataset = Dataset.from_dict({"text": texts})
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=config.LORA_PATH,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    warmup_ratio=0.05,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_steps=50,
    save_steps=200,
    save_total_limit=5,
    remove_unused_columns=False,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(config.LORA_PATH)
tokenizer.save_pretrained(config.LORA_PATH)