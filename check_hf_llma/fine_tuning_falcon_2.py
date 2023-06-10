#
# https://medium.com/@bnjmn_marie/fine-tune-falcon-7b-on-your-gpu-with-trl-and-qlora-4490fadc3fbb
#
import bitsandbytes
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("timdettmers/openassistant-guanaco")

model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()

# --

import torch, einops
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer


def create_and_prepare_model():
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b", quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value"
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=10000,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

model, peft_config, tokenizer = create_and_prepare_model()
model.config.use_cache = False
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="train",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=True,
)

trainer.train()