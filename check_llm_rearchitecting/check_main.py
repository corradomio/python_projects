from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from optipfair.pruning.depth import prune_model_depth
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import time
import json
import logging.config

# ---------------------------------------------------------------------------

logging.config.fileConfig('logging_config.ini')

freeze_support()

# ---------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-3-270m"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------------------------

original_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {original_params}")
print(f"Original layers: {len(model.model.layers)}")
print("=" * 20)
print(model)

# ---------------------------------------------------------------------------

def generate_text(model, tokenizer, prompt , max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            num_beams=3,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


MAX_NEW_TOKENS = 50
TEST_PROMPT = "Paris is the capital of"
generated_text = generate_text(
        model,
        tokenizer,
        TEST_PROMPT,
        MAX_NEW_TOKENS
    )
print(f"Prompt: '{TEST_PROMPT}'")
print(f"Generated Text: '{generated_text}'")

# ---------------------------------------------------------------------------

from utils import model_evaluation

benchmark_tasks = ['arc_easy', 'winogrande','hellaswag', 'lambada_openai']
baseline_results = model_evaluation(
    model,
    tokenizer,
    benchmark_tasks, limit=100,
    batch_size=1
)

print(baseline_results)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
