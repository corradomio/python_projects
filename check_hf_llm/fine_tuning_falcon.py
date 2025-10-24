#
# https://medium.com/@bnjmn_marie/fine-tune-falcon-7b-on-your-gpu-with-trl-and-qlora-4490fadc3fbb
#

#
# Non funziona:
# OutOfMemoryError: CUDA out of memory. Tried to allocate 316.00 MiB (GPU 0;
# 16.00 GiB total capacity; 15.18 GiB already allocated; 0 bytes free; 15.22 GiB
# reserved in total by PyTorch) If reserved memory is >> allocated memory try
# setting max_split_size_mb to avoid fragmentation.  See documentation for Memory
# Management and PYTORCH_CUDA_ALLOC_CONF
#
# Ma non e' vero: anche modificando max_split_size_mb non cambia nulla
# E nemmeno riducendo 'max_seq_length'

# https://pytorch.org/docs/stable/notes/cuda.html#memory-management
# PYTORCH_CUDA_ALLOC_CONF
#

import os
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:12000'

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
    # max_seq_length=256,
)
trainer.train()
print("done")
