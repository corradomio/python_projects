from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("garage-bAInd/Platypus2-7B", profiling_mode=True)
#model = AirLLMLlama2("garage-bAInd/Platypus2-7B", profiling_mode=False)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf", profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("baichuan-inc/Baichuan2-7B-Base", profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("Qwen/Qwen-7B", profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base", profiling_mode=True)
model = AutoModel.from_pretrained('/root/.cache/huggingface/hub/models--THUDM--chatglm3-6b-base/snapshots/f91a1de587fdc692073367198e65369669a0b49d/', profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

from airllm import AutoModel

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained("internlm/internlm-20b", profiling_mode=True)

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
    #'What is the capital of China?',
    'I like',
]

input_tokens = model.tokenizer(input_text,
                               return_tensors="pt",
                               return_attention_mask=False,
                               truncation=True,
                               max_length=MAX_LENGTH,
                               #padding=True
                               )

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=3,
    use_cache=True,
    return_dict_in_generate=True)

model.tokenizer.decode(generation_output.sequences[0])

# ---------------------------------------------------------------------------

