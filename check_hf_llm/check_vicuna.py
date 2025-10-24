import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# model = "gpt2-xl"
model = "bigscience/bloom-560m"


tokenizer = AutoTokenizer.from_pretrained(model)
text_gen = AutoModelForCausalLM.from_pretrained(model).to(device)


# input_txt = "Transformers are the"
input_txt = "Mickey Mouse is"
n_steps = 128

# input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)

# import pandas as pd
# iterations = []
# choices_per_step = 5
# with torch.no_grad():
#     for _ in range(n_steps):
#         iteration = dict()
#         iteration["Input"] = tokenizer.decode(input_ids[0])
#         output = model(input_ids=input_ids)
#         # Select logits of the first batch and the last token and apply softmax
#         next_token_logits = output.logits[0, -1, :]
#         next_token_probs = torch.softmax(next_token_logits, dim=-1)
#         sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
#         # Store tokens with highest probabilities
#         for choice_idx in range(choices_per_step):
#             token_id = sorted_ids[choice_idx]
#             token_prob = next_token_probs[token_id].cpu().numpy()
#             token_choice = (
#                 f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
#             )
#         iteration[f"Choice {choice_idx+1}"] = token_choice
#         # Append predicted next token to input
#         input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
#     iterations.append(iteration)
#
# pprint(pd.DataFrame(iterations))


input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output = text_gen.generate(input_ids, max_length=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))
print()
output = text_gen.generate(input_ids, max_length=n_steps, do_sample=True)
print(tokenizer.decode(output[0]))
