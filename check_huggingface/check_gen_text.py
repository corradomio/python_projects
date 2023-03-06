import pandas as pd
import torch
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# model_name = "gpt2-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

model_name = "gpt2-xl"
# model_name = "bigscience/bloom-560m"
# model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# model = BloomModel.from_pretrained(model_name)

# input_txt = "Transformers are the"
# input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
# iterations = []
# n_steps = 8
# choices_per_step = 5
#
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
#             iteration[f"Choice {choice_idx + 1}"] = token_choice
#         # Append predicted next token to input
#         input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
#         iterations.append(iteration)
#
# df = pd.DataFrame(iterations)
# print(df)
#
# input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
# output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
# print(tokenizer.decode(output[0]))

max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output_greedy[0]))
