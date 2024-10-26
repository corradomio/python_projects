from transformers import AutoConfig


def print_config(model_id):
    print(model_id)
    print(AutoConfig.from_pretrained(model_id))


# print_config("bigscience/bloomz-560m")
# print_config("bigscience/bloom-3b")
print_config("garage-bAInd/Platypus2-70B-instruct")
print_config("tiiuae/falcon-7b")
