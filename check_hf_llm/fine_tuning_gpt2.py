# https://metatext.io/blog/how-to-finetune-llm-hugging-face-transformers
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device('cuda')

# --

# Tokenize input data
text = "Sample text for fine-tuning GPT-2 model."
inputs = tokenizer.encode(text, return_tensors='pt')

# Prepare labels
labels = inputs.clone()
labels[0, :-1] = inputs[0, 1:]

# --

from torch.utils.data import Dataset, DataLoader


# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Train model on data
train_dataset = TextDataset(inputs)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in range(5):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['input_ids'].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

# --

# Generate text
generated = model.generate(inputs, max_length=50, do_sample=True)
decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
print(decoded)

