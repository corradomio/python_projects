#
# https://towardsdatascience.com/fine-tune-a-large-language-model-with-python-b1c09dbc58b2
#

import gzip
import shutil
import time

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchtext
import torch.backends.cudnn
import torch.utils.data

import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

print(transformers.__version__)

# --

torch.backends.cudnn.deterministic = True  # used for Reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EPOCHS = 3

# --

url = ('https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz')
filename = url.split('/')[-1]

with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with gzip.open('movie_data.csv.gz', 'rb') as f_in:
    with open('movie_data.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

df = pd.read_csv('movie_data.csv')
print(df.head(3))

# --

train_texts = df.iloc[:35_000]['review'].values
train_labels = df.iloc[:35_000]['sentiment'].values

valid_texts = df.iloc[35_000:40_000]['review'].values
valid_labels = df.iloc[35_000:40_000]['sentiment'].values

test_texts = df.iloc[40_000:]['review'].values
test_labels = df.iloc[40_000:]['sentiment'].values

# --

tokenizer = DistilBertTokenizerFast.from_pretrained(
    'distilbert-base-uncased'
)

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)


# --

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        '''
        encoding.items() ->
          -> input_ids : [1,34, 32, 67,...]
          -> attention_mask : [1,1,1,1,1,....]
        '''
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len((self.labels))


# --

# datasets
train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# dataloaders
bs = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

# --

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased'
)
model.to(device)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)


# --

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            ## prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)
            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()
    return correct_pred.float() / num_examples * 100


# --

start_time = time.time()

for epoch in range(N_EPOCHS):
    model.train()

    for batch_idx, batch in enumerate(train_loader):

        ## prepare data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        ## forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        ## backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        ## logging
        if not batch_idx % 250:
            print(f'Epoch : {epoch + 1}/{N_EPOCHS:04d}'
                  f' | Batch'
                  f'{batch_idx:04d}/'
                  f'{len(train_loader):04d} |'
                  f'Loss: {loss:.4f}')

        model.eval()

        with torch.set_grad_enabled(False):
            print(f'Training accuracy: '
                  f'{compute_accuracy(model, train_loader, device):.2f}%'
                  f'\nValid accuracy: '
                  f'{compute_accuracy(model, valid_loader, device):.2f}%')

    print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
print(f'Test Accuracy: {compute_accuracy(model, test_loader, device):.2f}%')
