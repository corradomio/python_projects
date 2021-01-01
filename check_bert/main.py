from pprint import pprint
from pandas import read_csv


training_data = read_csv("https://thigm85.github.io/data/cord19/cord19-query-title-label.csv")
pprint(training_data.head())

pprint(len(training_data["query"].unique()))

pprint(training_data[["title", "label"]].groupby("label").count())

from sklearn.model_selection import train_test_split
train_queries, val_queries, train_docs, val_docs, train_labels, val_labels = train_test_split(
    training_data["query"].tolist(),
    training_data["title"].tolist(),
    training_data["label"].tolist(),
    test_size=.2
)

from transformers import BertTokenizerFast

model_name = "google/bert_uncased_L-4_H-512_A-8"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(train_queries, train_docs, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_queries, val_docs, truncation=True, padding='max_length', max_length=128)

import torch

class Cord19Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Cord19Dataset(train_encodings, train_labels)
val_dataset = Cord19Dataset(val_encodings, val_labels)


from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(model_name)

for param in model.base_model.parameters():
    param.requires_grad = False


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
)


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()

from torch.onnx import export

device = torch.device("cuda")

model_onnx_path = "model.onnx"
dummy_input = (
    train_dataset[0]["input_ids"].unsqueeze(0).to(device),
    train_dataset[0]["token_type_ids"].unsqueeze(0).to(device),
    train_dataset[0]["attention_mask"].unsqueeze(0).to(device)
)
input_names = ["input_ids", "token_type_ids", "attention_mask"]
output_names = ["logits"]
export(
    model, dummy_input, model_onnx_path, input_names = input_names,
    output_names = output_names, verbose=False, opset_version=11
)

