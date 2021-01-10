from transformers import pipeline
import transformers as tr

print(pipeline('sentiment-analysis')('we hate and love you'))

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence = "A Titan RTX has 24GB of VRAM"
tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)

inputs = tokenizer(sequence)
encoded_sequence = inputs["input_ids"]
print(encoded_sequence)

decoded_sequence = tokenizer.decode(encoded_sequence)
print(decoded_sequence)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]