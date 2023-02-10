from transformers import pipeline
import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

classifier = pipeline("text-classification")
outputs = classifier(text)
print(pd.DataFrame(outputs))

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
print(pd.DataFrame(outputs))

reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
print(pd.DataFrame([outputs]))

summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=56, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])

