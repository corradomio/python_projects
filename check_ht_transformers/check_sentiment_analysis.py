from transformers import pipeline

classifier = pipeline("sentiment-analysis")

c = classifier("We are very happy to show you the 🤗 Transformers library.")

print(c)
