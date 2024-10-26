from pprint import pprint
from transformers import pipeline

classifier = pipeline('sentiment-analysis', device="cuda")

print(classifier('We are very happy to show you the 🤗 Transformers library.'))

results = classifier(["We are very happy to show you the 🤗 Transformers library.",
                      "We hope you don't hate it."])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


result = classifier("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = classifier("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
