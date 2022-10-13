from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-2.7b")
msg = generator("Hello, I'm am conscious and")
print(msg)

