from transformers import pipeline

# generator = pipeline('text-generation', model="facebook/opt-1.3b")
# generator = pipeline('text-generation', model="facebook/opt-2.7b")
generator = pipeline('text-generation', model="facebook/opt-6.7b")
resp = generator("Hello, I'm am conscious and")
print(resp)

