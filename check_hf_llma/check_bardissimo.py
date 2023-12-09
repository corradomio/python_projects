from transformers import pipeline

model = "huggingtweets/bardissimo"

generator = pipeline('text-generation', model=model)
print(generator("My dream is", num_return_sequences=5))
