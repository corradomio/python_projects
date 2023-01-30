import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "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"


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('output/embedded_1k_reviews.csv', index=False)