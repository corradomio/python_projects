from transformers import pipeline

print(pipeline('sentiment-analysis')('we hate and love you'))

