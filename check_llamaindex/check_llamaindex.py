from llama_index.llms.openai import OpenAI

llm = OpenAI(
    # model="gpt-3.5-turbo-1106",
    model="gpt-4.1-nano",
    temperature=0.2,
    max_tokens=50,
    additional_kwargs={
        "seed": 12345678,
        "top_p": 0.5
    }
)
response = llm.complete(
    "Explain the concept of gravity in one sentence"
)
print(response)
