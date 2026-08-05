from pprint import pprint

from openai import OpenAI

def check_completion(client):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': 'Say this is a test',
            }
        ],
        model='gpt-oss:20b',
    )
    pprint(chat_completion)
    print(chat_completion.choices[0].message.content)


def check_embedding(client):
    responses = client.embeddings.create(
        input=[
            "Hello my name is",
            "The best thing about vLLM is that it supports many different models"
        ],
        model="qwen3-embedding:8b"
    )

    for data in responses.data:
        print(data.embedding)


def main():
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='ollama',  # required but ignored
    )

    check_completion(client)
    check_embedding(client)

    pass


if __name__ == "__main__":
    main()

