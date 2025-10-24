from stdlib.jsonx import load
import openai


def main():
    # api_keys = load(r"D:\Projects.github\python_projects\api_key_tokens.json")
    # openai_key = api_keys["open_ai_3"]

    # lmstudio
    client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    # ollama
    # client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

    # Define the prompt for text generation
    prompt = "Write a short story about a robot who discovers emotions."

    try:
        # Create a chat completion request
        response = client.chat.completions.create(
            model="google/gemma-3-27b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Limit the length of the generated response
            temperature=0.7  # Control the randomness of the output (0.0 for deterministic, 1.0 for highly creative)
        )

        # Extract and print the generated text
        generated_text = response.choices[0].message.content
        print("Generated Story:")
        print(generated_text)

    except openai.APIError as e:
        print(f"OpenAI API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    pass


if __name__ == "__main__":
    main()
