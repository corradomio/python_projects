from stdlib.jsonx import load
import openai


def main():
    api_keys = load(r"D:\Projects.github\python_projects\api_key_tokens.json")
    openai_key = api_keys["open_ai_3"]

    client = openai.OpenAI(api_key=openai_key)

    # Define the prompt for text generation
    prompt = "Write a short story about a robot who discovers emotions."

    try:
        # Create a chat completion request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or another suitable model like "gpt-4"
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
