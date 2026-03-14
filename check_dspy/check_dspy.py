import stdlib.jsonx as jsonx
import dspy
from pprint import pprint

OPENAI_API_KEY = jsonx.load(r"D:\Projects.github\python_projects\api_key_tokens.json")["open_ai_2"]


def main():
    # lm = dspy.LM("openai/gpt-5-mini", api_key=OPENAI_API_KEY)
    lm = dspy.LM("lm_studio/phi4-reasoning", api_base="http://127.0.0.1:1234/v1", api_key="")
    # lm = dspy.LM("ollama/phi4-reasoning", api_base="http://127.0.0.1:11434", api_key=None)
    dspy.configure(lm=lm)

    math = dspy.ChainOfThought("question -> answer: float")
    response = math(question="Two dice, the first one with 20 faces and the second one with minus 17 faces, are tossed. What is the probability that the sum equals two?")
    pprint(response)
# end


if __name__ == "__main__":
    main()
