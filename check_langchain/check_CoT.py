#
# https://medium.com/@amit25173/langchain-chain-of-thought-8df031793011
#
from pprint import pprint
from typing import cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

# Initialize the LangChain model
# model = init_chat_model(model="gpt-4.1-nano", model_provider="openai")

# Input text
# input_text: str = "Explain the concept of a neural network."

# Generate response using LangChain
# response: LLMResult = model.generate([[HumanMessage(content=input_text)]])

# Output the response
# pprint(response.generations[0][0].text)

# -------------------------------------------------------

model: ChatOpenAI = cast(ChatOpenAI, init_chat_model(
    model="gpt-4.1-nano", model_provider="openai",
    temperature=0.7,
    max_tokens=150
))

# Input text for a complex query
input_text: str = "Think step by step. Describe the process of training a neural network with backpropagation."

# Generate response with Chain of Thought
response = model.generate([[HumanMessage(content=input_text)]])
print(response)

# Output the response
pprint(response.generations[0][0].text)