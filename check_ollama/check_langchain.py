from stdlib.jsonx import load
from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel



api_keys = load(r"D:\Projects.github\python_projects\api_key_tokens.json")
openai_key = api_keys["open_ai_2"]

# NO: model not supported
# model = OpenAI(model="gpt-3.5-turbo", api_key=openai_key)

# OK
# model = OpenAI(model="gpt-4o-mini", api_key=openai_key)
# print(model.invoke("The sky is"))

# OK
# model = OpenAI(model="airoboros-gpt-3.5-turbo-100k-7b", api_key="not-needed", base_url="http://localhost:1234/v1/")
# print(model.invoke("The sky is"))

# OK
# model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
# prompt = [HumanMessage("What is the capital of France?")]
# print(model.invoke(prompt))

# OK
# model = ChatOpenAI(model="airoboros-gpt-3.5-turbo-100k-7b", api_key="not-needed", base_url="http://localhost:1234/v1/")
# prompt = [HumanMessage("What is the capital of France?")]
# print(model.invoke(prompt))

# OK
# model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
# system_msg = SystemMessage(
# '''You are a helpful assistant that responds to questions with three
# exclamation marks.'''
# )
# human_msg = HumanMessage('What is the capital of France?')
# print(model.invoke([system_msg, human_msg]))

# OK
# model = ChatOpenAI(model="airoboros-gpt-3.5-turbo-100k-7b", api_key="not-needed", base_url="http://localhost:1234/v1/")
# system_msg = SystemMessage(
# '''You are a helpful assistant that responds to questions with three
# exclamation marks.'''
# )
# human_msg = HumanMessage('What is the capital of France?')
# print(model.invoke([system_msg, human_msg]))


# template = PromptTemplate.from_template("""Answer the question based on the
# context below. If the question cannot be answered using the information
# provided, answer with "I don't know".
# Context: {context}
# Question: {question}
# Answer: """)
# print(template.invoke({
# "context": """The most recent advancements in NLP are being driven by Large
# Language Models (LLMs). These models outperform their smaller
# counterparts and have become invaluable for developers who are creating
# applications with NLP capabilities. Developers can tap into these
# models through Hugging Face's `transformers` library, or by utilizing
# OpenAI and Cohere's offerings through the `openai` and `cohere`
# libraries, respectively.""",
# "question": "Which model providers offer LLMs?"
# }))


# OK
# # both `template` and `model` can be reused many times
# template = PromptTemplate.from_template("""Answer the question based on the
# context below. If the question cannot be answered using the information
# provided, answer with "I don't know".
# Context: {context}
# Question: {question}
# Answer: """)
# model = OpenAI(model="gpt-4o-mini", api_key=openai_key)
# # `prompt` and `completion` are the results of using template and model once
# prompt = template.invoke({
# "context": """The most recent advancements in NLP are being driven by Large
# Language Models (LLMs). These models outperform their smaller
# counterparts and have become invaluable for developers who are creating
# applications with NLP capabilities. Developers can tap into these
# models through Hugging Face's `transformers` library, or by utilizing
# OpenAI and Cohere's offerings through the `openai` and `cohere`
# libraries, respectively.""",
# "question": "Which model providers offer LLMs?"
# })
# completion = model.invoke(prompt)
# print(completion)


# OK
# # both `template` and `model` can be reused many times
# template = PromptTemplate.from_template("""Answer the question based on the
# context below. If the question cannot be answered using the information
# provided, answer with "I don't know".
# Context: {context}
# Question: {question}
# Answer: """)
# model = OpenAI(model="airoboros-gpt-3.5-turbo-100k-7b", api_key="not-needed", base_url="http://localhost:1234/v1/")
# # `prompt` and `completion` are the results of using template and model once
# prompt = template.invoke({
# "context": """The most recent advancements in NLP are being driven by Large
# Language Models (LLMs). These models outperform their smaller
# counterparts and have become invaluable for developers who are creating
# applications with NLP capabilities. Developers can tap into these
# models through Hugging Face's `transformers` library, or by utilizing
# OpenAI and Cohere's offerings through the `openai` and `cohere`
# libraries, respectively.""",
# "question": "Which model providers offer LLMs?"
# })
# completion = model.invoke(prompt)
# print(completion)


class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the
    answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)
print(structured_llm.invoke("""What weighs more, a pound of bricks or a pound of feathers"""))

