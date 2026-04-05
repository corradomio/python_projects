from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv(verbose=True)


tools = {}

# model = "openai:gpt-5"

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)

agent = create_agent(model, tools=tools)

print(agent)

