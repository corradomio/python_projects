from huggingface_hub import MCPClient

api_keys = load(r"D:\Projects.github\python_projects\api_key_tokens.json")
hf_key = api_keys["hf"]

# Initialize the MCPClient, specifying the server URL or agent ID
# If connecting to an agent on the Hub, you can use its ID.
# For a local Gradio MCP server, you'd use its URL (e.g., "http://localhost:7860/mcp")
client = MCPClient(base_url="https://hf.co/mcp", api_key=hf_key) # or server_url="your-server-url"

# You can then interact with the client to perform actions,
# such as getting available tools or running an agent.
# For example, to get a list of tools:
tools = client.available_tools
print("Available tools:", tools)

# To run an agent with a prompt:
# (This example assumes the agent is set up to handle prompts and use tools)
# from huggingface_hub.mcp import Agent
# agent = Agent(client)
# response = agent.chat("What can you do?")
# print(response)
