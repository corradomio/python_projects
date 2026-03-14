import os
from litellm import completion
from pprint import pprint

print("completion")
# response = completion(
#             # model="ollama/llama2",
#             model="ollama/gpt-oss",
#             messages = [{ "content": "Hello, how are you?","role": "user"}],
#             api_base="http://localhost:11434"
# )

# os.environ['LM_STUDIO_API_BASE'] = "http://localhost:1234"
# os.environ['LM_STUDIO_API_KEY'] = ""
response = completion(
            model="lm_studio/gpt-oss-12b",
            messages = [{
                "content": "Hello, how are you?",
                "role": "user"
            }],
            api_base="http://localhost:1234/v1"
)

pprint(response)
