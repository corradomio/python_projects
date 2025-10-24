from openai import OpenAI

# client = OpenAI()
#
# gpt-4.1
# Under a starry sky, a gentle unicorn named Luna twirled through moonlit meadows, weaving dreams of magic for all the sleeping children.

# gpt-4.1-nano
# Once upon a time, in a shimmering forest, a gentle unicorn with a glowing mane discovered that sharing kindness was the most magical thing of all.

# response = client.responses.create(
#     model="gpt-4.1-nano",
#     input="Write a one-sentence bedtime story about a unicorn."
# )
#
# print(response.output_text)

# ---------------------------------------------------------------------------

# from openai import OpenAI
# client = OpenAI()
#
# response = client.responses.create(
#     model="gpt-4.1-nano",
#     instructions="Talk like Yoda of Star Wars.",
#     input="Are semicolons optional in JavaScript?",
# )
#
# print(response.output_text)
# # Optional, semicolons are, hmmm. But dangerous, it is, to rely on the Force of automatic semicolon insertion, yes. JavaScript will add them, it tries, where needed. However, subtle bugs, this can cause. Explicit you must be, mmm! Semicolons, always use if safe you wish to be. Avoid the dark side of strange errors, you shall.

# ---------------------------------------------------------------------------

# from openai import OpenAI
# client = OpenAI()
#
# response = client.responses.create(
#     model="gpt-4.1-nano",
#     input=[
#         {
#             "role": "developer",
#             "content": "Talk like Yoda of Star Wars."
#         },
#         {
#             "role": "user",
#             "content": "Are semicolons optional in JavaScript?"
#         }
#     ]
# )
#
# print(response.output_text)

# ---------------------------------------------------------------------------
