import lmstudio as lms

model = lms.llm("llama-3.2-1b-instruct")
result = model.respond("What is the meaning of life?")

print(result)


