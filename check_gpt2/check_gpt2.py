from gpt2_client import GPT2Client

gpt2 = GPT2Client('117M') # This could also be `345M`, `774M`, or `1558M`. Rename `save_dir` to anything.
gpt2.load_model(force_download=False) # Use cached versions if available.


from gpt2_client import GPT2Client

gpt2 = GPT2Client('117M') # This could also be `345M`, `774M`, or `1558M`

gpt2.generate(interactive=True) # Asks user for prompt
gpt2.generate(n_samples=4) # Generates 4 pieces of text
text = gpt2.generate(return_text=True) # Generates text and returns it in an array
gpt2.generate(interactive=True, n_samples=3) # A different prompt each time


from gpt2_client import GPT2Client

gpt2 = GPT2Client('117M') # This could also be `345M`, `774M`, or `1558M`
gpt2.load_model()

# encoding a sentence
encs = gpt2.encode_seq("Hello world, this is a sentence")
# [15496, 995, 11, 428, 318, 257, 6827]

# decoding an encoded sequence
decs = gpt2.decode_seq(encs)
# Hello world, this is a sentence
