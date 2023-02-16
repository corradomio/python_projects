from huggingface_hub import notebook_login

notebook_login()

import transformers

print(transformers.__version__)

from transformers.utils import send_example_telemetry

send_example_telemetry("language_modeling_notebook", framework="pytorch")

