#!/bin/bash

# EQ-bench requirements:
pip install -q tqdm sentencepiece hf_transfer openai scipy torch peft bitsandbytes trl accelerate tensorboardX huggingface_hub anthropic scikit-learn matplotlib google-generativeai
# These are for qwen models:
pip install -q einops transformers_stream_generator==0.0.4 deepspeed tiktoken git+https://github.com/Dao-AILab/flash-attention.git auto-gptq optimum
# These are for uploading results
pip install -q gspread oauth2client firebase_admin
# Install latest transformers from source last
pip install -q git+https://github.com/huggingface/transformers.git