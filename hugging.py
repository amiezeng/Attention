import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Hugging Face API key from environment variable
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

# Model ID for llamba-3.2-1b on Hugging Face
huggingface_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

required_files = [
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "eval_results.json",
    "tokenizer.model",
    "tokenizer.json",
    "config.json"
]

# Download model files
for filename in required_files:
    download_location = hf_hub_download(
        repo_id=huggingface_model,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )
    print(f"File downloaded to: {download_location}")

# load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained(huggingface_model, attn_implementation="eager") # Enable trust_remote_code for safetensor
tokenizer = AutoTokenizer.from_pretrained(huggingface_model)

model.eval()

prompt = "tell me a funny programming joke"
inputs = tokenizer(prompt, return_tensors="pt") 

# directly generate output
output = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.4,
    do_sample=True,
    output_attentions=True,
    return_dict_in_generate=True
)

generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
print(generated_text)