import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

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

# turn off dropout 
model.eval()

prompt = "tell me a funny programming joke"
inputs = tokenizer(prompt, return_tensors="pt") 

# directly generate output
output = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.2,
    do_sample=True,
    output_attentions=True,
    return_dict_in_generate=True
)

generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
print(generated_text)

# get attention data, clean it up
num_steps = len(output.attentions)
num_layers = len(output.attentions[0]) # length is num of tokens generated / num steps model took

print(f"\nGenerated {num_steps} tokens")
print(f"Model has {num_layers} layers")
if num_steps > 1:
    print(f"Attention shape at step 1, layer 0: {output.attentions[1][0].shape}")

def get_attention_for_step(outputs, step, layer=None, head=None):
    step_attentions = outputs.attentions[step]
    if layer is not None:
        attention_layer = step_attentions[layer]
    else:
        stacked = torch.stack(step_attentions, dim=0)
        attention_layer = stacked.mean(dim=0)

    attention_layer = attention_layer[0]

    if head is not None:
        return attention_layer[head]
    return attention_layer.mean(dim=0)

if num_steps > 5:
    attention_map = get_attention_for_step(output, step=5, layer=num_layers - 1)
    print(f"\nAttention map for step 5 gen: {attention_map.shape}")

    len = inputs['input_ids'].shape[1]
    token_ids = output.sequences[0][:len + 5]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # pull weights and print table
    weights = attention_map[0].tolist()
    print("\n attention weight from generated token at step 5: ")
    for token, weight in zip(tokens, weights):
        print(f"   {token!r:>15}: {weight:.4f}")