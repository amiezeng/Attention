# Attention

## Installations:

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install huggingface_hub
pip install transformers
```

### Delete model from computer:

```bash
huggingface-cli scan-cache
huggingface-cli delete-cache
```
