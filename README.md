These are some efforts toward running VibeVoice on lower vram systems.

Quick rundown:
vibevoice_generate_4bit.py
Takes the full 7b vibevoice model found at https://huggingface.co/WestZhang/VibeVoice-Large-pt and runs it in nf4 quant.
Roughly 10gb vram required, slower than the 16 bit model. No need to quantize the model itself.


