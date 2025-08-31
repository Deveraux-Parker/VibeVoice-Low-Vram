These are some efforts toward running VibeVoice on lower vram systems.

I've put up pre-quanitzed versions of the VibeVoice here:
https://huggingface.co/DevParker/VibeVoice7b-low-vram

The 4 bit is about 6.6 gigabytes. 8 bit is about 10.6 gigabytes. You -should- be able to cram the 8 bit into 12gb vram as long as you're not running much overhead (GUI etc). It might be possible to barely fit and run the 4 bit on 8gb vram if you run almost headless).

I've provided a test_vram_minimal_overhead.py file where I experiment with trying to run these with minimal GPU. Feel free to try it out (it tests 16 bit, 8 bit, and 4 bit in sequence, so make sure you have all three).

In any of thise code files, make sure you modify your model_path to match where your models are.


Quick rundown:
vibevoice_generate_4bit.py
Takes the full 7b vibevoice model found at https://huggingface.co/WestZhang/VibeVoice-Large-pt and runs it in nf4 quant.
Roughly 10gb vram required, slower than the 16 bit model. No need to quantize the model itself.

If you want to quantize the actual model and produce a smaller model, I included quantize_and_save_vibevoice.py. To use it:

# VibeVoice Quantization Guide

Successfully quantized VibeVoice 7B model to both 4-bit and 8-bit versions using bitsandbytes!

## Model Sizes

| Model Version | Size | Memory Usage | Quality |
|---------------|------|--------------|---------|
| Original (fp16/bf16) | 18GB | ~18GB VRAM | Best |
| 8-bit Quantized | 9.9GB | ~10.6GB VRAM | Excellent |
| 4-bit Quantized (nf4) | 6.2GB | ~6.6GB VRAM | Very Good |

## How to Use Pre-Quantized Models

### 1. Loading 4-bit Model

```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# Load pre-quantized 4-bit model
model_path = "/path/to/VibeVoice-Large-4bit"
processor = VibeVoiceProcessor.from_pretrained(model_path)
model = VibeVoiceForConditionalGenerationInference.from_pretrained(
    model_path,
    device_map='cuda',
    torch_dtype=torch.bfloat16,
)
```

### 2. Loading 8-bit Model

```python
# Same code, just point to 8-bit model
model_path = "/path/to/VibeVoice-Large-8bit"
# ... rest is the same
```

## Creating Your Own Quantized Models

Use the provided script to quantize models:

```bash
# 4-bit quantization (nf4)
python quantize_and_save_vibevoice.py \
    --model_path /path/to/original/model \
    --output_dir /path/to/output/4bit \
    --bits 4 \
    --test

# 8-bit quantization
python quantize_and_save_vibevoice.py \
    --model_path /path/to/original/model \
    --output_dir /path/to/output/8bit \
    --bits 8 \
    --test
```

## Benefits

1. **Pre-quantized models load faster** - No on-the-fly quantization needed
2. **Lower VRAM requirements** - 4-bit uses only ~6.6GB vs 18GB
3. **Shareable** - Upload the quantized folder to share with others
4. **Quality preserved** - nf4 quantization maintains excellent output quality

## Distribution

To share quantized models:

1. Upload the entire quantized model directory (e.g., `VibeVoice-Large-4bit/`)
2. Include the `quantization_config.json` file (automatically created)
3. Users can load directly without any quantization setup

## Performance Notes

- 4-bit (nf4): Best for memory-constrained systems, minimal quality loss
- 8-bit: Better quality than 4-bit, still significant memory savings
- Both versions maintain the same generation speed as the original
- Flash Attention 2 is supported in all quantized versions

## Troubleshooting

If loading fails:
1. Ensure you have `bitsandbytes` installed: `pip install bitsandbytes`
2. Make sure you're on a CUDA-capable GPU
3. Check that all model files are present in the directory

## Files Created

Each quantized model directory contains:
- `model.safetensors.*` - Quantized model weights
- `config.json` - Model configuration with quantization settings
- `quantization_config.json` - Specific quantization parameters
- `processor/` - Audio processor files
- `load_quantized_Xbit.py` - Example loading script
