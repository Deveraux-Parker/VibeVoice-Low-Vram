#!/usr/bin/env python
"""
Simple example of using the pre-quantized VibeVoice model
No need for on-the-fly quantization - loads much faster!
"""

import os
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def main():
    # Path to the pre-quantized model
    model_path = "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-4bit"
    
    print("Loading pre-quantized VibeVoice 4-bit model...")
    
    # Load processor
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    
    # Load the pre-quantized model
    # The quantization config is already saved in the model
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    # Check memory usage
    memory_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✅ Model loaded! Memory usage: {memory_gb:.1f} GB")
    
    # Example generation
    text = "Speaker 1: Welcome to our podcast! Speaker 2: Thanks for having me!"
    
    # Voice samples (using demo voices)
    voices_dir = "/home/deveraux/Desktop/vibevoice/VibeVoice-main/demo/voices"
    speaker_voices = [
        os.path.join(voices_dir, "en-Alice_woman.wav"),
        os.path.join(voices_dir, "en-Carter_man.wav")
    ]
    
    # Process inputs
    inputs = processor(
        text=[text],
        voice_samples=[speaker_voices],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Generate
    print(f"\nGenerating: '{text}'")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
        )
    
    # Save output
    output_path = "quantized_output.wav"
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
    print(f"✅ Audio saved to: {output_path}")

if __name__ == "__main__":
    main()