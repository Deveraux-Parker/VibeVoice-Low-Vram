#!/usr/bin/env python
"""
Quantize and save VibeVoice model using bitsandbytes
Creates a pre-quantized model that can be shared and loaded directly
"""

import os
import json
import shutil
import torch
from pathlib import Path
from transformers import BitsAndBytesConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging
from safetensors.torch import save_file

logging.set_verbosity_info()

def quantize_and_save_model(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    quant_type: str = "nf4"
):
    """Quantize VibeVoice model and save it for distribution"""
    
    print(f"\n{'='*70}")
    print(f"VIBEVOICE QUANTIZATION - {bits}-bit ({quant_type})")
    print(f"{'='*70}")
    print(f"Source: {model_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Configure quantization
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=quant_type
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
    
    print("🔧 Loading and quantizing model...")
    
    # Load the model with quantization
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    
    # Get memory usage
    memory_gb = torch.cuda.memory_allocated() / 1e9
    print(f"💾 Quantized model memory usage: {memory_gb:.1f} GB")
    
    # Save the quantized model
    print("\n📦 Saving quantized model...")
    
    # Method 1: Try using save_pretrained with quantization info
    try:
        # Save model with quantization config
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        # Save the quantization config separately
        quant_config_dict = {
            "quantization_config": bnb_config.to_dict(),
            "quantization_method": "bitsandbytes",
            "bits": bits,
            "quant_type": quant_type
        }
        
        with open(output_path / "quantization_config.json", 'w') as f:
            json.dump(quant_config_dict, f, indent=2)
            
        print("✅ Model saved with integrated quantization")
        
    except Exception as e:
        print(f"⚠️ Standard save failed: {e}")
        print("Trying alternative save method...")
        
        # Method 2: Save state dict with quantized weights
        save_quantized_state_dict(model, output_path, bnb_config)
    
    # Copy processor files
    print("\n📋 Copying processor files...")
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    processor.save_pretrained(output_path)
    
    # Copy additional config files
    for file in ["config.json", "generation_config.json"]:
        src = Path(model_path) / file
        if src.exists():
            shutil.copy2(src, output_path / file)
    
    # Update config to indicate quantization
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["quantization_config"] = bnb_config.to_dict()
        config["_quantization_method"] = "bitsandbytes"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    print(f"\n✅ Quantized model saved to: {output_path}")
    
    # Create loading script
    create_loading_script(output_path, bits, quant_type)
    
    return output_path

def save_quantized_state_dict(model, output_path, bnb_config):
    """Alternative method to save quantized weights"""
    print("\n🔧 Saving quantized state dict...")
    
    # Get the state dict
    state_dict = model.state_dict()
    
    # Separate quantized and non-quantized parameters
    quantized_state = {}
    metadata = {
        "quantized_modules": [],
        "quantization_config": bnb_config.to_dict()
    }
    
    for name, param in state_dict.items():
        # Check if this is a quantized parameter
        if hasattr(param, 'quant_state'):
            # Store quantization state
            metadata["quantized_modules"].append(name)
            quantized_state[name] = param.data
        else:
            # Regular parameter
            quantized_state[name] = param
    
    # Save using safetensors
    save_file(quantized_state, output_path / "model.safetensors", metadata=metadata)
    
    # Save metadata
    with open(output_path / "quantization_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def create_loading_script(output_path, bits, quant_type):
    """Create a script to load the quantized model"""
    
    script_content = f'''#!/usr/bin/env python
"""
Load and use the {bits}-bit quantized VibeVoice model
"""

import torch
from transformers import BitsAndBytesConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def load_quantized_model(model_path="{output_path}"):
    """Load the pre-quantized VibeVoice model"""
    
    print("Loading {bits}-bit quantized VibeVoice model...")
    
    # The model is already quantized, but we need to specify the config
    # to ensure proper loading of quantized weights
    bnb_config = BitsAndBytesConfig(
        load_in_{bits}bit=True,
        bnb_{bits}bit_compute_dtype=torch.bfloat16,
        {"bnb_4bit_use_double_quant=True," if bits == 4 else ""}
        {"bnb_4bit_quant_type='" + quant_type + "'" if bits == 4 else ""}
    )
    
    # Load processor
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    
    # Load model
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"💾 Memory usage: {{torch.cuda.memory_allocated() / 1e9:.1f}} GB")
    
    return model, processor

# Example usage
if __name__ == "__main__":
    model, processor = load_quantized_model()
    
    # Generate audio
    text = "Speaker 1: Hello! Speaker 2: Hi there!"
    inputs = processor(
        text=[text],
        voice_samples=[["path/to/voice1.wav", "path/to/voice2.wav"]],
        padding=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # Save audio
    processor.save_audio(outputs.speech_outputs[0], "output.wav")
'''
    
    script_path = output_path / f"load_quantized_{bits}bit.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"📝 Created loading script: {script_path}")

def test_quantized_model(model_path):
    """Test loading and generating with the quantized model"""
    print(f"\n🧪 Testing quantized model from: {model_path}")
    
    try:
        # Load the quantized model
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        # Load with auto-detection of quantization
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )
        
        print("✅ Model loaded successfully!")
        
        # Quick generation test
        test_text = "Speaker 1: Testing quantized model. Speaker 2: It works!"
        print(f"\n🎤 Testing generation with: '{test_text}'")
        
        # Use demo voices
        voices_dir = "/home/deveraux/Desktop/vibevoice/VibeVoice-main/demo/voices"
        speaker_voices = [
            os.path.join(voices_dir, "en-Alice_woman.wav"),
            os.path.join(voices_dir, "en-Carter_man.wav")
        ]
        
        inputs = processor(
            text=[test_text],
            voice_samples=[speaker_voices],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.3,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
            )
        
        print("✅ Generation successful!")
        
        # Save test output
        output_path = Path(model_path) / "test_output.wav"
        processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
        print(f"🔊 Test audio saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quantize and save VibeVoice model")
    parser.add_argument("--model_path", default="/home/deveraux/Desktop/vibevoice/VibeVoice-Large-pt",
                       help="Path to the original model")
    parser.add_argument("--output_dir", default="/home/deveraux/Desktop/vibevoice/VibeVoice-Large-4bit",
                       help="Output directory for quantized model")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8],
                       help="Quantization bits (4 or 8)")
    parser.add_argument("--quant_type", default="nf4", choices=["nf4", "fp4"],
                       help="4-bit quantization type")
    parser.add_argument("--test", action="store_true",
                       help="Test the quantized model after saving")
    
    args = parser.parse_args()
    
    # Update output dir based on bits
    if str(args.bits) not in args.output_dir:
        args.output_dir = args.output_dir.replace("4bit", f"{args.bits}bit")
    
    # Quantize and save
    output_path = quantize_and_save_model(
        args.model_path,
        args.output_dir,
        args.bits,
        args.quant_type
    )
    
    # Test if requested
    if args.test:
        test_quantized_model(output_path)
    
    print(f"\n🎉 Done! Quantized model ready for distribution at: {output_path}")
    print(f"\n📦 To share this model:")
    print(f"1. Upload the entire '{output_path}' directory")
    print(f"2. Users can load it with the provided script or directly with transformers")
    print(f"3. The model will load in {args.bits}-bit without additional quantization")

if __name__ == "__main__":
    main()