#!/usr/bin/env python
"""
Accurate VRAM measurement for VibeVoice models
Shows the difference between allocated vs reserved memory
"""

import os
import gc
import torch
import subprocess
import time
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    if not torch.cuda.is_available():
        return {}
    
    # PyTorch memory stats
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    
    # Get nvidia-smi info
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(','))
            nvidia_used_gb = used / 1024  # Convert MB to GB
            nvidia_total_gb = total / 1024
        else:
            nvidia_used_gb = 0
            nvidia_total_gb = 0
    except:
        nvidia_used_gb = 0
        nvidia_total_gb = 0
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'nvidia_smi': nvidia_used_gb,
        'nvidia_total': nvidia_total_gb
    }

def print_memory_report(label, before, after):
    """Print detailed memory usage report"""
    print(f"\n{label}:")
    print(f"  PyTorch Allocated: {before['allocated']:.2f} GB → {after['allocated']:.2f} GB "
          f"(+{after['allocated'] - before['allocated']:.2f} GB)")
    print(f"  PyTorch Reserved:  {before['reserved']:.2f} GB → {after['reserved']:.2f} GB "
          f"(+{after['reserved'] - before['reserved']:.2f} GB)")
    print(f"  nvidia-smi Total:  {before['nvidia_smi']:.2f} GB → {after['nvidia_smi']:.2f} GB "
          f"(+{after['nvidia_smi'] - before['nvidia_smi']:.2f} GB)")
    print(f"  Memory Overhead:   {after['reserved'] - after['allocated']:.2f} GB (PyTorch cache)")

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force memory pool cleanup
        torch.cuda.reset_peak_memory_stats()

def test_model_memory(model_path, model_name):
    """Test model with detailed memory tracking"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")
    
    # Clear memory and get baseline
    clear_gpu_memory()
    time.sleep(2)  # Let memory settle
    
    baseline = get_gpu_memory_info()
    print(f"\nBaseline GPU Memory:")
    print(f"  PyTorch Allocated: {baseline['allocated']:.2f} GB")
    print(f"  PyTorch Reserved:  {baseline['reserved']:.2f} GB")
    print(f"  nvidia-smi Shows:  {baseline['nvidia_smi']:.2f} GB / {baseline['nvidia_total']:.2f} GB")
    
    # Load model
    print(f"\nLoading {model_name}...")
    load_start = time.time()
    
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    load_time = time.time() - load_start
    
    # Get memory after loading
    loaded = get_gpu_memory_info()
    print_memory_report("After Model Loading", baseline, loaded)
    
    # Test generation to see peak usage
    print(f"\nTesting generation...")
    test_text = "Speaker 1: Testing memory usage. Speaker 2: Let's see the results!"
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
    
    # Monitor during generation
    pre_gen = get_gpu_memory_info()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
        )
    
    post_gen = get_gpu_memory_info()
    print_memory_report("During Generation", pre_gen, post_gen)
    
    # Peak memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nPeak Memory Usage:")
        print(f"  Peak Allocated: {peak_memory:.2f} GB")
        print(f"  Peak Reserved:  {peak_reserved:.2f} GB")
    
    # Clean up
    del model
    del processor
    clear_gpu_memory()
    
    return {
        'name': model_name,
        'allocated': loaded['allocated'] - baseline['allocated'],
        'reserved': loaded['reserved'] - baseline['reserved'],
        'nvidia_smi': loaded['nvidia_smi'] - baseline['nvidia_smi'],
        'peak_allocated': peak_memory,
        'peak_reserved': peak_reserved
    }

def main():
    print("="*70)
    print("ACCURATE VRAM MEASUREMENT FOR VIBEVOICE")
    print("="*70)
    print("\nNote: PyTorch reserves extra memory for efficiency.")
    print("nvidia-smi shows total reserved memory, not just allocated.")
    
    models = [
        {
            "path": "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-pt",
            "name": "16-bit Original"
        },
        {
            "path": "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-4bit",
            "name": "4-bit Quantized"
        }
    ]
    
    results = []
    for model_info in models:
        try:
            result = test_model_memory(model_info["path"], model_info["name"])
            results.append(result)
            time.sleep(5)
        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("MEMORY USAGE SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Allocated':<12} {'Reserved':<12} {'nvidia-smi':<12} {'Peak':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['allocated']:<12.2f} "
              f"{r['reserved']:<12.2f} "
              f"{r['nvidia_smi']:<12.2f} "
              f"{r['peak_allocated']:<12.2f}")
    
    print("\n💡 Key Insights:")
    print("- 'Allocated' = Actual model weights in memory")
    print("- 'Reserved' = Total GPU memory reserved by PyTorch (includes cache)")
    print("- 'nvidia-smi' = What nvidia-smi reports (includes all overhead)")
    print("- The difference is PyTorch's memory pool for efficiency")

if __name__ == "__main__":
    main()