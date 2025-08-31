#!/usr/bin/env python
"""
Test actual VRAM usage for 16-bit, 8-bit, and 4-bit models
Using minimal overhead configuration
"""

import os
import gc
import torch
import time
import subprocess
from pathlib import Path
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# Set minimal memory overhead BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of VRAM

def get_nvidia_smi_memory():
    """Get memory usage from nvidia-smi in MB"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0
    except:
        return 0

def get_detailed_memory_stats():
    """Get comprehensive memory statistics"""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    nvidia_mb = get_nvidia_smi_memory()
    nvidia_gb = nvidia_mb / 1024
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'nvidia_smi_gb': nvidia_gb,
        'overhead_gb': reserved - allocated,
        'system_overhead_gb': nvidia_gb - reserved
    }

def clear_all_memory():
    """Aggressively clear all GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(2)  # Let memory settle

def test_model_configuration(model_path, config_name, quantization_config=None):
    """Test a specific model configuration"""
    print(f"\n{'='*70}")
    print(f"Testing {config_name}")
    print(f"{'='*70}")
    
    # Clear memory completely
    clear_all_memory()
    
    # Get baseline
    baseline = get_detailed_memory_stats()
    print(f"\nBaseline:")
    print(f"  nvidia-smi: {baseline['nvidia_smi_gb']:.2f} GB")
    
    # Load model with minimal overhead
    print(f"\nLoading {config_name}...")
    load_start = time.time()
    
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    
    if quantization_config:
        # For on-the-fly quantization
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        # For pre-quantized or full precision
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    
    model.eval()
    model.requires_grad_(False)
    
    load_time = time.time() - load_start
    
    # Force cleanup after loading
    gc.collect()
    torch.cuda.empty_cache()
    
    # Get memory after loading
    loaded = get_detailed_memory_stats()
    
    print(f"\n✅ Model loaded in {load_time:.1f}s")
    print(f"\nMemory Usage:")
    print(f"  Model weights:    {loaded['allocated_gb']:.2f} GB")
    print(f"  PyTorch reserved: {loaded['reserved_gb']:.2f} GB")
    print(f"  nvidia-smi total: {loaded['nvidia_smi_gb']:.2f} GB")
    print(f"  PyTorch overhead: {loaded['overhead_gb']:.2f} GB")
    print(f"  System overhead:  {loaded['system_overhead_gb']:.2f} GB")
    
    # Test generation
    print(f"\nTesting generation...")
    test_text = "Speaker 1: Testing memory usage. Speaker 2: Measuring VRAM consumption."
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
    
    # Clear cache before generation
    torch.cuda.empty_cache()
    pre_gen = get_detailed_memory_stats()
    
    gen_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
        )
    gen_time = time.time() - gen_start
    
    # Get peak usage
    peak = get_detailed_memory_stats()
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\nGeneration completed in {gen_time:.1f}s")
    print(f"Peak allocated:   {peak_allocated:.2f} GB")
    print(f"Peak nvidia-smi:  {peak['nvidia_smi_gb']:.2f} GB")
    
    # Save output
    output_dir = Path("/home/deveraux/Desktop/vibevoice/outputs/vram_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config_name.lower().replace(' ', '_')}_test.wav"
    processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
    
    # Clean up
    del model
    del processor
    del inputs
    del outputs
    clear_all_memory()
    
    return {
        'name': config_name,
        'baseline_nvidia': baseline['nvidia_smi_gb'],
        'allocated': loaded['allocated_gb'],
        'reserved': loaded['reserved_gb'],
        'nvidia_smi': loaded['nvidia_smi_gb'],
        'pytorch_overhead': loaded['overhead_gb'],
        'system_overhead': loaded['system_overhead_gb'],
        'peak_allocated': peak_allocated,
        'peak_nvidia': peak['nvidia_smi_gb'],
        'load_time': load_time,
        'gen_time': gen_time
    }

def main():
    print("="*70)
    print("VRAM USAGE TEST - MINIMAL OVERHEAD")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test configurations
    configs = [
        {
            'path': "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-pt",
            'name': "16-bit Original",
            'quantization': None
        },
        {
            'path': "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-8bit",
            'name': "8-bit Pre-quantized",
            'quantization': None
        },
        {
            'path': "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-4bit",
            'name': "4-bit Pre-quantized",
            'quantization': None
        }
    ]
    
    results = []
    for config in configs:
        try:
            result = test_model_configuration(
                config['path'],
                config['name'],
                config['quantization']
            )
            results.append(result)
            
            # Wait between tests
            print("\n⏳ Waiting before next test...")
            time.sleep(5)
            
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            continue
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY - ACTUAL VRAM USAGE")
    print("="*100)
    print(f"\n{'Model':<20} {'Weights':<10} {'Reserved':<10} {'nvidia-smi':<12} {'Overhead':<20} {'Peak':<10}")
    print("-"*100)
    
    for r in results:
        total_overhead = r['pytorch_overhead'] + r['system_overhead']
        print(f"{r['name']:<20} "
              f"{r['allocated']:<10.2f} "
              f"{r['reserved']:<10.2f} "
              f"{r['nvidia_smi']:<12.2f} "
              f"{total_overhead:<20.2f} "
              f"{r['peak_nvidia']:<10.2f}")
    
    # Detailed breakdown
    print("\n" + "="*100)
    print("OVERHEAD BREAKDOWN")
    print("="*100)
    print(f"\n{'Model':<20} {'PyTorch Cache':<15} {'System/CUDA':<15} {'Total Overhead':<15}")
    print("-"*80)
    
    for r in results:
        total_overhead = r['pytorch_overhead'] + r['system_overhead']
        print(f"{r['name']:<20} "
              f"{r['pytorch_overhead']:<15.2f} "
              f"{r['system_overhead']:<15.2f} "
              f"{total_overhead:<15.2f}")
    
    # Memory savings
    if len(results) > 1:
        print("\n" + "="*100)
        print("MEMORY SAVINGS vs 16-bit")
        print("="*100)
        
        baseline = results[0]
        for r in results[1:]:
            saved = baseline['nvidia_smi'] - r['nvidia_smi']
            percent = (saved / baseline['nvidia_smi']) * 100
            print(f"\n{r['name']}:")
            print(f"  VRAM saved: {saved:.2f} GB ({percent:.1f}%)")
            print(f"  From {baseline['nvidia_smi']:.2f} GB → {r['nvidia_smi']:.2f} GB")

if __name__ == "__main__":
    main()