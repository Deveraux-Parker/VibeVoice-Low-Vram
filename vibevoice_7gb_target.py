#!/usr/bin/env python
"""
Load VibeVoice 4-bit in ~7GB VRAM
Minimize PyTorch's memory pool overhead
"""

import os
import gc
import torch
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

# CRITICAL: Set these BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# Reduce memory fraction to force PyTorch to be more conservative
torch.cuda.set_per_process_memory_fraction(0.75)  # This limits reserved memory

def get_memory_stats():
    """Get detailed memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = torch.cuda.mem_get_info()[0] / 1e9
        total = torch.cuda.mem_get_info()[1] / 1e9
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'used': total - free
        }
    return {}

def load_model_minimal(model_path):
    """Load model with absolute minimal memory overhead"""
    print("Loading 4-bit model with minimal overhead...")
    
    # Start clean
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Report initial state
    stats = get_memory_stats()
    print(f"\nInitial state:")
    print(f"  GPU total: {stats['total']:.2f} GB")
    print(f"  GPU used:  {stats['used']:.2f} GB")
    print(f"  GPU free:  {stats['free']:.2f} GB")
    
    # Load processor
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    
    # Load model - let it use default device map
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Immediately set to eval and disable gradients
    model.eval()
    model.requires_grad_(False)
    
    # Force cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Report after loading
    stats = get_memory_stats()
    print(f"\nAfter loading:")
    print(f"  Allocated: {stats['allocated']:.2f} GB (actual model)")
    print(f"  Reserved:  {stats['reserved']:.2f} GB (PyTorch total)")
    print(f"  Overhead:  {stats['reserved'] - stats['allocated']:.2f} GB")
    print(f"  System reports: {stats['used']:.2f} GB used")
    
    return model, processor

def generate_minimal(model, processor, text, speaker_voices):
    """Generate with minimal memory overhead"""
    # Process inputs
    inputs = processor(
        text=[text],
        voice_samples=[speaker_voices],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Disable caching to save memory during generation
    with torch.no_grad():
        # Temporarily reduce memory fragmentation
        torch.cuda.empty_cache()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=1.3,
            tokenizer=processor.tokenizer,
            generation_config={
                'do_sample': False,
                'use_cache': True,  # Actually, keeping cache can be more efficient
            },
        )
    
    # Cleanup
    del inputs
    gc.collect()
    
    return outputs

def try_memory_reduction_tricks():
    """Additional tricks to reduce memory"""
    print("\n🔧 Applying memory reduction tricks...")
    
    # 1. Reduce CUDA kernel reservation
    if hasattr(torch.cuda, 'set_allocator_settings'):
        torch.cuda.set_allocator_settings(backend='native')
    
    # 2. Force synchronization and cleanup
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # 3. Try to release unused cached blocks
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    
    # This might help
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()
    
    if reserved_before > reserved_after:
        print(f"  ✓ Freed {(reserved_before - reserved_after) / 1e9:.2f} GB")

def main():
    # Paths
    model_path = "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-4bit"
    voices_dir = "/home/deveraux/Desktop/vibevoice/VibeVoice-main/demo/voices"
    
    print("="*60)
    print("VIBEVOICE 4-BIT - 7GB TARGET MODE")
    print("="*60)
    
    # Apply tricks before loading
    try_memory_reduction_tricks()
    
    # Load model
    model, processor = load_model_minimal(model_path)
    
    # Try to compact memory after loading
    try_memory_reduction_tricks()
    
    # Test generation
    test_text = "Speaker 1: Testing minimal memory. Speaker 2: Hope it works!"
    speaker_voices = [
        os.path.join(voices_dir, "en-Alice_woman.wav"),
        os.path.join(voices_dir, "en-Carter_man.wav")
    ]
    
    print("\n🎤 Generating audio...")
    outputs = generate_minimal(model, processor, test_text, speaker_voices)
    
    # Final stats
    stats = get_memory_stats()
    print(f"\nFinal memory usage:")
    print(f"  Allocated: {stats['allocated']:.2f} GB")
    print(f"  Reserved:  {stats['reserved']:.2f} GB")
    print(f"  Total used: {stats['used']:.2f} GB")
    
    # Save output
    output_path = "7gb_target_output.wav"
    processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
    print(f"\n✅ Audio saved to: {output_path}")
    
    # Analysis
    print("\n📊 Analysis:")
    overhead = stats['reserved'] - stats['allocated']
    print(f"The {overhead:.2f} GB overhead comes from:")
    print("- PyTorch memory pool fragmentation")
    print("- CUDA kernel workspace")
    print("- Temporary buffers for operations")
    print("\n💡 The model IS 6.6GB, but PyTorch needs workspace!")
    
    # Extreme option
    print("\n🚀 To truly get to 7GB total, you could:")
    print("1. Use bnb 3-bit quantization (experimental)")
    print("2. Prune some model layers")
    print("3. Use a custom CUDA allocator")
    print("4. Compile with torch.compile() for memory efficiency")

if __name__ == "__main__":
    main()