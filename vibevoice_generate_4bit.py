#!/usr/bin/env python
"""
Generate audio using 4-bit quantized VibeVoice 7B model
Interactive script for testing different prompts
"""

import os
import time
import torch
from transformers import BitsAndBytesConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

logging.set_verbosity_warning()

class VibeVoice4Bit:
    def __init__(self, model_path):
        """Initialize 4-bit quantized VibeVoice model"""
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load model with 4-bit quantization"""
        print("🚀 Loading VibeVoice 7B in 4-bit...")
        
        # 4-bit config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Load quantized model
        start_time = time.time()
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map='cuda',
            attn_implementation='flash_attention_2',
        )
        self.model.eval()
        
        # Set default diffusion steps
        self.model.set_ddpm_inference_steps(num_steps=20)
        
        load_time = time.time() - start_time
        memory_gb = torch.cuda.memory_allocated() / 1e9
        
        print(f"✅ Model loaded in {load_time:.1f}s")
        print(f"💾 Memory usage: {memory_gb:.1f} GB")
        print(f"⚡ Using Flash Attention 2")
        print()
        
    def generate(self, text, speaker_voices=None, output_name="output", cfg_scale=1.3, diffusion_steps=None):
        """Generate audio from text"""
        
        # Set diffusion steps if specified
        if diffusion_steps:
            self.model.set_ddpm_inference_steps(num_steps=diffusion_steps)
            print(f"Using {diffusion_steps} diffusion steps")
        
        # Default voices if not specified
        if speaker_voices is None:
            voices_dir = "/home/deveraux/Desktop/vibevoice/VibeVoice-main/demo/voices"
            speaker_voices = [
                os.path.join(voices_dir, "en-Alice_woman.wav"),
                os.path.join(voices_dir, "en-Carter_man.wav")
            ]
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            voice_samples=[speaker_voices],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Generate
        print(f"🎙️ Generating audio...")
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )
        
        generation_time = time.perf_counter() - start_time
        
        # Calculate metrics
        sample_rate = 24000
        audio_duration = outputs.speech_outputs[0].shape[-1] / sample_rate
        rtf = generation_time / audio_duration
        
        # Save output
        output_dir = "/home/deveraux/Desktop/vibevoice/outputs/4bit_test"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_name}.wav")
        self.processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
        
        print(f"\n✅ Generation complete!")
        print(f"⏱️ Time: {generation_time:.1f}s")
        print(f"🎵 Duration: {audio_duration:.1f}s")
        print(f"📊 RTF: {rtf:.2f}x")
        print(f"💾 Saved to: {output_path}")
        
        return output_path


def main():
    # Model path
    model_path = "/home/deveraux/Desktop/vibevoice/VibeVoice-Large-pt"
    
    print("="*70)
    print("VIBEVOICE 7B - 4-BIT GENERATION")
    print("="*70)
    
    # Initialize model
    model = VibeVoice4Bit(model_path)
    
    # Test prompts
    test_prompts = [
        {
            "name": "short_conversation",
            "text": "Speaker 1: Hello! How are you today? Speaker 2: I'm doing great, thanks for asking!",
            "steps": 20
        },
        {
            "name": "tech_discussion",
            "text": "Speaker 1: What do you think about the latest AI developments? Speaker 2: It's incredible how fast the field is moving. The capabilities we're seeing now were unimaginable just a few years ago.",
            "steps": 20
        },
        {
            "name": "podcast_intro",
            "text": "Speaker 1: Welcome to our technology podcast! Today we're discussing the future of artificial intelligence. Speaker 2: Thank you for having me! I'm excited to share my insights on this fascinating topic.",
            "steps": 15
        },
        {
            "name": "quick_test",
            "text": "Speaker 1: Testing the 4-bit model. Speaker 2: It sounds great!",
            "steps": 10
        }
    ]
    
    # Generate each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}/{len(test_prompts)}: {prompt['name']}")
        print(f"{'='*70}")
        print(f"Text: {prompt['text']}")
        
        model.generate(
            prompt['text'],
            output_name=prompt['name'],
            diffusion_steps=prompt.get('steps', 20)
        )
    
    # Interactive mode
    print(f"\n{'='*70}")
    print("INTERACTIVE MODE")
    print("='*70}")
    print("Enter your own dialogues! Format:")
    print("Speaker 1: Your first speaker's text")
    print("Speaker 2: Your second speaker's response")
    print("(Type 'quit' to exit)")
    print("='*70}")
    
    while True:
        print("\n")
        user_text = input("Enter dialogue (or 'quit'): ")
        
        if user_text.lower() == 'quit':
            break
            
        if user_text.strip():
            # Ask for diffusion steps
            steps_input = input("Diffusion steps (default 20): ").strip()
            steps = int(steps_input) if steps_input else 20
            
            # Generate
            timestamp = time.strftime("%H%M%S")
            model.generate(
                user_text,
                output_name=f"custom_{timestamp}",
                diffusion_steps=steps
            )


if __name__ == "__main__":
    main()