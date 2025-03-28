"""
# Trelis Orpheus Inference
Built on by Trelis Research (https://trelis.com) from an original notebook by Canopy Labs
"""

import os
from snac import SNAC
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import soundfile as sf
import librosa
from huggingface_hub import snapshot_download
import IPython.display as ipd

# Model configuration
my_model_name = "rohan2710/bono-orpheus"
model_name = "canopylabs/orpheus-3b-0.1-ft"


device = "cuda" # swap to "cuda" for Nvidia or "cpu" otherwise

# Installation & Setup
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load SNAC model
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to(device)

# Load tokenizer and model
tokeniser_name = "meta-llama/Llama-3.2-3B-Instruct"

# Download only model config and safetensors
def download_model():
    model_path = snapshot_download(
        repo_id=my_model_name,
        allow_patterns=[
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ],
        ignore_patterns=[
            "optimizer.pt",
            "pytorch_model.bin",
            "training_args.bin",
            "scheduler.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.*"
        ]
    )
    return model_path

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(my_model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Set default prompts and voice
default_prompts = [
    "Hey there, my name is Bono, lap time 34.2, box box, box box",
    # "Lap time 33.9, lap number 29, you are in p4",
    # "Lap time 34.2, lap number 10, kart 24 is behind you"
]

# Format prompts into correct template
def format_prompts(prompts, chosen_voice="Bono"):
    formatted_prompts = [f"{chosen_voice}: " + p for p in prompts]
    
    all_input_ids = []
    for prompt in formatted_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_input_ids.append(input_ids)
    
    start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of text, End of human
    
    all_modified_input_ids = []
    for input_ids in all_input_ids:
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
        all_modified_input_ids.append(modified_input_ids)
    
    all_padded_tensors = []
    all_attention_masks = []
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)
    
    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)
    
    input_ids = all_padded_tensors.to(device)
    attention_mask = all_attention_masks.to(device)
    
    return formatted_prompts, input_ids, attention_mask

# Generate Output
def generate_output(input_ids, attention_mask, model):
    print("*** Model.generate is slow - see vllm implementation on github for realtime streaming and inference")
    print("*** Increase/decrease inference params for more expressive less stable generations")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )
    return generated_ids

# Parse Output as speech
def parse_output_as_speech(generated_ids):
    # Define special tokens used in the model's tokenization
    token_to_find = 128257  # Likely a start-of-speech token
    token_to_remove = 128258  # Likely an end-of-speech token
    
    # Find all indices where the start-of-speech token appears
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    
    # Extract the last occurrence of the start-of-speech token
    if len(token_indices[1]) > 0:
        # Get the index of the last start-of-speech token
        last_occurrence_idx = token_indices[1][-1].item()
        # Crop the tensor to start after this token
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        # If no start-of-speech token is found, use the entire generated tensor
        cropped_tensor = generated_ids
    
    # Create a mask to remove specific tokens (end-of-speech tokens)
    mask = cropped_tensor != token_to_remove
    
    # Process each row of the cropped tensor
    processed_rows = []
    for row in cropped_tensor:
        # Remove end-of-speech tokens from each row
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)
    
    # Prepare to convert tokens to audio codes
    code_lists = []
    for row in processed_rows:
        # Ensure the row length is divisible by 7 (likely related to audio encoding)
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        
        # Subtract a base value from each token (normalization step)
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)
    
    return code_lists

# Function to redistribute audio codes into different layers
def redistribute_codes(code_list, snac_model):
    # Initialize layers for audio code reconstruction
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    # Reorganize codes into specific layers
    for i in range((len(code_list)+1)//7):
        # First layer: first code of each 7-token group
        layer_1.append(code_list[7*i])
        
        # Second layer: second code and fifth code, with offset subtraction
        layer_2.append(code_list[7*i+1]-4096)
        layer_2.append(code_list[7*i+4]-(4*4096))
        
        # Third layer: multiple codes with increasing offsets
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    
    # Convert layers to tensors and move to CPU
    codes = [torch.tensor(layer_1).unsqueeze(0).cpu(),
             torch.tensor(layer_2).unsqueeze(0).cpu(),
             torch.tensor(layer_3).unsqueeze(0).cpu()]
    
    # Move SNAC model to CPU and decode
    snac_model_cpu = snac_model.to('cpu')
    audio_hat = snac_model_cpu.decode(codes)
    return audio_hat

# Generate audio samples
def generate_audio_samples(code_lists, snac_model):
    my_samples = []
    for code_list in code_lists:
        # Convert each code list to an audio sample
        samples = redistribute_codes(code_list, snac_model)
        my_samples.append(samples)
    return my_samples

# Save audio to file
def save_audio_to_file(samples, output_file, sample_rate=24000):
    """Save audio samples to a WAV file"""
    audio_np = samples.detach().squeeze().to("cpu").numpy()
    sf.write(output_file, audio_np, sample_rate)
    print(f"Audio saved to {output_file}")

# Main function to run the entire pipeline
def run_inference(prompts=None, chosen_voice="Bono", output_dir="./output", save_audio=True):
    """
    Run the full Orpheus inference pipeline
    
    Args:
        prompts: List of text prompts to convert to speech
        chosen_voice: Voice to use for synthesis
        output_dir: Directory to save output files
        save_audio: Whether to save audio files
    
    Returns:
        List of audio samples
    """
    if prompts is None:
        prompts = default_prompts
    
    # Create output directory if it doesn't exist
    if save_audio and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Format prompts
    formatted_prompts, input_ids, attention_mask = format_prompts(prompts, chosen_voice)
    
    # Generate output
    generated_ids = generate_output(input_ids, attention_mask, model)
    
    # Parse output as speech
    code_lists = parse_output_as_speech(generated_ids)
    
    # Generate audio samples
    my_samples = generate_audio_samples(code_lists, snac_model)
    
    # Save audio files
    if save_audio:
        for i, samples in enumerate(my_samples):
            output_file = os.path.join(output_dir, f"output_{i}.wav")
            save_audio_to_file(samples, output_file)
    
    # Print prompts
    for i, prompt in enumerate(formatted_prompts):
        print(f"Prompt {i+1}: {prompt}")
    
    return my_samples

if __name__ == "__main__":
    # Example usage
    prompts = [
        "Hey there, my name is Bono, lap time 34.2, box box, box box",
        "Lap time 33.9, lap number 29, you are in p4",
    ]
    
    samples = run_inference(prompts)