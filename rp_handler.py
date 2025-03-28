"""
RunPod Serverless handler for Trelis Orpheus TTS model
"""

import os
import time
import base64
import runpod
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
from huggingface_hub import snapshot_download

# Global variables for model instances
SNAC_MODEL = None
ORPHEUS_MODEL = None
TOKENIZER = None

# Model configuration
MY_MODEL_NAME = os.environ.get("MODEL_NAME", "rohan2710/bono-orpheus")
TOKENISER_NAME = os.environ.get("TOKENISER_NAME", "meta-llama/Llama-3.2-3B-Instruct")
SNAC_MODEL_NAME = os.environ.get("SNAC_MODEL_NAME", "hubertsiuzdak/snac_24khz")
DEVICE = "cuda"  # RunPod provides CUDA-enabled containers

# Enable HF Transfer for better download speeds
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_models():
    """Load all necessary models into global variables"""
    global SNAC_MODEL, ORPHEUS_MODEL, TOKENIZER
    
    print("Loading models...")
    
    # Load SNAC model for audio decoding
    if SNAC_MODEL is None:
        print("Loading SNAC model...")
        SNAC_MODEL = SNAC.from_pretrained(SNAC_MODEL_NAME)
        SNAC_MODEL = SNAC_MODEL.to(DEVICE)
    
    # Load Orpheus model and tokenizer
    if ORPHEUS_MODEL is None or TOKENIZER is None:
        print("Loading Orpheus model and tokenizer...")
        ORPHEUS_MODEL = AutoModelForCausalLM.from_pretrained(
            MY_MODEL_NAME, 
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
        
        TOKENIZER = AutoTokenizer.from_pretrained(MY_MODEL_NAME)
    
    print("All models loaded successfully")
    return SNAC_MODEL, ORPHEUS_MODEL, TOKENIZER

def format_prompt(text, voice="Bono"):
    """Format a single prompt for the model"""
    global TOKENIZER
    
    # Format the prompt with the voice
    formatted_prompt = f"{voice}: {text}"
    
    # Tokenize the input
    input_ids = TOKENIZER(formatted_prompt, return_tensors="pt").input_ids
    
    # Add special tokens
    start_token = torch.tensor([[ 128259]], dtype=torch.int64)  # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
    
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH
    
    # Create attention mask (all 1s since we're not padding a batch)
    attention_mask = torch.ones(modified_input_ids.shape, dtype=torch.int64)
    
    # Move to device
    input_ids = modified_input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    
    return formatted_prompt, input_ids, attention_mask

def generate_speech(input_ids, attention_mask, temperature=0.6, top_p=0.95):
    """Generate speech tokens from text input"""
    global ORPHEUS_MODEL
    
    with torch.no_grad():
        generated_ids = ORPHEUS_MODEL.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258,
        )
    return generated_ids

def parse_output_as_speech(generated_ids):
    """Extract speech codes from model output"""
    # Define special tokens
    token_to_find = 128257  # start-of-speech token
    token_to_remove = 128258  # end-of-speech token
    
    # Find start-of-speech token
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
    
    # Extract after last start-of-speech token
    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids
    
    # Process the first row (we only generate one sequence)
    row = cropped_tensor[0]
    masked_row = row[row != token_to_remove]
    
    # Prepare audio codes
    row_length = masked_row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = masked_row[:new_length]
    
    # Subtract base value
    code_list = [t.item() - 128266 for t in trimmed_row]
    
    return code_list

def redistribute_codes(code_list):
    """Reorganize codes for audio synthesis"""
    global SNAC_MODEL
    
    # Initialize layers
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    # Redistribute codes to layers
    for i in range((len(code_list)+1)//7):
        # First layer
        layer_1.append(code_list[7*i])
        
        # Second layer
        layer_2.append(code_list[7*i+1]-4096)
        layer_2.append(code_list[7*i+4]-(4*4096))
        
        # Third layer
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
    
    # Convert to tensors
    codes = [
        torch.tensor(layer_1).unsqueeze(0).to(DEVICE),
        torch.tensor(layer_2).unsqueeze(0).to(DEVICE),
        torch.tensor(layer_3).unsqueeze(0).to(DEVICE)
    ]
    
    # Decode audio
    audio_hat = SNAC_MODEL.decode(codes)
    return audio_hat

def save_audio_to_file(samples, output_file, sample_rate=24000):
    """Save audio samples to a WAV file"""
    audio_np = samples.detach().squeeze().cpu().numpy()
    sf.write(output_file, audio_np, sample_rate)

def handler(event):
    """
    RunPod handler function for TTS inference
    """
    try:
        # Ensure models are loaded
        load_models()
        
        # Extract input parameters
        input_text = event.get("input", {}).get("text", "")
        if not input_text:
            return {"error": "No text provided for speech synthesis"}
        
        voice = event.get("input", {}).get("voice", "Bono")
        temperature = float(event.get("input", {}).get("temperature", 0.6))
        top_p = float(event.get("input", {}).get("top_p", 0.95))
        
        # Start timing
        start_time = time.time()
        
        # Format prompt
        formatted_prompt, input_ids, attention_mask = format_prompt(input_text, voice)
        
        # Generate speech tokens
        generated_ids = generate_speech(input_ids, attention_mask, temperature, top_p)
        
        # Parse output
        code_list = parse_output_as_speech(generated_ids)
        
        # Generate audio
        audio_samples = redistribute_codes(code_list)
        
        # Save to temporary file
        output_path = f"/tmp/output_{int(time.time())}.wav"
        save_audio_to_file(audio_samples, output_path)
        
        # Convert to base64
        with open(output_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up temp file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            "audio": audio_base64,
            "metadata": {
                "text": input_text,
                "voice": voice,
                "processing_time_seconds": processing_time,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Initialize models when container starts
if __name__ == "__main__":
    load_models()
    runpod.serverless.start({"handler": handler})