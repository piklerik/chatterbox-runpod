"""
RunPod Serverless Handler for Chatterbox TTS
Supports voice cloning via base64 audio reference
"""

import runpod
import torch
import torchaudio
import base64
import tempfile
import os
from io import BytesIO

# Global model - loaded once at worker startup
model = None

def load_model():
    """Load Chatterbox model at startup"""
    global model
    if model is None:
        print("Loading Chatterbox model...")
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cuda")
        print("Model loaded successfully!")
    return model


def decode_base64_audio(base64_string: str) -> str:
    """Decode base64 audio to a temp WAV file, return path"""
    # Handle data URL format: "data:audio/wav;base64,..."
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    audio_bytes = base64.b64decode(base64_string)
    
    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    
    return temp_file.name


def encode_audio_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Encode audio tensor to base64 WAV string"""
    buffer = BytesIO()
    torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:audio/wav;base64,{audio_base64}"


def handler(event):
    """
    Main handler for RunPod serverless
    
    Input parameters:
    - text (str, required): Text to synthesize
    - reference_audio_base64 (str, optional): Base64 encoded voice reference WAV
    - temperature (float, optional): Generation temperature (default: 0.6)
    - exaggeration (float, optional): Emotion exaggeration (default: 0.25)
    - cfg_weight (float, optional): CFG weight for generation (default: 0.3)
    - seed (int, optional): Random seed for reproducibility (default: None)
    
    Output:
    - audio_base64 (str): Base64 encoded WAV audio
    - duration_seconds (float): Audio duration
    """
    try:
        # Get input
        job_input = event.get("input", {})
        
        # Required
        text = job_input.get("text")
        if not text:
            return {"error": "Missing required parameter: text"}
        
        # Optional parameters with defaults
        reference_audio_b64 = job_input.get("reference_audio_base64")
        temperature = float(job_input.get("temperature", 0.6))
        exaggeration = float(job_input.get("exaggeration", 0.25))
        cfg_weight = float(job_input.get("cfg_weight", 0.3))
        seed = job_input.get("seed")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(int(seed))
        
        # Load model
        tts_model = load_model()
        
        # Decode reference audio if provided
        reference_path = None
        if reference_audio_b64:
            reference_path = decode_base64_audio(reference_audio_b64)
            print(f"Using voice reference: {reference_path}")
        
        # Generate audio
        print(f"Generating audio for text ({len(text)} chars)...")
        print(f"Params: temp={temperature}, exag={exaggeration}, cfg={cfg_weight}")
        
        audio_tensor = tts_model.generate(
            text,
            audio_prompt_path=reference_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
        )
        
        # Calculate duration
        sample_rate = tts_model.sr
        duration_seconds = audio_tensor.shape[1] / sample_rate
        
        # Encode to base64
        audio_base64 = encode_audio_base64(audio_tensor, sample_rate)
        
        # Cleanup temp file
        if reference_path and os.path.exists(reference_path):
            os.remove(reference_path)
        
        print(f"Generated {duration_seconds:.2f}s of audio")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate": sample_rate
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error: {error_msg}")
        return {"error": error_msg}


# Pre-load model at worker startup
print("Initializing worker...")
load_model()

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
