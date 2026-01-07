"""
RunPod Serverless Handler for Chatterbox TTS
Supports voice cloning via base64 audio reference
With automatic chunking for long text (handles 40s model limit)

v2.0 - Added chunking support
"""

import runpod
import torch
import torchaudio
import base64
import tempfile
import os
import re
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


# ============================================
# CHUNKING FUNCTIONS
# ============================================

def split_into_sentences(text: str) -> list:
    """
    Split text into sentences at natural boundaries.
    Handles common abbreviations to avoid false splits.
    """
    # Protect common abbreviations
    protected = text
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Jr.', 'Sr.', 'vs.', 'etc.', 'i.e.', 'e.g.', 'Prof.', 'Inc.', 'Ltd.']
    for abbr in abbrevs:
        protected = protected.replace(abbr, abbr.replace('.', '<DOT>'))
    
    # Also protect numbers with decimals (e.g., "3.5 billion")
    protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', protected)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', protected)
    
    # Restore protected dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    # Filter empty strings
    return [s.strip() for s in sentences if s.strip()]


def estimate_duration(text: str, min_wpm: float) -> float:
    """
    Estimate audio duration using the slowest (safest) WPM.
    """
    word_count = len(text.split())
    return (word_count / min_wpm) * 60


def chunk_text(text: str, min_wpm: float, chunk_threshold_seconds: float) -> list:
    """
    Split text into chunks that will each be under the threshold duration.
    Splits at sentence boundaries for natural speech.
    
    Args:
        text: Full text to generate
        min_wpm: Slowest measured WPM (from calibration)
        chunk_threshold_seconds: Max seconds per chunk (default 35, model limit is 40)
    
    Returns:
        List of text chunks, each estimated to be under threshold
    """
    # Calculate max words per chunk
    max_words_per_chunk = int((min_wpm * chunk_threshold_seconds) / 60)
    
    print(f"Chunking: max {max_words_per_chunk} words/chunk ({chunk_threshold_seconds}s at {min_wpm} WPM)")
    
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If single sentence exceeds limit, include it anyway with warning
        # (better to let model truncate than skip content)
        if sentence_words > max_words_per_chunk:
            # Finish current chunk first if it has content
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Add oversized sentence as its own chunk
            print(f"WARNING: Sentence exceeds chunk limit ({sentence_words} > {max_words_per_chunk} words)")
            chunks.append(sentence)
            continue
        
        # Would adding this sentence exceed the limit?
        if current_word_count + sentence_words > max_words_per_chunk:
            # Finish current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def concatenate_audio_chunks(audio_tensors: list, sample_rate: int, crossfade_ms: int = 50):
    """
    Concatenate audio chunks with crossfade for smooth transitions.
    
    Args:
        audio_tensors: List of audio tensors from each chunk
        sample_rate: Audio sample rate
        crossfade_ms: Crossfade duration in milliseconds
    
    Returns:
        Single concatenated audio tensor
    """
    if len(audio_tensors) == 1:
        return audio_tensors[0]
    
    crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
    
    if crossfade_samples == 0:
        # Simple concatenation
        return torch.cat(audio_tensors, dim=-1)
    
    # Concatenate with crossfade
    result = audio_tensors[0]
    
    for i, next_audio in enumerate(audio_tensors[1:], 1):
        # Check if chunks are long enough for crossfade
        if result.shape[-1] < crossfade_samples or next_audio.shape[-1] < crossfade_samples:
            # Chunk too short, just concatenate directly
            result = torch.cat([result, next_audio], dim=-1)
            continue
        
        # Create fade curves
        fade_out = torch.linspace(1, 0, crossfade_samples, device=result.device)
        fade_in = torch.linspace(0, 1, crossfade_samples, device=next_audio.device)
        
        # Apply fades to overlapping regions
        result_end = result[..., -crossfade_samples:] * fade_out
        next_start = next_audio[..., :crossfade_samples] * fade_in
        
        # Combine the crossfaded region
        crossfaded = result_end + next_start
        
        # Build final result: everything before crossfade + crossfade + everything after
        result = torch.cat([
            result[..., :-crossfade_samples],
            crossfaded,
            next_audio[..., crossfade_samples:]
        ], dim=-1)
        
        print(f"Crossfaded chunk {i+1}, total samples: {result.shape[-1]}")
    
    return result


def generate_with_chunking(
    tts_model,
    text: str,
    reference_path: str,
    min_wpm: float,
    chunk_threshold_seconds: float,
    temperature: float,
    exaggeration: float,
    cfg_weight: float,
    seed: int = None,
    crossfade_ms: int = 50
):
    """
    Main generation function with automatic chunking for long text.
    
    Returns:
        Tuple of (audio_tensor, sample_rate, chunks_used)
    """
    # Estimate total duration
    word_count = len(text.split())
    estimated_duration = estimate_duration(text, min_wpm)
    
    print(f"=== GENERATE WITH CHUNKING ===")
    print(f"Text: {word_count} words")
    print(f"Estimated duration: {estimated_duration:.1f}s (at {min_wpm} WPM)")
    print(f"Chunk threshold: {chunk_threshold_seconds}s")
    
    # Check if chunking needed
    if estimated_duration <= chunk_threshold_seconds:
        print("No chunking needed - single generation")
        
        if seed is not None:
            torch.manual_seed(int(seed))
        
        audio = tts_model.generate(
            text,
            audio_prompt_path=reference_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )
        
        return audio, tts_model.sr, 1
    
    # Chunking needed
    chunks = chunk_text(text, min_wpm, chunk_threshold_seconds)
    print(f"Split into {len(chunks)} chunks")
    
    audio_tensors = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = len(chunk.split())
        chunk_est = estimate_duration(chunk, min_wpm)
        print(f"  Chunk {i+1}/{len(chunks)}: {chunk_words} words, ~{chunk_est:.1f}s")
        
        # Set seed for first chunk only (subsequent chunks get natural variation)
        if seed is not None and i == 0:
            torch.manual_seed(int(seed))
        
        chunk_audio = tts_model.generate(
            chunk,
            audio_prompt_path=reference_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )
        
        audio_tensors.append(chunk_audio)
        print(f"    Generated: {chunk_audio.shape[-1] / tts_model.sr:.1f}s actual")
    
    # Concatenate all chunks
    print(f"Concatenating {len(audio_tensors)} audio segments with {crossfade_ms}ms crossfade...")
    combined = concatenate_audio_chunks(audio_tensors, tts_model.sr, crossfade_ms)
    
    total_duration = combined.shape[-1] / tts_model.sr
    print(f"Final combined audio: {total_duration:.1f}s")
    
    return combined, tts_model.sr, len(chunks)


# ============================================
# MAIN HANDLER
# ============================================

def handler(event):
    """
    Main handler for RunPod serverless
    
    Input parameters:
    - text (str, required): Text to synthesize
    - reference_audio_base64 (str, required): Base64 encoded voice reference WAV
    - min_wpm (float, required): Minimum WPM from voice calibration - REQUIRED for chunking
    - chunk_threshold_seconds (float, optional): Max seconds before chunking (default from caller)
    - temperature (float, required): Generation temperature
    - exaggeration (float, required): Emotion exaggeration  
    - cfg_weight (float, required): CFG weight for generation
    - seed (int, optional): Random seed for reproducibility
    - crossfade_ms (int, optional): Crossfade duration between chunks (default: 50)
    
    Output:
    - audio_base64 (str): Base64 encoded WAV audio
    - duration_seconds (float): Audio duration
    - sample_rate (int): Audio sample rate
    - chunks_generated (int): Number of chunks used (1 if no chunking needed)
    """
    try:
        # Get input
        job_input = event.get("input", {})
        
        # === REQUIRED PARAMETERS - NO DEFAULTS ===
        text = job_input.get("text")
        if not text:
            return {"error": "Missing required parameter: text"}
        
        reference_audio_b64 = job_input.get("reference_audio_base64")
        if not reference_audio_b64:
            return {"error": "Missing required parameter: reference_audio_base64"}
        
        min_wpm = job_input.get("min_wpm")
        if min_wpm is None:
            return {"error": "Missing required parameter: min_wpm (run voice calibration first)"}
        min_wpm = float(min_wpm)
        
        chunk_threshold_seconds = job_input.get("chunk_threshold_seconds")
        if chunk_threshold_seconds is None:
            return {"error": "Missing required parameter: chunk_threshold_seconds"}
        chunk_threshold_seconds = float(chunk_threshold_seconds)
        
        temperature = job_input.get("temperature")
        if temperature is None:
            return {"error": "Missing required parameter: temperature"}
        temperature = float(temperature)
        
        exaggeration = job_input.get("exaggeration")
        if exaggeration is None:
            return {"error": "Missing required parameter: exaggeration"}
        exaggeration = float(exaggeration)
        
        cfg_weight = job_input.get("cfg_weight")
        if cfg_weight is None:
            return {"error": "Missing required parameter: cfg_weight"}
        cfg_weight = float(cfg_weight)
        
        # === OPTIONAL PARAMETERS ===
        seed = job_input.get("seed")  # None is valid
        crossfade_ms = int(job_input.get("crossfade_ms", 50))  # Operational default OK
        
        # Load model
        tts_model = load_model()
        
        # Decode reference audio
        reference_path = decode_base64_audio(reference_audio_b64)
        print(f"Using voice reference: {reference_path}")
        
        # Generate audio (with automatic chunking if needed)
        print(f"Generating audio for text ({len(text)} chars, {len(text.split())} words)...")
        print(f"Params: temp={temperature}, exag={exaggeration}, cfg={cfg_weight}, min_wpm={min_wpm}")
        
        audio_tensor, sample_rate, chunks_used = generate_with_chunking(
            tts_model=tts_model,
            text=text,
            reference_path=reference_path,
            min_wpm=min_wpm,
            chunk_threshold_seconds=chunk_threshold_seconds,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            crossfade_ms=crossfade_ms
        )
        
        # Calculate duration
        duration_seconds = audio_tensor.shape[-1] / sample_rate
        
        # Encode to base64
        audio_base64 = encode_audio_base64(audio_tensor, sample_rate)
        
        # Cleanup temp file
        if reference_path and os.path.exists(reference_path):
            os.remove(reference_path)
        
        print(f"Generated {duration_seconds:.2f}s of audio ({chunks_used} chunk(s))")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate": sample_rate,
            "chunks_generated": chunks_used
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
