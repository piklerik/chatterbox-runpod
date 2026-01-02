# Chatterbox TTS - RunPod Serverless Worker

High-quality text-to-speech with voice cloning support. Optimized for narration and long-form content.

## Features

- Voice cloning from any WAV reference audio
- Full parameter control (temperature, exaggeration, cfg_weight)
- Returns base64 encoded WAV audio
- Pre-loaded model for faster response times

## Deploy to RunPod

1. Fork this repo to your GitHub account
2. Go to [RunPod Serverless](https://console.runpod.io/serverless)
3. Click **New Endpoint** → **GitHub Repo**
4. Enter your forked repo URL
5. Select GPU type (RTX 4090 recommended for speed)
6. Deploy

First request takes 3-5 minutes (cold start). Subsequent requests: 5-15 seconds.

## API Usage

### Endpoint

```
POST https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run
```

### Headers

```
Content-Type: application/json
Authorization: Bearer {YOUR_RUNPOD_API_KEY}
```

### Request Body

```json
{
  "input": {
    "text": "Your text to synthesize here.",
    "reference_audio_base64": "data:audio/wav;base64,UklGR...",
    "temperature": 0.6,
    "exaggeration": 0.25,
    "cfg_weight": 0.3,
    "seed": 42
  }
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize |
| `reference_audio_base64` | string | No | - | Base64 encoded WAV for voice cloning |
| `temperature` | float | No | 0.6 | Generation randomness (0.4-1.0) |
| `exaggeration` | float | No | 0.25 | Emotion intensity (0.0-1.0) |
| `cfg_weight` | float | No | 0.3 | Classifier-free guidance (0.0-1.0) |
| `seed` | int | No | random | Random seed for reproducibility |

### Response

```json
{
  "id": "job-id",
  "status": "COMPLETED",
  "output": {
    "audio_base64": "data:audio/wav;base64,...",
    "duration_seconds": 12.5,
    "sample_rate": 24000
  }
}
```

## Example: cURL

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "text": "Welcome to the sleep science lab. Tonight we explore the mysteries of deep rest.",
      "temperature": 0.6,
      "exaggeration": 0.25,
      "cfg_weight": 0.3
    }
  }'

# Response: {"id": "abc123", "status": "IN_QUEUE"}

# Poll for result
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/abc123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Example: n8n Integration

```javascript
// HTTP Request node - Submit Job
{
  "method": "POST",
  "url": "https://api.runpod.ai/v2/{{ $json.endpoint_id }}/run",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer {{ $credentials.runpod_api_key }}"
  },
  "body": {
    "input": {
      "text": "{{ $json.script_segment }}",
      "reference_audio_base64": "{{ $json.voice_reference_base64 }}",
      "temperature": 0.6,
      "exaggeration": 0.25,
      "cfg_weight": 0.3
    }
  }
}
```

## Recommended Settings for Sleep Content

For calm, soothing narration:
- `temperature`: 0.5-0.6 (consistent, predictable)
- `exaggeration`: 0.2-0.3 (subtle, not dramatic)
- `cfg_weight`: 0.3 (balanced adherence to reference)

## Costs

RunPod Serverless pricing (~$0.00031/second GPU time):
- Per 20-second audio generation: ~$0.01-0.02
- Per video (150 scenes): ~$1.50-3.00

## Credits

Built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI.
