# Microplastic Classification API

REST API for microplastic classification using dual models (image + photodiode data).

## Endpoints
- POST /classify - Main classification endpoint
- GET /health - Health check
- GET / - API info

## Input Format
```json
{
  "device_id": "ESP32_MP_001",
  "timestamp": "2025-09-17T14:25:30Z",
  "photodiode_data": [120, 135, 150, 128],
  "image_data": "<base64_encoded_image>"
}