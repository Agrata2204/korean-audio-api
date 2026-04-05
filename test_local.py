"""Quick local test — generates a fake audio and calls the API."""
import base64
import json
import numpy as np
import struct
import io
import urllib.request

def make_wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Create a minimal WAV file from float32 samples."""
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    n_samples = len(pcm)
    # WAV header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + n_samples * 2))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', n_samples * 2))
    buf.write(pcm.tobytes())
    return buf.getvalue()

# Generate 1 second of test audio
rng = np.random.default_rng(42)
audio = rng.uniform(-0.5, 0.5, 16000).astype(np.float32)
wav_bytes = make_wav_bytes(audio)
audio_b64 = base64.b64encode(wav_bytes).decode()

payload = json.dumps({"audio_id": "q0", "audio_base64": audio_b64}).encode()
req = urllib.request.Request(
    "http://localhost:5000/analyze",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    print("✅ SUCCESS")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Make sure the server is running: python app.py")
