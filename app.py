from flask import Flask, request, jsonify
import base64
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def compute_stats(arr):
    """Compute all required statistics for a 1D numpy array."""
    arr = arr.astype(np.float64)
    
    # Mode: value that appears most often (using histogram binning for floats)
    # For continuous audio data, we bin into 1000 bins
    counts, bin_edges = np.histogram(arr, bins=1000)
    mode_bin_idx = np.argmax(counts)
    mode_val = float((bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2)
    
    unique_vals = np.unique(np.round(arr, 6))
    
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "variance": float(np.var(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mode": mode_val,
        "range": float(np.max(arr) - np.min(arr)),
        "allowed_values": sorted(unique_vals[:50].tolist()),  # first 50 unique values
        "value_range": [float(np.min(arr)), float(np.max(arr))]
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json(force=True)
        audio_id = data.get('audio_id', 'unknown')
        audio_b64 = data.get('audio_base64', '')
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        
        # Try to load with librosa first, fall back to raw PCM
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception:
            try:
                import soundfile as sf
                audio, sr = sf.read(io.BytesIO(audio_bytes))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
            except Exception:
                # Raw PCM fallback (16-bit)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sr = 16000
        
        # Flatten to 1D
        audio = np.array(audio).flatten()
        n = len(audio)
        
        # Column name
        col = "amplitude"
        stats = compute_stats(audio)
        
        # Correlation: autocorrelation at lag 1 (single value for 1 column)
        if n > 1:
            corr_val = float(np.corrcoef(audio[:-1], audio[1:])[0, 1])
            if np.isnan(corr_val):
                corr_val = 0.0
        else:
            corr_val = 0.0
        
        response = {
            "rows": n,
            "columns": [col],
            "mean": {col: stats["mean"]},
            "std": {col: stats["std"]},
            "variance": {col: stats["variance"]},
            "min": {col: stats["min"]},
            "max": {col: stats["max"]},
            "median": {col: stats["median"]},
            "mode": {col: stats["mode"]},
            "range": {col: stats["range"]},
            "allowed_values": {col: stats["allowed_values"]},
            "value_range": {col: stats["value_range"]},
            "correlation": [[1.0, corr_val], [corr_val, 1.0]]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Korean Audio Analysis API is running"})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "Korean Audio Analysis API",
        "endpoint": "/analyze",
        "method": "POST",
        "body": {"audio_id": "q0", "audio_base64": "..."}
    })

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
