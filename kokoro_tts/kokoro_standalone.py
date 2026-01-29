#!/usr/bin/env python3
"""
Kokoro TTS - Complete Standalone Version with Web UI
Combines CLI and Web UI functionality in a single file
"""

# Standard library imports
import os
import sys
import tempfile
import time
import io
import threading

# Flask imports (conditionally imported)
try:
    from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Web UI will not work. Install with: pip install flask")

# Third-party imports
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np

# Voice configuration
VOICES = ['af_sarah', 'af_bella', 'af_nicole', 'af_sky', 'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis']
LANGUAGES = ['en-us', 'en-gb', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'zh', 'ko']

def create_web_ui():
    """Create and configure Flask web UI."""
    if not FLASK_AVAILABLE:
        print("Error: Flask is not installed. Install with: pip install flask")
        sys.exit(1)
    
    # Find templates directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    template_folder = os.path.join(parent_dir, 'templates')
    
    if not os.path.exists(template_folder):
        print(f"Error: Templates directory not found at {template_folder}")
        sys.exit(1)
    
    app = Flask(__name__, template_folder=template_folder)
    
    # Find model files in parent directory
    model_path = os.path.join(parent_dir, 'kokoro-v1.0.onnx')
    voices_path = os.path.join(parent_dir, 'voices-v1.0.bin')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(voices_path):
        print(f"Error: Voices file not found at {voices_path}")
        sys.exit(1)
    
    # Initialize Kokoro model
    kokoro = Kokoro(model_path, voices_path)
    
    @app.route('/favicon.ico')
    def favicon():
        return '', 204
    
    @app.route('/')
    def index():
        return render_template('index.html', voices=VOICES, languages=LANGUAGES)
    
    @app.route('/synthesize', methods=['POST'])
    def synthesize():
        """Generate speech from text."""
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'af_sarah')
        language = data.get('language', 'en-us')
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        try:
            # Generate audio
            samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=language)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, samples, sample_rate)
                temp_path = f.name
            
            return send_file(temp_path, mimetype='audio/wav', as_attachment=True, download_name='output.wav')
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    """Main entry point."""
    # If no arguments provided, launch web UI
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ['--web', 'web', 'ui', '--ui']):
        if not FLASK_AVAILABLE:
            print("Error: Flask is required for web UI. Install with: pip install flask")
            sys.exit(1)
        
        app = create_web_ui()
        print("\n" + "="*60)
        print("🎙️  Kokoro TTS Web UI")
        print("="*60)
        print(f"\n✓ Server starting on http://localhost:5000")
        print(f"✓ Open your browser and navigate to the URL above")
        print(f"✓ Press Ctrl+C to stop the server\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Error: CLI mode not implemented in standalone version.")
        print("Usage: python kokoro_standalone.py")
        print("       This will launch the web UI on http://localhost:5000")
        sys.exit(1)

if __name__ == '__main__':
    main()
