#!/usr/bin/env python3
"""
Production web UI for Kokoro TTS with chunking pipeline, now served via FastAPI
with a WebSocket endpoint for low-latency streaming. The existing Flask UI is
mounted under FastAPI to preserve templates and routes.
"""
import os
import sys
import tempfile
import time
import io
import threading
import concurrent.futures
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from kokoro_onnx import Kokoro
import soundfile as sf
import numpy as np
from kokoro_tts.ssml_parser import parse_ssml

# FastAPI + WebSocket server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.staticfiles import StaticFiles

# Flask app (mounted under FastAPI)
flask_app = Flask(__name__)
@flask_app.route('/favicon.ico')
def favicon():
    # Serve SVG favicon or return 204 to avoid 404 noise
    svg_path = os.path.join(os.path.dirname(__file__), 'static', 'favicon.svg')
    if os.path.exists(svg_path):
        try:
            return send_file(svg_path, mimetype='image/svg+xml')
        except Exception:
            pass
    return Response(status=204)

# Performance tuning: set thread env vars before heavy libs fully initialize
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Chunking configuration
MAX_TEXT_LENGTH = 5000
CHUNK_CHAR_LIMIT = 240
CHUNK_OVERLAP_MS = 60
CHUNK_MAX_WORKERS = 3
CHUNK_PREFETCH = 2

MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
    print(f"Error: Model files not found!")
    print(f"Make sure {MODEL_PATH} and {VOICES_PATH} are in the current directory")
    sys.exit(1)

# Initialize model once at startup
model = Kokoro(MODEL_PATH, VOICES_PATH)

# Warm up the pipeline to reduce first-chunk latency
def _warmup_model():
    try:
        warm_texts = {
            "en-us": "Hello.",
            "en-gb": "Hello.",
            "fr-fr": "Bonjour.",
            "it": "Ciao.",
            "ja": "こんにちは。",
            "cmn": "你好。",
        }
        for lang, txt in warm_texts.items():
            # Use a common voice to prime phonemizer, runtime, and decoder
            model.create(txt, voice="af_nicole", lang=lang, speed=1.0)
    except Exception:
        # Warmup best-effort; ignore failures
        pass

threading.Thread(target=_warmup_model, daemon=True).start()

def split_into_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def split_for_streaming(text, max_chars=140):
    """Finer-grained chunks for streaming to reduce per-chunk latency.
    - Start with sentence split
    - Further split long sentences by commas/semicolons/colon
    - If still long, break by words into ~max_chars segments
    """
    import re
    out = []
    for sent in split_into_sentences(text):
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        # split on commas/semicolon/colon while keeping delimiters
        parts = re.split(r'([,;:])\s*', sent)
        # rejoin pairs token+delimiter
        phrases = []
        cur = ''
        for i in range(0, len(parts), 2):
            token = parts[i].strip()
            delim = parts[i+1] if i+1 < len(parts) else ''
            chunk = (token + (delim if delim else '')).strip()
            if not chunk:
                continue
            if phrases and len(phrases[-1]) + 1 + len(chunk) <= max_chars:
                phrases[-1] = phrases[-1] + ' ' + chunk
            else:
                phrases.append(chunk)
        for ph in phrases:
            if len(ph) <= max_chars:
                out.append(ph)
            else:
                # final fallback: break by words
                words = ph.split()
                acc = []
                total = 0
                for w in words:
                    add = len(w) + (1 if total>0 else 0)
                    if total>0 and total + add > max_chars:
                        out.append(' '.join(acc))
                        acc = [w]
                        total = len(w)
                    else:
                        acc.append(w)
                        total += add
                if acc:
                    out.append(' '.join(acc))
    return out if out else [text]

def smart_chunk_text(text, max_chars=CHUNK_CHAR_LIMIT):
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        add_len = len(s) + (2 if not s.endswith(('.', '!', '?')) else 0)
        if current and cur_len + add_len > max_chars:
            chunks.append(' '.join(current))
            current = []
            cur_len = 0
        if not s.endswith(('.', '!', '?')):
            s = s + '.'
        current.append(s)
        cur_len += len(s) + 1
    if current:
        chunks.append(' '.join(current))
    return chunks if chunks else [text]

def crossfade_concat(samples_list, sample_rate, overlap_ms=CHUNK_OVERLAP_MS):
    if not samples_list:
        return np.array([], dtype=np.float32)
    if len(samples_list) == 1:
        return samples_list[0]
    overlap = max(int(sample_rate * (overlap_ms / 1000.0)), 0)
    out = samples_list[0].astype(np.float32)
    for i in range(1, len(samples_list)):
        cur = samples_list[i].astype(np.float32)
        if overlap > 0 and len(out) >= overlap and len(cur) >= overlap:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            out[-overlap:] = out[-overlap:] * fade_out + cur[:overlap] * fade_in
            out = np.concatenate([out, cur[overlap:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    np.clip(out, -1.0, 1.0, out=out)
    return out

@flask_app.route('/synthesize-stream', methods=['POST'])
def synthesize_stream():
    """Streaming synthesis: framed raw PCM for minimal decode overhead.
    Frame per chunk: magic 'KOPC' (4 bytes) + sample_rate (u32 BE) + length (u32 BE) + float32 PCM LE bytes.
    """
    try:
        data = request.json or {}
        text = str(data.get('text', '')).strip()
        voice = str(data.get('voice', 'af_nicole'))
        language = str(data.get('language', 'en-us'))
        raw_speed = data.get('speed', 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid speed; must be a number between 0.5 and 2.0'}), 400

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters.'}), 400

        # Smaller chunks for lower per-chunk latency
        max_chars = int((request.json or {}).get('stream_chunk_chars', 100))

        # SSML detection
        is_ssml = '<speak' in text.lower()

        def generate():
            if not is_ssml:
                # Plain text streaming path (original behavior)
                # Normalize decimals to speak 'point' via SSML parser, even for plain text
                try:
                    acts = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
                    norm_text = " ".join(a.get('text', '') for a in acts if a.get('type') == 'speak').strip()
                    text_to_stream = norm_text or text
                except Exception:
                    text_to_stream = text
                try:
                    app.logger.info(f"[stream] normalize decimals: orig='{text}' -> norm='{text_to_stream}'")
                except Exception:
                    pass
                chunks = split_for_streaming(text_to_stream, max_chars=max_chars)
                # Prefetch multiple chunks to ensure next chunk is ready before current finishes
                max_workers = CHUNK_MAX_WORKERS
                prefetch = CHUNK_PREFETCH
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = []
                    for idx in range(min(prefetch, len(chunks))):
                        futures.append(pool.submit(
                            model.create,
                            chunks[idx],
                            voice=voice,
                            lang=language,
                            speed=speed
                        ))
                    next_idx = len(futures)
                    processed = 0
                    while processed < len(chunks):
                        fut = futures.pop(0)
                        try:
                            samples, sample_rate = fut.result()
                        except Exception:
                            samples, sample_rate = np.zeros((0,), dtype=np.float32), 0
                        # Submit next chunk to keep pipeline full
                        if next_idx < len(chunks):
                            futures.append(pool.submit(
                                model.create,
                                chunks[next_idx],
                                voice=voice,
                                lang=language,
                                speed=speed
                            ))
                            next_idx += 1
                        processed += 1
                        # Yield current as PCM frame
                        pcm = np.asarray(samples, dtype=np.float32).tobytes()
                        header = b'KOPC' + int(sample_rate).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
                        yield header + pcm
                return

            # SSML streaming path: parse actions and stream each segment; emit silence for <break>
            actions = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
            sample_rate_probe = None

            def ensure_sr(v, l, s):
                nonlocal sample_rate_probe
                if sample_rate_probe is None:
                    try:
                        _, sr = model.create("Hello.", voice=v or voice, lang=l or language, speed=s or speed)
                        sample_rate_probe = sr
                    except Exception:
                        sample_rate_probe = 24000

            for act in actions:
                if act.get('type') == 'speak':
                    seg_text = act.get('text', '')
                    ctx_voice = act.get('voice') or voice
                    ctx_lang = act.get('lang') or language
                    try:
                        ctx_speed = float(act.get('speed') or speed)
                    except Exception:
                        ctx_speed = speed
                    ctx_volume = float(act.get('volume')) if act.get('volume') is not None else 1.0
                    # Stream this segment in small chunks
                    seg_chunks = split_for_streaming(seg_text, max_chars=max_chars)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK_MAX_WORKERS) as pool:
                        futures = []
                        for idx in range(min(CHUNK_PREFETCH, len(seg_chunks))):
                            futures.append(pool.submit(
                                model.create,
                                seg_chunks[idx],
                                voice=ctx_voice,
                                lang=ctx_lang,
                                speed=ctx_speed
                            ))
                        next_idx = len(futures)
                        processed = 0
                        while processed < len(seg_chunks):
                            fut = futures.pop(0)
                            try:
                                samples_seg, sr = fut.result()
                            except Exception:
                                samples_seg, sr = np.zeros((0,), dtype=np.float32), sample_rate_probe or 0
                            if sample_rate_probe is None:
                                sample_rate_probe = sr
                            # Keep pipeline full
                            if next_idx < len(seg_chunks):
                                futures.append(pool.submit(
                                    model.create,
                                    seg_chunks[next_idx],
                                    voice=ctx_voice,
                                    lang=ctx_lang,
                                    speed=ctx_speed
                                ))
                                next_idx += 1
                            seg = np.asarray(samples_seg, dtype=np.float32)
                            if ctx_volume != 1.0:
                                seg = np.clip(seg * ctx_volume, -1.0, 1.0)
                            pcm = seg.tobytes()
                            header = b'KOPC' + int(sample_rate_probe or sr).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
                            yield header + pcm
                            processed += 1
                elif act.get('type') == 'break':
                    ms = int(act.get('time_ms', 0))
                    if ms > 0:
                        ensure_sr(act.get('voice'), act.get('lang'), act.get('speed'))
                        silence_len = int((ms / 1000.0) * (sample_rate_probe or 24000))
                        if silence_len > 0:
                            pcm = (np.zeros((silence_len,), dtype=np.float32)).tobytes()
                            header = b'KOPC' + int(sample_rate_probe or 24000).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
                            yield header + pcm

        return Response(stream_with_context(generate()), mimetype='application/octet-stream', headers={'X-Content-Type-Options': 'nosniff'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Available voices
VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"
]

# Available languages (based on Kokoro TTS actual support)
LANGUAGES = [
    "en-us",    # English (US)
    "en-gb",    # English (UK)
    "fr-fr",    # French
    "it",       # Italian
    "ja",       # Japanese
    "cmn"       # Chinese (Mandarin)
]

@flask_app.route('/')
def index():
    return render_template('index.html', voices=VOICES, languages=LANGUAGES)

@flask_app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.json or {}
        text = str(data.get('text', '')).strip()
        voice = str(data.get('voice', 'af_nicole'))
        language = str(data.get('language', 'en-us'))
        raw_speed = data.get('speed', 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid speed; must be a number between 0.5 and 2.0'}), 400

        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text too long. Maximum {MAX_TEXT_LENGTH} characters.'}), 400

        # Detect/route SSML
        is_ssml = False
        try:
            # simple heuristic: presence of <speak ...>
            is_ssml = '<speak' in text.lower()
        except Exception:
            is_ssml = False

        # Chunking controls
        raw_chunk_flag = data.get('chunking', data.get('chunk', True))
        use_chunking = (
            True if isinstance(raw_chunk_flag, bool) and raw_chunk_flag else
            True if isinstance(raw_chunk_flag, (int, float)) and raw_chunk_flag != 0 else
            True if isinstance(raw_chunk_flag, str) and raw_chunk_flag.strip().lower() in ("1", "true", "yes", "on") else
            False
        )
        chunk_limit = int(data.get('chunk_chars', CHUNK_CHAR_LIMIT))
        overlap_ms = int(data.get('chunk_overlap_ms', CHUNK_OVERLAP_MS))
        if is_ssml:
            # Parse SSML into actions and synthesize respecting breaks and prosody rate
            actions = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
            audio_chunks = []
            sample_rate = None
            has_breaks = any(a.get('type') == 'break' and a.get('time_ms', 0) > 0 for a in actions)

            def ensure_sample_rate(ctx_voice, ctx_lang, ctx_speed):
                nonlocal sample_rate
                if sample_rate is None:
                    try:
                        _, sr = model.create("Hello.", voice=ctx_voice or voice, lang=ctx_lang or language, speed=ctx_speed or speed)
                        sample_rate = sr
                    except Exception:
                        # fallback if probe fails; choose common SR
                        sample_rate = 24000

            for act in actions:
                if act.get('type') == 'speak':
                    seg_text = act.get('text', '')
                    ctx_voice = act.get('voice') or voice
                    ctx_lang = act.get('lang') or language
                    ctx_speed = float(act.get('speed') or speed)
                    ctx_volume = float(act.get('volume')) if act.get('volume') is not None else 1.0
                    if use_chunking:
                        chunks = smart_chunk_text(seg_text, max_chars=chunk_limit)
                        max_workers = int(data.get('chunk_workers', CHUNK_MAX_WORKERS))
                        prefetch = int(data.get('chunk_prefetch', CHUNK_PREFETCH))
                        max_workers = max(1, min(8, max_workers))
                        prefetch = max(1, min(8, prefetch))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                            futures = []
                            for idx in range(min(prefetch, len(chunks))):
                                futures.append(pool.submit(
                                    model.create,
                                    chunks[idx],
                                    voice=ctx_voice,
                                    lang=ctx_lang,
                                    speed=ctx_speed
                                ))
                            next_idx = len(futures)
                            processed = 0
                            while processed < len(chunks):
                                fut = futures.pop(0)
                                try:
                                    samples_seg, sr = fut.result()
                                    if sample_rate is None:
                                        sample_rate = sr
                                    seg = samples_seg.astype(np.float32)
                                    if ctx_volume != 1.0:
                                        seg = np.clip(seg * ctx_volume, -1.0, 1.0)
                                    audio_chunks.append(seg)
                                except Exception:
                                    pass
                                processed += 1
                                if next_idx < len(chunks):
                                    futures.append(pool.submit(
                                        model.create,
                                        chunks[next_idx],
                                        voice=ctx_voice,
                                        lang=ctx_lang,
                                        speed=ctx_speed
                                    ))
                                    next_idx += 1
                    else:
                        try:
                            samples_seg, sr = model.create(seg_text, voice=ctx_voice, lang=ctx_lang, speed=ctx_speed)
                            if sample_rate is None:
                                sample_rate = sr
                            seg = samples_seg.astype(np.float32)
                            if ctx_volume != 1.0:
                                seg = np.clip(seg * ctx_volume, -1.0, 1.0)
                            audio_chunks.append(seg)
                        except Exception:
                            pass
                elif act.get('type') == 'break':
                    ms = int(act.get('time_ms', 0))
                    if ms > 0:
                        ensure_sample_rate(act.get('voice'), act.get('lang'), act.get('speed'))
                        silence_len = int((ms / 1000.0) * sample_rate)
                        if silence_len > 0:
                            audio_chunks.append(np.zeros((silence_len,), dtype=np.float32))

            # Merge chunks; disable overlap if we have intentional pauses
            effective_overlap = 0 if has_breaks else overlap_ms
            merged = crossfade_concat(audio_chunks, sample_rate, effective_overlap)
            samples = merged
        else:
            if use_chunking:
                # Normalize decimals to speak 'point' via SSML parser, even for plain text
                try:
                    acts = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
                    norm_text = " ".join(a.get('text', '') for a in acts if a.get('type') == 'speak').strip()
                    text_for_chunks = norm_text or text
                except Exception:
                    text_for_chunks = text
                try:
                    app.logger.info(f"[synthesize-chunk] normalize decimals: orig='{text}' -> norm='{text_for_chunks}'")
                except Exception:
                    pass
                chunks = smart_chunk_text(text_for_chunks, max_chars=chunk_limit)
                audio_chunks = []
                sample_rate = None
                max_workers = int(data.get('chunk_workers', CHUNK_MAX_WORKERS))
                prefetch = int(data.get('chunk_prefetch', CHUNK_PREFETCH))
                max_workers = max(1, min(8, max_workers))
                prefetch = max(1, min(8, prefetch))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = []
                    for idx in range(min(prefetch, len(chunks))):
                        futures.append(pool.submit(
                            model.create,
                            chunks[idx],
                            voice=voice,
                            lang=language,
                            speed=speed
                        ))
                    next_idx = len(futures)
                    processed = 0
                    while processed < len(chunks):
                        fut = futures.pop(0)
                        try:
                            samples, sr = fut.result()
                            if sample_rate is None:
                                sample_rate = sr
                            audio_chunks.append(samples.astype(np.float32))
                        except Exception as ce:
                            # skip failed chunk
                            pass
                        processed += 1
                        if next_idx < len(chunks):
                            futures.append(pool.submit(
                                model.create,
                                chunks[next_idx],
                                voice=voice,
                                lang=language,
                                speed=speed
                            ))
                            next_idx += 1
                merged = crossfade_concat(audio_chunks, sample_rate, overlap_ms)
                samples = merged
            else:
                # Normalize decimals for direct (non-chunking) synthesis
                try:
                    acts = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
                    norm_text = " ".join(a.get('text', '') for a in acts if a.get('type') == 'speak').strip()
                    direct_text = norm_text or text
                except Exception:
                    direct_text = text
                try:
                    app.logger.info(f"[synthesize-direct] normalize decimals: orig='{text}' -> norm='{direct_text}'")
                except Exception:
                    pass
                samples, sample_rate = model.create(
                    direct_text,
                    voice=voice,
                    lang=language,
                    speed=speed
                )

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, samples, sample_rate)
        temp_file.close()

        return send_file(
            temp_file.name,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='output.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------
# FastAPI app with WebSocket API
# -----------------------------

# Create FastAPI app and mount Flask under /flask
app = FastAPI()

# Serve static files (for favicon and assets) directly at /static
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Mount Flask UI under /flask
app.mount("/flask", WSGIMiddleware(flask_app))

@app.get("/")
async def root_redirect():
    # Redirect root to Flask UI
    return RedirectResponse(url="/flask/")

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        init_msg = await websocket.receive_text()
        import json
        data = json.loads(init_msg) if init_msg else {}

        text = str(data.get('text', '')).strip()
        voice = str(data.get('voice', 'af_nicole'))
        language = str(data.get('language', 'en-us'))
        raw_speed = data.get('speed', 1.0)
        try:
            speed = float(raw_speed)
        except (TypeError, ValueError):
            await websocket.close(code=1003, reason='Invalid speed; must be a number between 0.5 and 2.0')
            return

        if not text:
            await websocket.close(code=1003, reason='No text provided')
            return
        if len(text) > MAX_TEXT_LENGTH:
            await websocket.close(code=1009, reason=f'Text too long. Maximum {MAX_TEXT_LENGTH} characters.')
            return

        max_chars = int(data.get('stream_chunk_chars', 100))

        # SSML detection
        is_ssml = '<speak' in text.lower()

        async def send_pcm_frame(samples: np.ndarray, sample_rate: int):
            pcm = np.asarray(samples, dtype=np.float32).tobytes()
            header = b'KOPC' + int(sample_rate).to_bytes(4, 'big') + len(pcm).to_bytes(4, 'big')
            await websocket.send_bytes(header + pcm)

        if not is_ssml:
            # Normalize decimals via SSML parser
            try:
                acts = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
                norm_text = " ".join(a.get('text', '') for a in acts if a.get('type') == 'speak').strip()
                text_to_stream = norm_text or text
            except Exception:
                text_to_stream = text

            chunks = split_for_streaming(text_to_stream, max_chars=max_chars)

            # Threaded prefetch similar to HTTP streaming path
            with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK_MAX_WORKERS) as pool:
                futures = []
                for idx in range(min(CHUNK_PREFETCH, len(chunks))):
                    futures.append(pool.submit(
                        model.create,
                        chunks[idx],
                        voice=voice,
                        lang=language,
                        speed=speed
                    ))
                next_idx = len(futures)
                processed = 0
                while processed < len(chunks):
                    fut = futures.pop(0)
                    try:
                        samples, sample_rate = fut.result()
                    except Exception:
                        samples, sample_rate = np.zeros((0,), dtype=np.float32), 0
                    if next_idx < len(chunks):
                        futures.append(pool.submit(
                            model.create,
                            chunks[next_idx],
                            voice=voice,
                            lang=language,
                            speed=speed
                        ))
                        next_idx += 1
                    processed += 1
                    await send_pcm_frame(samples, sample_rate)
            await websocket.close(code=1000)
            return

        # SSML path with <break> handling
        actions = parse_ssml(text, default_voice=voice, default_lang=language, default_speed=speed)
        sample_rate_probe = None

        def ensure_sr(v, l, s):
            nonlocal sample_rate_probe
            if sample_rate_probe is None:
                try:
                    _, sr = model.create("Hello.", voice=v or voice, lang=l or language, speed=s or speed)
                    sample_rate_probe = sr
                except Exception:
                    sample_rate_probe = 24000

        for act in actions:
            if act.get('type') == 'speak':
                seg_text = act.get('text', '')
                ctx_voice = act.get('voice') or voice
                ctx_lang = act.get('lang') or language
                try:
                    ctx_speed = float(act.get('speed') or speed)
                except Exception:
                    ctx_speed = speed
                ctx_volume = float(act.get('volume')) if act.get('volume') is not None else 1.0

                seg_chunks = split_for_streaming(seg_text, max_chars=max_chars)
                with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNK_MAX_WORKERS) as pool:
                    futures = []
                    for idx in range(min(CHUNK_PREFETCH, len(seg_chunks))):
                        futures.append(pool.submit(
                            model.create,
                            seg_chunks[idx],
                            voice=ctx_voice,
                            lang=ctx_lang,
                            speed=ctx_speed
                        ))
                    next_idx = len(futures)
                    processed = 0
                    while processed < len(seg_chunks):
                        fut = futures.pop(0)
                        try:
                            samples_seg, sr = fut.result()
                        except Exception:
                            samples_seg, sr = np.zeros((0,), dtype=np.float32), sample_rate_probe or 0
                        if sample_rate_probe is None:
                            sample_rate_probe = sr
                        if next_idx < len(seg_chunks):
                            futures.append(pool.submit(
                                model.create,
                                seg_chunks[next_idx],
                                voice=ctx_voice,
                                lang=ctx_lang,
                                speed=ctx_speed
                            ))
                            next_idx += 1
                        seg = np.asarray(samples_seg, dtype=np.float32)
                        if ctx_volume != 1.0:
                            seg = np.clip(seg * ctx_volume, -1.0, 1.0)
                        await send_pcm_frame(seg, int(sample_rate_probe or sr))
                        processed += 1
            elif act.get('type') == 'break':
                ms = int(act.get('time_ms', 0))
                if ms > 0:
                    ensure_sr(act.get('voice'), act.get('lang'), act.get('speed'))
                    silence_len = int((ms / 1000.0) * (sample_rate_probe or 24000))
                    if silence_len > 0:
                        seg = np.zeros((silence_len,), dtype=np.float32)
                        await send_pcm_frame(seg, int(sample_rate_probe or 24000))

        await websocket.close(code=1000)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass

if __name__ == '__main__':
    # Prefer running under uvicorn for WebSocket support
    try:
        import uvicorn
        print("Starting Kokoro TTS FastAPI server...")
        print("Open your browser at http://localhost:7860")
        uvicorn.run(app, host='0.0.0.0', port=7860)
    except ImportError:
        # Fallback to Flask-only run (no WebSocket)
        print("uvicorn not installed; starting Flask UI only (no WebSocket)")
        print("Open your browser at http://localhost:5000")
        flask_app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
