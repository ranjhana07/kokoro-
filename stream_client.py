import argparse
import sys
import time
import json
import struct
import requests
import numpy as np
import sounddevice as sd


MAGIC_LEGACY = b"KOPC"
MAGIC_SESSION = b"KOKO"


def parse_frames_legacy(byte_buffer: bytearray):
    """Yield (sample_rate, pcm_bytes) tuples from legacy framed stream.
    Format: 'KOPC' (4) + sample_rate (u32 BE) + length (u32 BE) + float32 PCM LE bytes.
    Modifies byte_buffer by consuming parsed bytes.
    """
    out = []
    i = 0
    while len(byte_buffer) - i >= 12:
        if byte_buffer[i:i+4] != MAGIC_LEGACY:
            i += 1
            continue
        sr = int.from_bytes(byte_buffer[i+4:i+8], "big")
        length = int.from_bytes(byte_buffer[i+8:i+12], "big")
        if len(byte_buffer) - (i + 12) < length:
            break
        payload = bytes(byte_buffer[i+12:i+12+length])
        out.append((sr, payload))
        i += 12 + length
    if i:
        del byte_buffer[:i]
    return out


def parse_frames_session(byte_buffer: bytearray):
    """Yield (sample_rate, pcm_bytes) tuples from session framed stream.
    Format: 'KOKO'(4)+version(1)+flags(1)+sample_rate(4)+length(4)+timestamp(8)+payload.
    Modifies byte_buffer by consuming parsed bytes.
    """
    out = []
    i = 0
    header_len = 4 + 1 + 1 + 4 + 4 + 8  # 22 bytes
    while len(byte_buffer) - i >= header_len:
        if byte_buffer[i:i+4] != MAGIC_SESSION:
            i += 1
            continue
        try:
            magic, ver, flags, sr, length, ts = struct.unpack_from('4sBBIIQ', byte_buffer, i)
        except struct.error:
            break
        total = header_len + int(length)
        if len(byte_buffer) - i < total:
            break
        start = i + header_len
        payload = bytes(byte_buffer[start:start+length])
        out.append((int(sr), payload))
        i += total
    if i:
        del byte_buffer[:i]
    return out


def _resample_linear(f32: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return f32
    if f32.size == 0:
        return f32
    ratio = dst_sr / float(src_sr)
    new_len = int(round(f32.size * ratio))
    x_old = np.linspace(0, 1, f32.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0, 1, new_len, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, f32).astype(np.float32)


def _play_frames(resp_iter, device: int | None, frame_parser, initial_sr: int | None = None):
    byte_buffer = bytearray()
    sr = initial_sr
    stream = None
    out_sr = None
    buffered_seconds = 0.0

    try:
        for chunk in resp_iter:
            if not chunk:
                continue
            byte_buffer.extend(chunk)
            frames = frame_parser(byte_buffer)
            for frame_sr, pcm_le in frames:
                if frame_sr <= 0:
                    continue
                if sr is None:
                    sr = frame_sr
                    try:
                        out_sr = sr
                        stream = sd.OutputStream(channels=1, samplerate=out_sr, dtype="float32", device=device)
                        stream.start()
                    except Exception:
                        out_sr = 48000
                        stream = sd.OutputStream(channels=1, samplerate=out_sr, dtype="float32", device=device)
                        stream.start()
                f32 = np.frombuffer(pcm_le, dtype=np.float32)
                if f32.size == 0:
                    continue
                if out_sr and out_sr != sr:
                    f32 = _resample_linear(f32, sr, out_sr)
                audio = f32.reshape(-1, 1)
                buffered_seconds += audio.shape[0] / float(sr)
                if buffered_seconds >= 0.2 and stream:
                    stream.write(audio)
                    buffered_seconds = 0.0
    finally:
        try:
            if stream:
                time.sleep(0.05)
                stream.stop()
                stream.close()
        except Exception:
            pass


def stream_play_legacy(host: str, text: str, voice: str, language: str, speed: float,
                       chunking: bool = True, chunk_chars: int = 140, chunk_overlap_ms: int = 60,
                       chunk_workers: int = 2, chunk_prefetch: int = 2,
                       device: int | None = None):
    url = host.rstrip("/") + "/synthesize-stream-legacy"
    payload = {
        "text": text,
        "voice": voice,
        "language": language,
        "speed": float(speed),
        "chunking": bool(chunking),
        "stream_chunk_chars": int(chunk_chars),
        "chunk_overlap_ms": int(chunk_overlap_ms),
        "chunk_workers": int(chunk_workers),
        "chunk_prefetch": int(chunk_prefetch),
    }
    resp = requests.post(url, json=payload, stream=True)
    resp.raise_for_status()
    return _play_frames(resp.iter_content(chunk_size=16384), device, parse_frames_legacy)


def _read_until_double_newline(resp) -> tuple[int | None, dict | None, bytes]:
    """Read bytes until '\n\n' encountered; return (sample_rate, metadata, remainder)."""
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=1024):
        if not chunk:
            continue
        buf.extend(chunk)
        idx = buf.find(b"\n\n")
        if idx != -1:
            meta_bytes = bytes(buf[:idx])
            remainder = bytes(buf[idx+2:])
            try:
                metadata = json.loads(meta_bytes.decode(errors="ignore"))
                sr = int(metadata.get("sample_rate", 24000))
            except Exception:
                metadata = None
                sr = None
            return sr, metadata, remainder
    return None, None, bytes(buf)


def stream_play_session(host: str, text: str, voice: str, language: str, speed: float,
                        chunk_chars: int = 120, target_latency_ms: int = 150,
                        device: int | None = None):
    start_url = host.rstrip("/") + "/stream/start"
    start_payload = {
        "text": text,
        "voice": voice,
        "language": language,
        "speed": float(speed),
        "stream_chunk_chars": int(chunk_chars),
        "target_latency_ms": int(target_latency_ms),
    }
    start_resp = requests.post(start_url, json=start_payload)
    start_resp.raise_for_status()
    start_data = start_resp.json()
    session_id = start_data.get("session_id")
    if not session_id:
        raise RuntimeError("No session_id returned from server")
    audio_url = host.rstrip("/") + f"/stream/{session_id}/audio"

    resp = requests.get(audio_url, stream=True)
    resp.raise_for_status()

    # Consume initial metadata line (JSON) ending with \n\n, then pass remainder + subsequent chunks to parser/player
    sr, metadata, remainder = _read_until_double_newline(resp)
    iter_gen = (chunk for chunk in ([remainder] if remainder else []) )
    def combined_iter():
        if remainder:
            yield remainder
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                yield chunk

    return _play_frames(combined_iter(), device, parse_frames_session, initial_sr=sr)


def main():
    ap = argparse.ArgumentParser(description="Kokoro TTS streaming client")
    ap.add_argument("text", help="Text or SSML to synthesize")
    ap.add_argument("--host", default="http://localhost:5000", help="Server host URL")
    ap.add_argument("--voice", default="af_nicole", help="Voice name")
    ap.add_argument("--language", default="en-us", help="Language code")
    ap.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier")
    ap.add_argument("--device", type=int, default=None, help="sounddevice output device index")
    ap.add_argument("--protocol", choices=["session", "legacy"], default="session", help="Streaming protocol")
    # Legacy tuning flags retained for compatibility
    ap.add_argument("--chunking", action="store_true", help="Enable server chunking (legacy)")
    ap.add_argument("--chunk-chars", type=int, default=140, help="Approx chars per stream chunk (legacy)")
    ap.add_argument("--chunk-overlap-ms", type=int, default=60, help="Server-side overlap crossfade ms (legacy)")
    ap.add_argument("--chunk-workers", type=int, default=2, help="Server chunk synthesis workers (legacy)")
    ap.add_argument("--chunk-prefetch", type=int, default=2, help="Server chunk prefetch count (legacy)")
    # Session tuning flags
    ap.add_argument("--target-latency-ms", type=int, default=150, help="Target latency for session protocol")
    args = ap.parse_args()

    try:
        if args.protocol == "legacy":
            stream_play_legacy(
                host=args.host,
                text=args.text,
                voice=args.voice,
                language=args.language,
                speed=args.speed,
                chunking=args.chunking,
                chunk_chars=args.chunk_chars,
                chunk_overlap_ms=args.chunk_overlap_ms,
                chunk_workers=args.chunk_workers,
                chunk_prefetch=args.chunk_prefetch,
                device=args.device,
            )
        else:
            stream_play_session(
                host=args.host,
                text=args.text,
                voice=args.voice,
                language=args.language,
                speed=args.speed,
                chunk_chars=120,
                target_latency_ms=args.target_latency_ms,
                device=args.device,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
