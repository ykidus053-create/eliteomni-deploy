"""
EliteOmni Video Editor
Capabilities:
- Transcribe video/audio via Whisper
- Remove silences, filler words, bad takes
- Generate SRT subtitles
- Trim/cut/merge clips via FFmpeg
- Extract audio
"""
import os, subprocess, tempfile, json, re
from pathlib import Path

UPLOAD_DIR = os.path.expanduser("~/eliteomni_uploads")
OUTPUT_DIR = os.path.expanduser("~/eliteomni_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ffmpeg(args: list, timeout=120) -> tuple[int, str, str]:
    result = subprocess.run(["ffmpeg", "-y"] + args,
        capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr

def transcribe(video_path: str, model_size="base") -> dict:
    """Transcribe video/audio using Whisper. Returns segments with timestamps."""
    try:
        import whisper
        model = whisper.load_model(model_size)
        result = model.transcribe(video_path, word_timestamps=True)
        return result
    except Exception as e:
        return {"error": str(e), "text": "", "segments": []}

def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio track from video."""
    if not output_path:
        output_path = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    code, _, err = ffmpeg(["-i", video_path, "-vn", "-acodec", "mp3", output_path])
    if code != 0:
        raise RuntimeError(f"FFmpeg error: {err}")
    return output_path

def trim_clip(video_path: str, start: float, end: float, output_path: str = None) -> str:
    """Trim video to [start, end] seconds."""
    if not output_path:
        output_path = video_path.rsplit(".", 1)[0] + f"_trim_{start:.1f}_{end:.1f}.mp4"
    code, _, err = ffmpeg([
        "-i", video_path,
        "-ss", str(start), "-to", str(end),
        "-c", "copy", output_path
    ])
    if code != 0:
        raise RuntimeError(f"FFmpeg error: {err}")
    return output_path

def remove_silences(video_path: str, silence_threshold=-35, min_silence=0.5) -> str:
    """Remove silent sections from video using FFmpeg silencedetect."""
    output_path = video_path.rsplit(".", 1)[0] + "_nosilence.mp4"
    # Detect silences
    code, _, err = ffmpeg([
        "-i", video_path,
        "-af", f"silencedetect=noise={silence_threshold}dB:d={min_silence}",
        "-f", "null", "-"
    ], timeout=60)
    # Parse silence intervals
    silences = []
    starts = re.findall(r"silence_start: ([\d.]+)", err)
    ends = re.findall(r"silence_end: ([\d.]+)", err)
    for s, e in zip(starts, ends):
        silences.append((float(s), float(e)))
    if not silences:
        return video_path
    # Build keep segments
    duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    if duration_match:
        h, m, s = duration_match.groups()
        total = int(h)*3600 + int(m)*60 + float(s)
    else:
        total = 9999
    keeps = []
    prev = 0.0
    for s_start, s_end in silences:
        if s_start > prev + 0.1:
            keeps.append((prev, s_start))
        prev = s_end
    if prev < total - 0.1:
        keeps.append((prev, total))
    if not keeps:
        return video_path
    # Cut and concat
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (start, end) in enumerate(keeps):
            part = os.path.join(tmp, f"part_{i}.mp4")
            ffmpeg(["-i", video_path, "-ss", str(start), "-to", str(end),
                    "-c", "copy", part])
            parts.append(part)
        list_file = os.path.join(tmp, "list.txt")
        with open(list_file, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file,
                "-c", "copy", output_path])
    return output_path

def generate_srt(transcript: dict, output_path: str) -> str:
    """Generate SRT subtitle file from Whisper transcript."""
    def fmt_time(secs):
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    lines = []
    for i, seg in enumerate(transcript.get("segments", []), 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    return output_path

def burn_subtitles(video_path: str, srt_path: str, output_path: str = None) -> str:
    """Burn SRT subtitles into video."""
    if not output_path:
        output_path = video_path.rsplit(".", 1)[0] + "_subtitled.mp4"
    code, _, err = ffmpeg([
        "-i", video_path,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy", output_path
    ])
    if code != 0:
        raise RuntimeError(f"FFmpeg error: {err}")
    return output_path

def merge_clips(clip_paths: list, output_path: str) -> str:
    """Merge multiple video clips into one."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name
    code, _, err = ffmpeg(["-f", "concat", "-safe", "0",
                            "-i", list_file, "-c", "copy", output_path])
    os.unlink(list_file)
    if code != 0:
        raise RuntimeError(f"FFmpeg error: {err}")
    return output_path

def video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path
    ], capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except Exception:
        return {}
