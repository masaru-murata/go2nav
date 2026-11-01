"""Front camera: frame cache, recording (ffmpeg H.264, 1-minute segments) and image utilities (design 12)."""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..common import mono, now, JsonlWriter

log = logging.getLogger("go2nav.camera")


def resize_long_edge(jpeg: bytes, long_edge: int, quality: int = 85) -> bytes:
    import cv2
    arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jpeg
    h, w = arr.shape[:2]
    s = long_edge / float(max(h, w))
    if s < 1.0:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf) if ok else jpeg


def center_crop(jpeg: bytes, frac_w: float = 0.5, frac_h: float = 0.6, quality: int = 92) -> bytes:
    """High-resolution crop of the image centre (for reading signage)."""
    import cv2
    arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return jpeg
    h, w = arr.shape[:2]
    cw, ch = int(w * frac_w), int(h * frac_h)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    ok, buf = cv2.imencode(".jpg", arr[y0:y0 + ch, x0:x0 + cw], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bytes(buf) if ok else jpeg


class CameraService:
    """Polls the adapter at `record_fps`, keeps the latest frame, and streams frames to ffmpeg while recording."""

    def __init__(self, adapter, cfg: dict):
        self.adapter = adapter
        self.cfg = cfg
        self.fps = int(cfg.get("record_fps", 10))
        self.segment_s = int(cfg.get("segment_s", 60))
        self._latest: Optional[tuple[bytes, float, float]] = None   # (jpeg, t_mono, t_wall)
        self._lock = threading.Lock()
        self._run = False
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._index: Optional[JsonlWriter] = None
        self._frame_no = 0
        self._rec_dir: Optional[Path] = None
        self._rec_lock = threading.Lock()
        self.failures = 0
        self.frames_ok = 0

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def stop(self) -> None:
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.stop_recording()

    def _loop(self) -> None:
        period = 1.0 / max(1, self.fps)
        while self._run:
            t0 = mono()
            jpeg = None
            try:
                jpeg = self.adapter.get_image()
            except Exception as e:  # pragma: no cover
                log.debug("get_image error: %s", e)
            if jpeg:
                with self._lock:
                    self._latest = (jpeg, mono(), now())
                self.frames_ok += 1
                self._record_frame(jpeg)
            else:
                self.failures += 1
            dt = mono() - t0
            if dt < period:
                time.sleep(period - dt)

    # -- frames -----------------------------------------------------------------------
    def latest(self) -> Optional[tuple[bytes, float, float]]:
        with self._lock:
            return self._latest

    def age_ms(self) -> Optional[int]:
        l = self.latest()
        return None if l is None else int((mono() - l[1]) * 1000)

    def fresh_frame(self, max_age_ms: int = 300) -> Optional[bytes]:
        l = self.latest()
        if l is not None and (mono() - l[1]) * 1000 <= max_age_ms:
            return l[0]
        try:
            jpeg = self.adapter.get_image()
        except Exception:  # pragma: no cover
            jpeg = None
        if jpeg:
            with self._lock:
                self._latest = (jpeg, mono(), now())
            self._record_frame(jpeg)
            return jpeg
        return l[0] if l is not None else None

    # -- recording --------------------------------------------------------------------
    def start_recording(self, cam_dir: Path) -> bool:
        cam_dir = Path(cam_dir)
        cam_dir.mkdir(parents=True, exist_ok=True)
        with self._rec_lock:
            if self._proc is not None:
                self.stop_recording()
            ffmpeg = self.cfg.get("ffmpeg", "ffmpeg")
            codec = self.cfg.get("video_codec", "libx264")
            crf = str(self.cfg.get("crf", 26))
            seg = str(self.segment_s)
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "image2pipe", "-framerate", str(self.fps), "-c:v", "mjpeg", "-i", "-",
                   "-c:v", codec, "-preset", "veryfast", "-crf", crf, "-pix_fmt", "yuv420p",
                   "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
                   "-f", "segment", "-segment_time", seg, "-reset_timestamps", "1",
                   str(cam_dir / "cam_%06d.mp4")]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                log.error("ffmpeg not found (%s); recording disabled", ffmpeg)
                self._proc = None
                return False
            self._index = JsonlWriter(cam_dir / "cam_index.jsonl")
            self._index.write({"meta": True, "fps": self.fps, "segment_s": self.segment_s, "t_wall_start": now(), "t_mono_start": mono()})
            self._frame_no = 0
            self._rec_dir = cam_dir
            log.info("recording to %s", cam_dir)
            return True

    def stop_recording(self) -> None:
        with self._rec_lock:
            p, self._proc = self._proc, None
            idx, self._index = self._index, None
            self._rec_dir = None
        if p is not None:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        if idx is not None:
            idx.close()

    def recording(self) -> Optional[str]:
        return None if self._rec_dir is None else str(self._rec_dir)

    def _record_frame(self, jpeg: bytes) -> None:
        with self._rec_lock:
            if self._proc is None:
                return
            try:
                self._proc.stdin.write(jpeg)
                self._index.write({"frame": self._frame_no, "t_wall": now(), "t_mono": mono()})
                self._frame_no += 1
            except (BrokenPipeError, ValueError, OSError) as e:
                log.error("recording pipe broken: %s", e)
                self._proc = None

    def video_time(self, t_wall: float) -> Optional[dict]:
        """Map a wall time to (segment, seconds) using the frame index of the current recording."""
        return None
