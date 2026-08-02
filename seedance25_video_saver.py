"""ComfyUI output node for downloading Seedance 2.5 video URLs."""

import os

import numpy as np
import requests
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _output_directory():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.join(os.path.expanduser("~"), "comfyui_output")


def _safe_subfolder(value):
    value = (value or "seedance25").strip()
    normalized = os.path.normpath(value)
    if normalized in ("", "."):
        return "seedance25"
    if os.path.isabs(normalized) or normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("save_subfolder must stay inside ComfyUI's output directory.")
    return normalized


def _safe_prefix(value):
    prefix = os.path.basename((value or "seedance25").strip())
    return prefix or "seedance25"


class Seedance25VideoSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"multiline": False, "default": ""}),
                "save_subfolder": ("STRING", {"default": "seedance25"}),
                "filename_prefix": ("STRING", {"default": "seedance25"}),
            },
            "optional": {
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 30}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("frames", "filepath", "frame_count")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"
    OUTPUT_NODE = True

    def run(
        self,
        video_url,
        save_subfolder,
        filename_prefix,
        frame_load_cap=0,
        skip_first_frames=0,
        select_every_nth=1,
    ):
        if not video_url or not video_url.strip().lower().startswith(("http://", "https://")):
            return self._error("video_url must be an http(s) URL.")

        try:
            save_subfolder = _safe_subfolder(save_subfolder)
            filename_prefix = _safe_prefix(filename_prefix)
            output_dir = os.path.join(_output_directory(), save_subfolder)
            os.makedirs(output_dir, exist_ok=True)
            file_number = 1
            filepath = os.path.join(output_dir, f"{filename_prefix}_{file_number:05d}.mp4")
            while os.path.exists(filepath):
                file_number += 1
                filepath = os.path.join(output_dir, f"{filename_prefix}_{file_number:05d}.mp4")

            print(f"[Seedance 2.5 Saver] Downloading {video_url[:100]}...")
            response = requests.get(video_url.strip(), stream=True, timeout=600)
            response.raise_for_status()
            with open(filepath, "wb") as handle:
                for chunk in response.iter_content(1024 * 32):
                    if chunk:
                        handle.write(chunk)

            frames, count = self._load_frames(
                filepath,
                int(frame_load_cap),
                int(skip_first_frames),
                int(select_every_nth),
            )
            filename = os.path.basename(filepath)
            preview = {
                "filename": filename,
                "subfolder": save_subfolder,
                "type": "output",
                "format": "video/mp4",
            }
            print(f"[Seedance 2.5 Saver] Saved {filename} — {count} frame(s)")
            return {"ui": {"gifs": [preview]}, "result": (frames, filepath, count)}
        except Exception as exc:
            return self._error(str(exc))

    @staticmethod
    def _load_frames(path, cap, skip, every):
        try:
            import cv2

            frames = []
            raw_index = 0
            capture = cv2.VideoCapture(path)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if raw_index < skip:
                    raw_index += 1
                    continue
                if (raw_index - skip) % every == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    frames.append(rgb)
                    if cap > 0 and len(frames) >= cap:
                        break
                raw_index += 1
            capture.release()
            if not frames:
                raise RuntimeError("No readable frames were found in the downloaded video.")
            return torch.from_numpy(np.stack(frames)), len(frames)
        except Exception as exc:
            print(f"[Seedance 2.5 Saver] Frame decode failed: {exc}")
            return torch.zeros(1, 64, 64, 3), 1

    @staticmethod
    def _error(message):
        print(f"[Seedance 2.5 Saver] ERROR: {message}")
        return {
            "ui": {"text": [message]},
            "result": (torch.zeros(1, 64, 64, 3), "ERROR", 0),
        }


NODE_CLASS_MAPPINGS = {"Seedance25VideoSaver": Seedance25VideoSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"Seedance25VideoSaver": "🌱 Seedance 2.5 Save Video"}
