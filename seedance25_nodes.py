"""ComfyUI nodes for the Seedance 2.5 API.

The nodes call MuAPI directly, which keeps this custom node pack independent
from the Python SDK while following the endpoint and payload contract from
the upstream Seedance 2.5 API project.

Seedance 2.5 endpoints currently exposed here:

* Text-to-video (720p and 480p)
* Image-to-video (720p and 480p, one start image)
* First/last-frame interpolation (720p and 480p)
* Omni-reference video (720p and 480p)
* Spicy text-to-video and image-to-video (720p)
* Character sheets and sheet-anchored consistent video

Seedance 2.5 does not currently have dedicated extend/edit endpoints. The
extend node therefore uses the Seedance 2.0-compatible endpoint documented by
the upstream API wrapper.
"""

import io
import json
import mimetypes
import os
import tempfile
import time

import numpy as np
import requests
import torch
from PIL import Image


BASE_URL = os.getenv("SEEDANCE25_API_BASE_URL", "https://api.muapi.ai/api/v1")
POLL_INTERVAL = 5
MAX_WAIT = 1800

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus")

_NONE_CHOICE = "(none)"
_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21"]
_RESOLUTIONS = ["720p", "480p"]
_QUALITIES = ["basic", "high"]
_OUTPUT_FORMATS = ["mp4", "mov"]


def _input_directory():
    try:
        import folder_paths

        return folder_paths.get_input_directory()
    except Exception:
        return None


def _list_input_files(extensions):
    """Return ComfyUI input files suitable for a dropdown widget."""
    input_dir = _input_directory()
    if not input_dir or not os.path.isdir(input_dir):
        return [_NONE_CHOICE]
    try:
        names = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
            and name.lower().endswith(extensions)
        ]
    except OSError:
        return [_NONE_CHOICE]
    return [_NONE_CHOICE] + sorted(names)


def _load_api_key(api_key_input=""):
    """Resolve a key from the node, environment, or MuAPI CLI config."""
    if api_key_input and str(api_key_input).strip():
        return str(api_key_input).strip()

    for env_name in ("MUAPI_API_KEY", "SEEDANCE25_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    config_path = os.path.expanduser("~/.muapi/config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            value = config.get("api_key") or config.get("MUAPI_API_KEY") or ""
            if str(value).strip():
                return str(value).strip()
        except (OSError, ValueError, TypeError):
            pass

    raise RuntimeError(
        "No MuAPI key found. Paste one into the API Key node, set MUAPI_API_KEY, "
        "or run `muapi auth configure --api-key YOUR_KEY`."
    )


def _api_url(endpoint):
    return f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"


def _check_response(response):
    if response.status_code == 401:
        raise RuntimeError("MuAPI authentication failed. Check the API key.")
    if response.status_code == 402:
        raise RuntimeError("MuAPI rejected the request because the account lacks credits.")
    if response.status_code == 429:
        raise RuntimeError("MuAPI rate limit reached. Retry the workflow later.")
    if response.ok:
        return

    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:500]
    raise RuntimeError(f"MuAPI HTTP {response.status_code}: {detail}")


def _submit(api_key, endpoint, payload):
    try:
        response = requests.post(
            _api_url(endpoint),
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not submit the MuAPI request: {exc}") from exc

    _check_response(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("MuAPI returned a non-JSON submission response.") from exc

    request_id = data.get("request_id") or data.get("id")
    if not request_id:
        raise RuntimeError(f"MuAPI submission did not include request_id: {data}")
    return str(request_id)


def _poll(api_key, request_id):
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        try:
            response = requests.get(
                _api_url(f"predictions/{request_id}/result"),
                headers={"x-api-key": api_key},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not poll MuAPI request {request_id}: {exc}") from exc

        _check_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("MuAPI returned a non-JSON polling response.") from exc

        status = str(data.get("status", "")).strip().lower()
        print(f"[Seedance 2.5] {status or 'unknown'} — {request_id}")
        if status in {"completed", "complete", "succeeded", "success", "done"}:
            return data
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            error = data.get("error") or data.get("message") or data
            raise RuntimeError(f"Seedance 2.5 generation failed: {error}")

        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"Timed out waiting for Seedance 2.5 request {request_id}.")


def _output_value(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("url", "video_url", "image_url", "file_url", "output"):
            found = _output_value(value.get(key))
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _output_value(item)
            if found:
                return found
    return None


def _output_url(result):
    for key in ("outputs", "output", "video_url", "url", "file_url"):
        found = _output_value(result.get(key))
        if found:
            return found
    raise RuntimeError(f"MuAPI result did not contain an output URL: {result}")


def _image_url(result):
    for key in ("outputs", "output", "image_url", "sheet_url", "url", "file_url"):
        found = _output_value(result.get(key))
        if found:
            return found
    raise RuntimeError(f"MuAPI result did not contain an image URL: {result}")


def _download_image(url):
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    except (OSError, ValueError, requests.RequestException) as exc:
        raise RuntimeError(f"Could not download image output: {exc}") from exc


def _first_frame(video_url):
    """Download the generated video and return its first frame as IMAGE."""
    temp_path = None
    try:
        import cv2

        response = requests.get(video_url, timeout=180, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            temp_path = handle.name
            for chunk in response.iter_content(1024 * 32):
                if chunk:
                    handle.write(chunk)

        capture = cv2.VideoCapture(temp_path)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError("the downloaded video had no readable first frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).unsqueeze(0)
    except Exception as exc:
        print(f"[Seedance 2.5] Could not decode first frame: {exc}")
        return torch.zeros(1, 64, 64, 3)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _image_bytes(image_tensor):
    if image_tensor is None:
        raise ValueError("An image input is required.")
    tensor = image_tensor.detach() if hasattr(image_tensor, "detach") else image_tensor
    if getattr(tensor, "dim", lambda: 0)() == 4:
        tensor = tensor[0]
    array = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError("Expected a ComfyUI IMAGE tensor with shape [H, W, C].")
    array = array.astype(np.float32)
    if array.size and float(np.max(array)) > 1.0:
        array /= 255.0
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] < 3:
        raise ValueError("The image input must have at least one color channel.")
    array = np.clip(array[..., :3], 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((array * 255.0).round().astype(np.uint8), "RGB").save(
        buffer, format="JPEG", quality=95
    )
    buffer.seek(0)
    return buffer


def _upload_image(api_key, image_tensor):
    buffer = _image_bytes(image_tensor)
    try:
        response = requests.post(
            _api_url("upload_file"),
            headers={"x-api-key": api_key},
            files={"file": ("image.jpg", buffer, "image/jpeg")},
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not upload image to MuAPI: {exc}") from exc
    _check_response(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("MuAPI returned a non-JSON image upload response.") from exc
    url = _output_value(data)
    if not url:
        raise RuntimeError(f"MuAPI image upload did not return a URL: {data}")
    return url


def _resolve_media_ref(api_key, reference, kind):
    """Resolve a URL or local ComfyUI/input path to a remote URL."""
    import pathlib

    if not reference or not str(reference).strip() or reference == _NONE_CHOICE:
        return None
    reference = str(reference).strip().strip('"').strip("'")
    if reference.lower().startswith(("http://", "https://")):
        return reference

    path = pathlib.Path(os.path.expanduser(reference))
    if not path.is_file():
        input_dir = _input_directory()
        if input_dir:
            candidate = pathlib.Path(input_dir) / reference
            if candidate.is_file():
                path = candidate
    if not path.is_file():
        raise RuntimeError(
            f"{kind.title()} reference not found: {reference!r}. Use an http(s) URL, "
            "an absolute path, or a file in ComfyUI/input."
        )

    mime = mimetypes.guess_type(path.name)[0]
    if not mime:
        mime = "video/mp4" if kind == "video" else "audio/mpeg"
    try:
        with path.open("rb") as handle:
            response = requests.post(
                _api_url("upload_file"),
                headers={"x-api-key": api_key},
                files={"file": (path.name, handle, mime)},
                timeout=600,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not upload {kind} reference: {exc}") from exc
    _check_response(response)
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"MuAPI returned a non-JSON {kind} upload response.") from exc
    url = _output_value(data)
    if not url:
        raise RuntimeError(f"MuAPI {kind} upload did not return a URL: {data}")
    return url


def _endpoint(base, resolution):
    return f"{base}-480p" if resolution == "480p" else base


def _generation_options(
    aspect_ratio,
    quality,
    duration,
    generate_audio,
    camera_fixed,
    seed,
    output_format,
):
    """Build shared 2.5 generation options without adding the UI resolution."""
    payload = {
        "aspect_ratio": aspect_ratio,
        "duration": int(duration),
        "quality": quality,
        "generate_audio": bool(generate_audio),
        "camera_fixed": bool(camera_fixed),
        "output_format": output_format,
    }
    if seed is not None and int(seed) >= 0:
        payload["seed"] = int(seed)
    return payload


def _fixed_resolution_options():
    return {
        "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
        "quality": (_QUALITIES, {"default": "basic"}),
        "duration": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
        "generate_audio": ("BOOLEAN", {"default": True}),
        "camera_fixed": ("BOOLEAN", {"default": False}),
        "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
        "output_format": (_OUTPUT_FORMATS, {"default": "mp4"}),
    }


def _generation_inputs():
    inputs = {"resolution": (_RESOLUTIONS, {"default": "720p"})}
    inputs.update(_fixed_resolution_options())
    return inputs


def _with_character_prompt(prompt, character_id):
    prompt = (prompt or "").strip()
    character_id = (character_id or "").strip()
    if character_id and "@character:" not in prompt:
        prompt = f"@character:{character_id} {prompt}".strip()
    return prompt


class _Seedance25GenerationNode:
    def _complete(self, api_key, endpoint, payload, label):
        print(f"[{label}] Submitting to {endpoint}...")
        request_id = _submit(api_key, endpoint, payload)
        result = _poll(api_key, request_id)
        video_url = _output_url(result)
        print(f"[{label}] Done — request_id={request_id}")
        return video_url, _first_frame(video_url), request_id


class Seedance25ApiKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "MuAPI key. Get one at muapi.ai → Dashboard → API Keys.",
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(self, api_key):
        return (_load_api_key(api_key),)


class Seedance25TextToVideo(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "A cinematic aerial shot of a futuristic city at dusk, volumetric lighting",
                },
            )
        }
        required.update(_generation_inputs())
        return {
            "required": required,
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "character_id": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Optional character_id from Seedance 2.5 Character.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        resolution,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        character_id="",
    ):
        payload = {
            "prompt": _with_character_prompt(prompt, character_id),
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            _load_api_key(api_key),
            _endpoint("seedance-2.5-text-to-video", resolution),
            payload,
            "Seedance 2.5 T2V",
        )


class Seedance25ImageToVideo(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "Animate the scene with gentle cinematic camera movement",
                },
            )
        }
        required.update(_generation_inputs())
        return {
            "required": required,
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "image": ("IMAGE",),
                "character_id": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        resolution,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        image=None,
        character_id="",
    ):
        if image is None:
            raise ValueError("Connect one IMAGE to the Seedance 2.5 Image-to-Video node.")
        key = _load_api_key(api_key)
        image_url = _upload_image(key, image)
        payload = {
            "prompt": _with_character_prompt(prompt, character_id),
            "image_url": image_url,
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            key,
            _endpoint("seedance-2.5-image-to-video", resolution),
            payload,
            "Seedance 2.5 I2V",
        )


class Seedance25FirstLastFrame(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "Smooth cinematic transition from the first frame to the last frame",
                },
            )
        }
        required.update(_generation_inputs())
        return {
            "required": required,
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "first_image": ("IMAGE",),
                "last_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        resolution,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        first_image=None,
        last_image=None,
    ):
        if first_image is None or last_image is None:
            raise ValueError(
                "Connect both first_image and last_image to the Seedance 2.5 First/Last Frame node."
            )
        key = _load_api_key(api_key)
        first_url = _upload_image(key, first_image)
        last_url = _upload_image(key, last_image)
        payload = {
            "prompt": prompt.strip(),
            "images_list": [first_url, last_url],
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            key,
            _endpoint("seedance-2.5-first-last-frame", resolution),
            payload,
            "Seedance 2.5 First/Last Frame",
        )


class Seedance25SpicyTextToVideo(_Seedance25GenerationNode):
    """720p relaxed-moderation sibling documented by the 2.5 API."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"multiline": True, "default": "A cinematic scene"})
        }
        required.update(_fixed_resolution_options())
        return {
            "required": required,
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "character_id": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        character_id="",
    ):
        payload = {
            "prompt": _with_character_prompt(prompt, character_id),
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            _load_api_key(api_key),
            "seedance-2.5-spicy-text-to-video",
            payload,
            "Seedance 2.5 Spicy T2V",
        )


class Seedance25SpicyImageToVideo(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"multiline": True, "default": "Animate the image cinematically"})
        }
        required.update(_fixed_resolution_options())
        return {
            "required": required,
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "image": ("IMAGE",),
                "character_id": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        image=None,
        character_id="",
    ):
        if image is None:
            raise ValueError("Connect one IMAGE to the Seedance 2.5 Spicy Image-to-Video node.")
        key = _load_api_key(api_key)
        payload = {
            "prompt": _with_character_prompt(prompt, character_id),
            "image_url": _upload_image(key, image),
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            key,
            "seedance-2.5-spicy-image-to-video",
            payload,
            "Seedance 2.5 Spicy I2V",
        )


class Seedance25OmniReference(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        video_choices = _list_input_files(VIDEO_EXTS)
        audio_choices = _list_input_files(AUDIO_EXTS)
        required = {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "A dramatic cinematic scene matching the reference media",
                },
            )
        }
        required.update(_generation_inputs())
        optional = {
            "api_key": ("STRING", {"multiline": False, "default": ""}),
            "character_id": ("STRING", {"multiline": False, "default": ""}),
        }
        for index in range(1, 21):
            optional[f"image_{index}"] = ("IMAGE",)
        for index in range(1, 7):
            optional[f"video_file_{index}"] = (
                video_choices,
                {
                    "default": _NONE_CHOICE,
                    "tooltip": "Select a file in ComfyUI/input, or use the URL/path field.",
                },
            )
            optional[f"video_url_{index}"] = (
                "STRING",
                {"multiline": False, "default": "", "tooltip": "URL or local path override."},
            )
        for index in range(1, 7):
            optional[f"audio_file_{index}"] = (
                audio_choices,
                {
                    "default": _NONE_CHOICE,
                    "tooltip": "Select a file in ComfyUI/input, or use the URL/path field.",
                },
            )
            optional[f"audio_url_{index}"] = (
                "STRING",
                {"multiline": False, "default": "", "tooltip": "URL or local path override."},
            )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        resolution,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        character_id="",
        **kwargs,
    ):
        key = _load_api_key(api_key)

        images_list = []
        for index in range(1, 21):
            image = kwargs.get(f"image_{index}")
            if image is not None:
                print(f"[Seedance 2.5 Omni] Uploading image {index}...")
                images_list.append(_upload_image(key, image))

        videos_list = []
        for index in range(1, 7):
            selected = kwargs.get(f"video_file_{index}", _NONE_CHOICE)
            override = kwargs.get(f"video_url_{index}", "")
            reference = selected if selected and selected != _NONE_CHOICE else override
            resolved = _resolve_media_ref(key, reference, "video")
            if resolved:
                videos_list.append(resolved)

        audios_list = []
        for index in range(1, 7):
            selected = kwargs.get(f"audio_file_{index}", _NONE_CHOICE)
            override = kwargs.get(f"audio_url_{index}", "")
            reference = selected if selected and selected != _NONE_CHOICE else override
            resolved = _resolve_media_ref(key, reference, "audio")
            if resolved:
                audios_list.append(resolved)

        payload = {
            "prompt": _with_character_prompt(prompt, character_id),
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        if images_list:
            payload["images_list"] = images_list
        if videos_list:
            payload["videos_list"] = videos_list
        if audios_list:
            payload["audios_list"] = audios_list

        print(
            "[Seedance 2.5 Omni] Submitting "
            f"({len(images_list)} image(s), {len(videos_list)} video(s), "
            f"{len(audios_list)} audio(s))..."
        )
        return self._complete(
            key,
            _endpoint("seedance-2.5-omni-reference", resolution),
            payload,
            "Seedance 2.5 Omni",
        )


class Seedance25Character(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outfit_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A distinctive cinematic outfit with clear colors and accessories",
                    },
                )
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "character_name": ("STRING", {"multiline": False, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("sheet_image", "sheet_url", "character_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        outfit_description,
        api_key="",
        character_name="",
        image_1=None,
        image_2=None,
        image_3=None,
    ):
        key = _load_api_key(api_key)
        images_list = []
        for index, image in enumerate((image_1, image_2, image_3), 1):
            if image is not None:
                print(f"[Seedance 2.5 Character] Uploading reference image {index}...")
                images_list.append(_upload_image(key, image))
        if not images_list:
            raise ValueError("Connect at least one reference IMAGE to create a character sheet.")

        payload = {"images_list": images_list, "prompt": outfit_description.strip()}
        if character_name and character_name.strip():
            payload["character_name"] = character_name.strip()
        print(f"[Seedance 2.5 Character] Creating sheet from {len(images_list)} image(s)...")
        request_id = _submit(key, "seedance-2-character", payload)
        result = _poll(key, request_id)

        try:
            sheet_url = _image_url(result)
            sheet_image = _download_image(sheet_url)
        except RuntimeError as exc:
            print(f"[Seedance 2.5 Character] Sheet image unavailable: {exc}")
            sheet_url = ""
            sheet_image = torch.zeros(1, 64, 64, 3)
        print(f"[Seedance 2.5 Character] Done — character_id={request_id}")
        return sheet_image, sheet_url, request_id


class Seedance25ConsistentVideo(_Seedance25GenerationNode):
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": (
                "STRING",
                {
                    "multiline": True,
                    "default": "The character walks through a neon-lit city at night, cinematic motion",
                },
            )
        }
        required.update(_generation_inputs())
        optional = {
            "api_key": ("STRING", {"multiline": False, "default": ""}),
            "sheet_image": ("IMAGE",),
            "sheet_url": ("STRING", {"multiline": False, "default": ""}),
        }
        for index in range(2, 6):
            optional[f"scene_image_{index}"] = ("IMAGE",)
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        prompt,
        resolution,
        aspect_ratio,
        quality,
        duration,
        generate_audio,
        camera_fixed,
        seed,
        output_format,
        api_key="",
        sheet_image=None,
        sheet_url="",
        **kwargs,
    ):
        key = _load_api_key(api_key)
        images_list = []
        if sheet_image is not None:
            images_list.append(_upload_image(key, sheet_image))
        elif sheet_url and sheet_url.strip():
            images_list.append(sheet_url.strip())
        else:
            raise ValueError("Connect sheet_image or provide sheet_url for consistent video.")

        for index in range(2, 6):
            image = kwargs.get(f"scene_image_{index}")
            if image is not None:
                images_list.append(_upload_image(key, image))

        anchored_prompt = prompt.strip()
        if "@image1" not in anchored_prompt:
            anchored_prompt = f"@image1 {anchored_prompt}".strip()
        payload = {
            "prompt": anchored_prompt,
            "images_list": images_list,
            **_generation_options(
                aspect_ratio,
                quality,
                duration,
                generate_audio,
                camera_fixed,
                seed,
                output_format,
            ),
        }
        return self._complete(
            key,
            _endpoint("seedance-2.5-omni-reference", resolution),
            payload,
            "Seedance 2.5 Consistent Video",
        )


class Seedance25Extend(_Seedance25GenerationNode):
    """Use the Seedance-family extension endpoint supported by MuAPI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "request_id": (
                    "STRING",
                    {"multiline": False, "default": "", "tooltip": "Completed generation request_id."},
                ),
                "quality": (_QUALITIES, {"default": "basic"}),
                "duration": ("INT", {"default": 5, "min": 4, "max": 30, "step": 1}),
                "output_format": (_OUTPUT_FORMATS, {"default": "mp4"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("video_url", "first_frame", "new_request_id")
    FUNCTION = "run"
    CATEGORY = "🌱 Seedance 2.5"

    def run(
        self,
        request_id,
        quality,
        duration,
        output_format,
        api_key="",
        prompt="",
    ):
        request_id = request_id.strip()
        if not request_id:
            raise ValueError("A completed request_id is required to extend a video.")
        payload = {
            "request_id": request_id,
            "prompt": (prompt or "").strip(),
            "duration": int(duration),
            "quality": quality,
            "output_format": output_format,
        }
        return self._complete(
            _load_api_key(api_key),
            "seedance-v2.0-extend",
            payload,
            "Seedance 2.5 Extend",
        )


NODE_CLASS_MAPPINGS = {
    "Seedance25ApiKey": Seedance25ApiKey,
    "Seedance25TextToVideo": Seedance25TextToVideo,
    "Seedance25ImageToVideo": Seedance25ImageToVideo,
    "Seedance25FirstLastFrame": Seedance25FirstLastFrame,
    "Seedance25SpicyTextToVideo": Seedance25SpicyTextToVideo,
    "Seedance25SpicyImageToVideo": Seedance25SpicyImageToVideo,
    "Seedance25OmniReference": Seedance25OmniReference,
    "Seedance25Character": Seedance25Character,
    "Seedance25ConsistentVideo": Seedance25ConsistentVideo,
    "Seedance25Extend": Seedance25Extend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance25ApiKey": "🔑 Seedance 2.5 API Key",
    "Seedance25TextToVideo": "🌱 Seedance 2.5 Text-to-Video",
    "Seedance25ImageToVideo": "🌱 Seedance 2.5 Image-to-Video",
    "Seedance25FirstLastFrame": "🌱 Seedance 2.5 First/Last Frame",
    "Seedance25SpicyTextToVideo": "🌶️ Seedance 2.5 Spicy Text-to-Video",
    "Seedance25SpicyImageToVideo": "🌶️ Seedance 2.5 Spicy Image-to-Video",
    "Seedance25OmniReference": "🌱 Seedance 2.5 Omni Reference",
    "Seedance25Character": "🌱 Seedance 2.5 Consistent Character",
    "Seedance25ConsistentVideo": "🌱 Seedance 2.5 Consistent Video",
    "Seedance25Extend": "🌱 Seedance 2.5 Extend",
}
