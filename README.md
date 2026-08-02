# Seedance 2.5 ComfyUI

ComfyUI custom nodes for the [Seedance 2.5 API](https://github.com/SamurAIGPT/Seedance-2.5-API), delivered through MuAPI.

This pack follows the same workflow pattern as `seedance2-comfyui`: add one API Key node, connect it to a generation node, then connect `video_url` to Save Video and `first_frame` to Preview Image.

## Related Projects

- [MuAPI](https://muapi.ai) — Unified API for running Seedance and other image, video, and audio models. Explore the [Seedance 2.5 landing page](https://muapi.ai/seedance-2.5) or try [Seedance 2.5 text-to-video](https://muapi.ai/playground/seedance-2.5-text-to-video).
- [Seedance-2.5-API](https://github.com/SamurAIGPT/Seedance-2.5-API) — Python wrapper for the same Seedance 2.5 API.
- [seedance2-comfyui](https://github.com/Anil-matcha/seedance2-comfyui) — Related Seedance ComfyUI nodes and workflows.

## Included nodes

| Node | What it does |
| --- | --- |
| Seedance 2.5 API Key | Stores one MuAPI key for the workflow. |
| Text-to-Video | Generates a 720p or 480p clip from a prompt. |
| Image-to-Video | Animates one input image using the 2.5 `image_url` contract. |
| First/Last Frame | Interpolates between exactly two input images. |
| Omni Reference | Combines up to 20 images, 6 videos, and 6 audio references. |
| Spicy Text/Image-to-Video | 720p relaxed-moderation siblings exposed by the API. |
| Consistent Character | Creates a reusable character sheet from 1–3 reference images. |
| Consistent Video | Anchors Omni Reference on a character sheet. |
| Extend | Uses MuAPI’s Seedance-family extension endpoint. |
| Save Video | Downloads a URL to ComfyUI/output and returns decoded frames. |

## Installation

Copy this directory into `ComfyUI/custom_nodes/seedance2.5-comfyui`, then install the Python dependencies in the same environment ComfyUI uses:

```bash
pip install -r ComfyUI/custom_nodes/seedance2.5-comfyui/requirements.txt
```

Restart ComfyUI. The example workflows can be loaded with **File → Load**.

Included examples are `Seedance25_T2V_Example.json`, `Seedance25_FirstLastFrame_Example.json`, and `Seedance25_ConsistentCharacter_Example.json`.

## API key

Use the **🔑 Seedance 2.5 API Key** node and connect its output to the generation nodes. Alternatively, leave the node field empty and configure one of these:

```bash
export MUAPI_API_KEY="your_muapi_key"
```

or configure the MuAPI CLI:

```bash
muapi auth configure --api-key YOUR_KEY
```

The node also reads `~/.muapi/config.json` when it contains an `api_key` field.

## Quick workflow

1. Add **🔑 Seedance 2.5 API Key**.
2. Add **🌱 Seedance 2.5 Text-to-Video**.
3. Choose `720p` or `480p`, aspect ratio, duration, and output format.
4. Connect `video_url` to **🌱 Seedance 2.5 Save Video**.
5. Connect `first_frame` to ComfyUI’s **Preview Image** node.
6. Queue the prompt.

`duration` accepts 4–30 seconds. `seed = -1` uses the provider’s default random behavior. `generate_audio`, `camera_fixed`, `output_format`, and the optional character ID are passed through to the API.

## Reference inputs

### Image-to-Video

Connect one `IMAGE`. The node uploads it and sends it as `image_url` to the 2.5 endpoint.

### First/Last Frame

Connect `first_image` and `last_image`. The node uploads both in order and sends:

```json
{
  "images_list": ["first-frame-url", "last-frame-url"]
}
```

### Omni Reference

Use `@image1`, `@video1`, and `@audio1` in the prompt. Connect media slots from 1 upward so the prompt numbering matches the order of the uploaded references. Video and audio fields accept a URL, an absolute local path, or a filename from `ComfyUI/input`.

The 2.5 limits documented by the API are 20 images, 6 videos, and 6 audio clips. If a local video/audio file is selected, it is uploaded through `/upload_file` before generation.

### Consistent character

```text
LoadImage → Seedance 2.5 Consistent Character → Consistent Video → Save Video
```

The character node returns `sheet_image`, `sheet_url`, and `character_id`. Connect `sheet_image` to `Consistent Video` for an image anchor, or connect `character_id` to a text/image/omni node and reference it inline as `@character:<id>`.

## API endpoints

The generation nodes use the documented 2.5 endpoints:

| Capability | 720p | 480p |
| --- | --- | --- |
| Text-to-Video | `seedance-2.5-text-to-video` | `seedance-2.5-text-to-video-480p` |
| Image-to-Video | `seedance-2.5-image-to-video` | `seedance-2.5-image-to-video-480p` |
| First/Last Frame | `seedance-2.5-first-last-frame` | `seedance-2.5-first-last-frame-480p` |
| Omni Reference | `seedance-2.5-omni-reference` | `seedance-2.5-omni-reference-480p` |

All requests use `x-api-key`, poll `predictions/{request_id}/result`, and wait for completion before returning the URL and first frame. Seedance 2.5 currently exposes 480p and 720p tiers; this pack does not advertise 1080p or 4K controls.

The upstream API repository documents video extension and editing through Seedance 2.0-compatible endpoints. The included Extend node therefore targets `seedance-v2.0-extend` intentionally.

## Requirements

Python 3.8+, `requests`, `Pillow`, `numpy`, and `opencv-python`. ComfyUI supplies the installed PyTorch runtime used for IMAGE tensors.
