import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("Contract tests require the PyTorch runtime bundled with ComfyUI.")

from seedance25_nodes import (
    Seedance25FirstLastFrame,
    Seedance25ImageToVideo,
    Seedance25OmniReference,
    Seedance25TextToVideo,
    _endpoint,
)


class Seedance25NodeContractTests(unittest.TestCase):
    def test_resolution_selects_documented_endpoint(self):
        self.assertEqual(
            _endpoint("seedance-2.5-text-to-video", "720p"),
            "seedance-2.5-text-to-video",
        )
        self.assertEqual(
            _endpoint("seedance-2.5-text-to-video", "480p"),
            "seedance-2.5-text-to-video-480p",
        )

    @patch("seedance25_nodes._first_frame", return_value=torch.zeros(1, 2, 2, 3))
    @patch("seedance25_nodes._poll", return_value={"outputs": ["https://video.test/t2v.mp4"]})
    @patch("seedance25_nodes._submit", return_value="t2v-request")
    def test_t2v_uses_25_endpoint_and_prompt_payload(self, submit, _poll, _frame):
        node = Seedance25TextToVideo()
        result = node.run(
            "A red kite over the sea",
            "480p",
            "16:9",
            "basic",
            5,
            True,
            False,
            -1,
            "mp4",
            api_key="test-key",
        )
        self.assertEqual(result[0], "https://video.test/t2v.mp4")
        submit.assert_called_once()
        key, endpoint, payload = submit.call_args.args
        self.assertEqual(key, "test-key")
        self.assertEqual(endpoint, "seedance-2.5-text-to-video-480p")
        self.assertEqual(payload["prompt"], "A red kite over the sea")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(payload["duration"], 5)

    @patch("seedance25_nodes._upload_image", return_value="https://image.test/start.jpg")
    @patch("seedance25_nodes._first_frame", return_value=torch.zeros(1, 2, 2, 3))
    @patch("seedance25_nodes._poll", return_value={"outputs": ["https://video.test/i2v.mp4"]})
    @patch("seedance25_nodes._submit", return_value="i2v-request")
    def test_i2v_sends_single_image_url(self, submit, _poll, _frame, upload):
        image = torch.from_numpy(np.zeros((1, 2, 2, 3), dtype=np.float32))
        Seedance25ImageToVideo().run(
            "Animate the image",
            "720p",
            "9:16",
            "basic",
            5,
            True,
            False,
            -1,
            "mp4",
            api_key="test-key",
            image=image,
        )
        self.assertEqual(upload.call_count, 1)
        payload = submit.call_args.args[2]
        self.assertEqual(payload["image_url"], "https://image.test/start.jpg")
        self.assertNotIn("images_list", payload)

    @patch("seedance25_nodes._upload_image", side_effect=["https://image.test/first.jpg", "https://image.test/last.jpg"])
    @patch("seedance25_nodes._first_frame", return_value=torch.zeros(1, 2, 2, 3))
    @patch("seedance25_nodes._poll", return_value={"outputs": ["https://video.test/transition.mp4"]})
    @patch("seedance25_nodes._submit", return_value="transition-request")
    def test_first_last_sends_exactly_two_images(self, submit, _poll, _frame, upload):
        image = torch.zeros(1, 2, 2, 3)
        Seedance25FirstLastFrame().run(
            "Transition",
            "720p",
            "16:9",
            "basic",
            5,
            True,
            False,
            -1,
            "mp4",
            api_key="test-key",
            first_image=image,
            last_image=image,
        )
        self.assertEqual(submit.call_args.args[1], "seedance-2.5-first-last-frame")
        self.assertEqual(
            submit.call_args.args[2]["images_list"],
            ["https://image.test/first.jpg", "https://image.test/last.jpg"],
        )

    @patch("seedance25_nodes._first_frame", return_value=torch.zeros(1, 2, 2, 3))
    @patch("seedance25_nodes._poll", return_value={"outputs": ["https://video.test/omni.mp4"]})
    @patch("seedance25_nodes._submit", return_value="omni-request")
    def test_omni_uses_25_reference_field_names(self, submit, _poll, _frame):
        Seedance25OmniReference().run(
            "A scene",
            "720p",
            "16:9",
            "basic",
            5,
            True,
            False,
            -1,
            "mp4",
            api_key="test-key",
            video_url_1="https://video.test/ref.mp4",
            audio_url_1="https://audio.test/ref.mp3",
        )
        payload = submit.call_args.args[2]
        self.assertEqual(payload["videos_list"], ["https://video.test/ref.mp4"])
        self.assertEqual(payload["audios_list"], ["https://audio.test/ref.mp3"])
        self.assertNotIn("video_files", payload)
        self.assertNotIn("audio_files", payload)


if __name__ == "__main__":
    unittest.main()
