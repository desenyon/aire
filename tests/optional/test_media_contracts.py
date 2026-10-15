import importlib.util
import shutil
import subprocess

import pytest

from aire.core.content import ImageContent
from aire.core.types import Capability
from aire.models.base import run_sync
from aire.models.builtin import EchoModel
from aire.models.types import GenerationResult

pytestmark = pytest.mark.optional


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="install ffmpeg")
def test_real_video_frames_are_jpeg_and_cleaned_up(tmp_path):
    from aire.vision.video import VideoPipeline

    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:d=2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    class Vision(EchoModel):
        @property
        def info(self):
            return super().info.model_copy(update={"capabilities": [Capability.VISION_INPUT]})

        async def generate(self, request):
            images = [c for c in request.messages[0].content if isinstance(c, ImageContent)]
            assert images and all(c.data.startswith(b"\xff\xd8") for c in images)
            return GenerationResult.text_result("blue frames", model="test:vision")

    result = run_sync(VideoPipeline(Vision()).summarize(video, max_frames=2))
    assert result.metadata["mode"] == "vision_frames"
    assert not result.metadata["stub"]
    assert result.frames_used > 0


@pytest.mark.skipif(
    importlib.util.find_spec("pytesseract") is None or shutil.which("tesseract") is None,
    reason="install OCR profile and tesseract",
)
def test_real_scanned_pdf_ocr(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    from aire.docs.pdf import load_pdf

    image = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(image)
    font = (
        ImageFont.truetype("Arial.ttf", 60)
        if __import__("sys").platform == "darwin"
        else ImageFont.truetype("DejaVuSans.ttf", 60)
    )
    draw.text((40, 80), "AIRE DOCUMENT 2026", font=font, fill="black")
    path = tmp_path / "scan.pdf"
    image.save(path, "PDF")
    doc = load_pdf(path, ocr=True, raise_on_empty=True)
    assert "AIRE" in doc.text.upper()
    assert doc.pages[0].ocr
