
"""
Lightweight Image Generator using Stability AI's REST API.
This avoids downloading large model weights locally.
"""
import requests
import base64
import io
from PIL import Image

# === SECURITY NOTICE =======================================================
# This key is embedded for convenience because the user explicitly requested
# it.  **Never commit production keys to public repositories.**
# ===========================================================================
API_KEY = "CfQRqqmkhNO7xCHmADtzTtR4k1CTa2gFDdWgE8pD0tqW0EGXhb9rWseHZhzu"


class ImageGeneratorAPI:
    """Minimal wrapper around Stability AI text‑to‑image endpoint."""

    def __init__(self,
                 engine: str = "stable-diffusion-xl-1024-v1-0",
                 base_url: str = "https://api.stability.ai"):
        self.engine = engine
        self.url = f"{base_url.rstrip('/')}/v1/generation/{self.engine}/text-to-image"
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def generate(self,
                 prompt: str,
                 width: int = 1024,
                 height: int = 1024,
                 steps: int = 30,
                 cfg_scale: float = 7.0,
                 seed: int | None = None) -> Image.Image:
        """Generate an image from a text prompt and return a PIL.Image."""
        payload = {
            "text_prompts": [
                {"text": prompt}
            ],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "samples": 1,
            "steps": steps
        }
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            b64 = data["artifacts"][0]["base64"]
        except (KeyError, IndexError):
            raise RuntimeError(f"Unexpected API response: {data}") from None

        img_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(img_bytes))
