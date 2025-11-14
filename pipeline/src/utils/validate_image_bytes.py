import io
import argparse
from typing import Tuple, Optional, Dict

import urllib.request
from PIL import Image, UnidentifiedImageError


# --- Fetch image bytes (same logic as your embedding pipeline) ---
def fetch_image_bytes(url: str, timeout: int = 10) -> bytes | None:
    """
    helper to read image from url
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status >= 400:
            return None
        return resp.read()



# --- Validate image bytes ---
def validate_image_bytes(
    raw: bytes,
    *,
    min_resolution: Tuple[int, int] = (64, 64),
    max_resolution: Tuple[int, int] = (8000, 8000),
    require_rgb: bool = False,
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    if not raw:
        return False, "Empty or unreadable bytes", None

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # corruption check
    except UnidentifiedImageError:
        return False, "Not an image / unsupported format", None
    except Exception as exc:
        return False, f"Corrupt or invalid image: {exc}", None

    # Must reopen after verify
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        return False, f"Image cannot be decoded: {exc}", None

    w, h = img.size
    mode = img.mode

    min_w, min_h = min_resolution
    max_w, max_h = max_resolution

    if w < min_w or h < min_h:
        return False, f"Image too small ({w}x{h})", {"width": w, "height": h, "mode": mode}

    if w > max_w or h > max_h:
        return False, f"Image too large ({w}x{h})", {"width": w, "height": h, "mode": mode}

    if require_rgb and mode not in ("RGB", "RGBA"):
        return False, f"Invalid mode: {mode}", {"width": w, "height": h, "mode": mode}

    return True, None, {"width": w, "height": h, "mode": mode}


def main():
    parser = argparse.ArgumentParser(description="Validate a single image URL.")
    parser.add_argument("url", type=str, help="The image URL to validate")

    args = parser.parse_args()
    url = args.url

    raw = fetch_image_bytes(url)
    if raw is None:
        print(f"[BAD]  {url} — Failed to fetch image bytes")
        return 1

    ok, reason, info = validate_image_bytes(raw)

    if ok:
        print(
            f"[OK]   {url} — {info['width']}x{info['height']} mode={info['mode']}"
        )
        return 0
    else:
        print(f"[BAD]  {url} — {reason}")
        return 1


###  python -m src.utils.validate_image_bytes "https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/c2/P00939495_b1.jpg"
### [OK]   https://img.mytheresa.com/2310/2610/100/jpeg/catalog/product/c2/P00939495_b1.jpg — 2160x2442 mode=RGB

if __name__ == "__main__":
    raise SystemExit(main())