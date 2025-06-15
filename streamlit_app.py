"""
AI Image Studio – local Stable Diffusion
---------------------------------------
Single-file Streamlit app that runs Stable Diffusion v1-5 locally.
"""

import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import contextlib, io, time, torch

# ---------------------------------------------------------------------
# ⚠️  HARD DEPENDENCY CHECK
# ---------------------------------------------------------------------
# StableDiffusionPipeline relies on Hugging Face 🤗 Transformers.
# Raise a friendly error early if it's missing.
try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    st.error(
        "The 'transformers' library is missing. "
        "Run `pip install transformers` (or use the requirements.txt below) "
        "and restart the app."
    )
    st.stop()

from diffusers import StableDiffusionPipeline


# ---------------------------------------------------------------------
# 📦  Local Stable Diffusion loader (cached)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5",
                  device: str | None = None) -> StableDiffusionPipeline:
    """
    Download & cache the Stable Diffusion pipeline once.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    # (Optional) disable safety checker for speed
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = lambda images, **kwargs: (images, False)

    return pipe


def generate_image(prompt: str,
                   guidance_scale: float = 7.5,
                   steps: int = 30,
                   seed: int | None = None) -> Image.Image:
    pipe = load_pipeline()
    device = pipe.device
    gen = (torch.Generator(device=device).manual_seed(seed)
           if seed is not None else None)

    with torch.autocast(device.type) if device.type == "cuda" else contextlib.nullcontext():
        out = pipe(prompt,
                   guidance_scale=guidance_scale,
                   num_inference_steps=steps,
                   generator=gen)
    return out.images[0]


# ---------------------------------------------------------------------
# 🎨  Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Image Studio (Local)",
                   page_icon="🎨", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------- styles ----------
st.markdown("""
<style>
.main-header{ text-align:center;padding:2rem 0;
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:#fff;border-radius:15px;margin-bottom:2rem;
  box-shadow:0 10px 30px rgba(0,0,0,.1);}
.main-header h1{font-size:3rem;margin:0;font-weight:700;}
.processing{ text-align:center;padding:2rem;
  background:linear-gradient(45deg,#ff9a9e 0%,#fecfef 50%,#fecfef 100%);
  border-radius:15px;color:#333;}
.stButton>button{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:#fff;border:none;padding:.75rem 2rem;border-radius:25px;
  font-weight:600;font-size:1.1rem;cursor:pointer;width:100%;
  transition:all .3s ease;}
.stButton>button:hover{ transform:translateY(-2px);
  box-shadow:0 5px 15px rgba(0,0,0,.2);}
</style>
""", unsafe_allow_html=True)


def apply_edit(image: Image.Image, prompt: str) -> Image.Image:
    """Very lightweight local effects – feel free to extend."""
    img = image.copy()
    p = prompt.lower()
    if "blur" in p or "dream" in p:
        img = img.filter(ImageFilter.GaussianBlur(2))
    elif "sharp" in p or "enhance" in p:
        img = img.filter(ImageFilter.SHARPEN)
    elif "bright" in p or "sunny" in p:
        img = ImageEnhance.Brightness(img).enhance(1.3)
    elif "dark" in p or "shadow" in p:
        img = ImageEnhance.Brightness(img).enhance(0.7)
    elif "colorful" in p or "vibrant" in p:
        img = ImageEnhance.Color(img).enhance(1.5)
    elif "vintage" in p or "sepia" in p:
        img = ImageOps.colorize(ImageOps.grayscale(img), '#704214', '#C0A882')
    elif "mono" in p or "black" in p or "white" in p:
        img = ImageOps.grayscale(img)
    else:
        img = ImageEnhance.Color(img).enhance(1.2)
    return img


def spinner():
    ph = st.empty()
    steps = ["🔍 Analyzing prompt…", "🧠 Running model…",
             "✨ Painting pixels…", "🚀 Almost done…"]
    for i, txt in enumerate(steps):
        ph.markdown(
            f"<div class='processing'><h3>{txt}</h3>"
            f"<div style='width:{25*(i+1)}%;height:4px;"
            "background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);"
            "border-radius:2px;margin:1rem auto;'></div></div>",
            unsafe_allow_html=True
        )
        time.sleep(0.8)
    ph.empty()


# ---------- header ----------
st.markdown("""<div class='main-header'>
<h1>🎨 AI Image Studio</h1>
<p>Generate and edit images entirely on your own machine.</p>
</div>""", unsafe_allow_html=True)

tab_generate, tab_edit = st.tabs(["🖼️ Generate", "🛠️ Edit"])

# ---------- generate tab ----------
with tab_generate:
    prompt = st.text_area("Enter a creative prompt",
                           height=100,
                           placeholder="e.g. A serene landscape in neon cyberpunk style")
    col_seed, col_scale, col_steps = st.columns(3)
    with col_seed:
        seed = st.number_input("Seed (optional)", value=0, step=1)
        seed = None if seed == 0 else int(seed)
    with col_scale:
        scale = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)
    with col_steps:
        steps = st.slider("Steps", 10, 60, 30, 5)

    if st.button("🚀 Generate Image", use_container_width=True):
        if not prompt.strip():
            st.error("Please enter a prompt.")
        else:
            spinner()
            try:
                image = generate_image(prompt,
                                       guidance_scale=scale,
                                       steps=steps,
                                       seed=seed)
                st.image(image, caption="Generated image", use_column_width=True)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button("📥 Download",
                                   data=buf.getvalue(),
                                   file_name="generated.png",
                                   mime="image/png")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ---------- edit tab ----------
with tab_edit:
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    edit_prompt = st.text_input("Describe your transformation")

    if st.button("✨ Transform", use_container_width=True):
        if not up or not edit_prompt.strip():
            st.error("Please upload an image and enter a prompt.")
        else:
            spinner()
            original = Image.open(up)
            transformed = apply_edit(original, edit_prompt)
            st.image(original, caption="Original", use_column_width=True)
            st.image(transformed, caption="Transformed", use_column_width=True)
            buf = io.BytesIO()
            transformed.save(buf, format="PNG")
            st.download_button("📥 Download",
                               data=buf.getvalue(),
                               file_name="transformed.png",
                               mime="image/png")


# ---------- footer ----------
st.markdown("""
---
<div style='text-align:center;color:#666;padding:2rem;'>
Made with ❤️ using Streamlit, Diffusers & Transformers
</div>""", unsafe_allow_html=True)
