
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torch
from diffusers import StableDiffusionPipeline
import contextlib, io, time
import random

# --------------------------------------------------------------------------
# 📦  Local Stable Diffusion Generator
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5",
                  device: str | None = None) -> StableDiffusionPipeline:
    """Load the Stable Diffusion pipeline once and cache it."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    # Disable safety checker for speed (optional)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = lambda images, **kwargs: (images, False)
    return pipe

def generate_image(prompt: str,
                   guidance_scale: float = 7.5,
                   steps: int = 30,
                   seed: int | None = None) -> Image.Image:
    pipe = load_pipeline()
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    with torch.autocast(device.type) if device.type == "cuda" else contextlib.nullcontext():
        result = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        )
    return result.images[0]

# --------------------------------------------------------------------------
# 🎨  Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Image Studio (Local)",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""<style>
.main-header {text-align:center;padding:2rem 0;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:white;border-radius:15px;margin-bottom:2rem;
    box-shadow:0 10px 30px rgba(0,0,0,.1);}
.main-header h1{font-size:3rem;margin:0;font-weight:700;}
.upload-section,.result-section,.feature-card{
    background:white;padding:2rem;border-radius:15px;
    box-shadow:0 5px 20px rgba(0,0,0,.1);margin-bottom:2rem;}
.prompt-section{
    background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);
    padding:2rem;border-radius:15px;margin-bottom:2rem;color:white;}
.processing-animation{text-align:center;padding:2rem;
    background:linear-gradient(45deg,#ff9a9e 0%,#fecfef 50%,#fecfef 100%);
    border-radius:15px;color:#333;}
.stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    color:white;border:none;padding:.75rem 2rem;border-radius:25px;
    font-weight:600;font-size:1.1rem;cursor:pointer;transition:all .3s ease;width:100%;}
.stButton>button:hover{transform:translateY(-2px);
    box-shadow:0 5px 15px rgba(0,0,0,.2);}
</style>""", unsafe_allow_html=True)

# ------------------------  Helper functions  -------------------------------
def apply_mock_transformation(image: Image.Image, prompt: str) -> Image.Image:
    img = image.copy()
    p = prompt.lower()
    if any(w in p for w in ['blur', 'dream']):
        img = img.filter(ImageFilter.GaussianBlur(2))
    elif any(w in p for w in ['sharp', 'enhance']):
        img = img.filter(ImageFilter.SHARPEN)
    elif any(w in p for w in ['bright', 'sunny']):
        img = ImageEnhance.Brightness(img).enhance(1.3)
    elif any(w in p for w in ['dark', 'shadow']):
        img = ImageEnhance.Brightness(img).enhance(0.7)
    elif any(w in p for w in ['colorful', 'vibrant']):
        img = ImageEnhance.Color(img).enhance(1.5)
    elif any(w in p for w in ['vintage', 'sepia']):
        img = ImageOps.colorize(ImageOps.grayscale(img), '#704214', '#C0A882')
    elif any(w in p for w in ['mono', 'black', 'white']):
        img = ImageOps.grayscale(img)
    else:
        img = ImageEnhance.Color(img).enhance(1.2)
    return img

def create_processing_animation():
    container = st.empty()
    msgs = [
        "🔍 Analyzing prompt...",
        "🧠 Running Stable Diffusion locally...",
        "✨ Painting pixels...",
        "🚀 Almost done..."
    ]
    for i, m in enumerate(msgs):
        container.markdown(f"""<div class='processing-animation'>
        <h3>{m}</h3>
        <div style='width:{(i+1)*25}%;height:4px;
        background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
        border-radius:2px;margin:1rem auto;'></div></div>""", unsafe_allow_html=True)
        time.sleep(0.8)
    container.empty()

# -----------------------------  Layout  ------------------------------------
st.markdown("""<div class='main-header'>
    <h1>🎨 AI Image Studio</h1>
    <p>Generate & transform images with on‑device Stable Diffusion.</p>
</div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ Generate", "🛠️ Edit (local)"])

# -------------------------  Tab 1 – Generate  ------------------------------
with tab1:
    prompt_gen = st.text_area("Enter a creative prompt", height=100,
                               placeholder="e.g., A serene landscape in neon cyberpunk style")
    seed_col, scale_col, steps_col = st.columns(3)
    with seed_col:
        seed_val = st.number_input("Seed (optional)", value=0, step=1)
        if seed_val == 0:
            seed_val = None
    with scale_col:
        scale_val = st.slider("Guidance scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
    with steps_col:
        steps_val = st.slider("Steps", min_value=10, max_value=60, value=30, step=5)

    if st.button("🚀 Generate Image", use_container_width=True):
        if not prompt_gen.strip():
            st.error("Please enter a prompt.")
        else:
            create_processing_animation()
            try:
                img = generate_image(prompt_gen, guidance_scale=scale_val,
                                     steps=steps_val, seed=seed_val)
                st.image(img, caption="Generated Image", use_column_width=True)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.download_button("📥 Download", data=buf.getvalue(),
                                   file_name="generated.png", mime="image/png")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# -------------------------  Tab 2 – Local Edit  ----------------------------
with tab2:
    uploaded = st.file_uploader("Upload an image to transform", type=['png','jpg','jpeg'])
    prompt_edit = st.text_input("Describe your transformation")
    cols = st.columns([1,1,1,1])
    with cols[1]:
        if st.button("✨ Transform", key="transform_local", use_container_width=True):
            if not uploaded or not prompt_edit.strip():
                st.error("Please upload an image and enter a prompt.")
            else:
                create_processing_animation()
                img_in = Image.open(uploaded)
                img_out = apply_mock_transformation(img_in, prompt_edit)
                st.image(img_in, caption="Original", use_column_width=True)
                st.image(img_out, caption="Transformed", use_column_width=True)
                buf = io.BytesIO()
                img_out.save(buf, format="PNG")
                st.download_button("📥 Download", data=buf.getvalue(),
                                   file_name="transformed.png", mime="image/png")

# -----------------------------  Footer  ------------------------------------
st.markdown("""---<div style='text-align:center;color:#666;padding:2rem;'>
Made with ❤️ using Streamlit & Stable Diffusion</div>""", unsafe_allow_html=True)
