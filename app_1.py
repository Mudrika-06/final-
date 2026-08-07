import os
import time
import io
import urllib.parse
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from gtts import gTTS

# RAG Libraries
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Brand Voice & Ad Generator",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------------------------------------------
# 2. RAG SYSTEM SETUP
# -------------------------------------------------------------------
@st.cache_resource
def init_rag_system():
    brand_knowledge_base = [
        "Instagram Reels Rule: Hooks must appear within the first 3 seconds. Use bright contrast and clear text overlays.",
        "Sephora Hydrate Brand Voice: Fresh, bubbly, scientific yet accessible, clean aesthetic, emphasizing moisture and dewy glass skin.",
        "TikTok Reel Rule: Keep pacing fast, dynamic lighting changes, authentic conversational audio, 15 to 30 second optimal duration.",
        "Luxury Brand Tone: Slow cinematic motion, deep elegant voiceover, minimal clutter, focal lighting on product textures.",
        "Call to Action Strategy: Always offer clear single-step CTAs like 'Tap to Shop' or 'Link in Bio'."
    ]
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(brand_knowledge_base)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return embedder, index, brand_knowledge_base

embedder, faiss_index, knowledge_base = init_rag_system()

def query_rag(query_text: str, k: int = 2) -> str:
    """Retrieves relevant context from the Brand Knowledge Base."""
    try:
        query_vector = embedder.encode([query_text]).astype('float32')
        distances, indices = faiss_index.search(query_vector, k)
        results = [knowledge_base[i] for i in indices[0] if i < len(knowledge_base)]
        return "\n".join(results)
    except Exception:
        return "Focus on clear product positioning, engaging hooks, and strong calls to action."

# -------------------------------------------------------------------
# 3. FREE GENERATION ENGINE FUNCTIONS
# -------------------------------------------------------------------

def generate_free_image(prompt: str) -> Image.Image:
    """Generates visual assets for free using Pollinations API (FLUX/SD engine)."""
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=1024&nologo=true&seed=42"
    
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to fetch image from Pollinations API (Status: {response.status_code})")

def animate_image_to_video_gif(base_img: Image.Image, num_frames: int = 15) -> bytes:
    """Generates an animated video/GIF reel from the product photo locally."""
    base = base_img.convert("RGB").resize((480, 640))
    frames = []

    for i in range(num_frames):
        # Create subtle pan/zoom effect
        zoom_factor = 1.0 + (i * 0.015)
        w, h = base.size
        nw, nh = int(w * zoom_factor), int(h * zoom_factor)
        resized = base.resize((nw, nh), Image.Resampling.LANCZOS)
        
        # Crop centered
        left = (nw - w) // 2
        top = (nh - h) // 2
        frame = resized.crop((left, top, left + w, top + h))
        
        # Dynamic pulse adjustment
        brightness = 1.0 + (0.05 * np.sin(i * 0.5))
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(brightness)
        
        frames.append(frame)

    output_buffer = io.BytesIO()
    frames[0].save(
        output_buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0
    )
    output_buffer.seek(0)
    return output_buffer.getvalue()

def generate_voiceover(text: str) -> io.BytesIO:
    """Generates localized speech audio via gTTS."""
    clean_text = text.replace("*", "").replace("#", "").strip()
    tts = gTTS(text=clean_text[:250], lang='en', slow=False)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# -------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------

st.title("⚡ Multi-Modal Ad & Video Reel Generator")
st.caption("Live Demo Mode — Free Visual, Audio & Animated Video Generation Engine")

col_left, col_right = st.columns([1, 1])

with col_left:
    brand_name = st.text_input("Brand Name", value="Sephora")
    product_name = st.text_input("Product Name", value="Hydrate Dewy Serum")
    user_context = st.text_area(
        "Product & Motion Style Instructions",
        value="Hydrating skincare product with water droplets, dynamic studio lighting, cinematic glass skin aesthetic."
    )
    uploaded_file = st.file_uploader("Upload Base Product Photo", type=["jpg", "png", "jpeg"])

with col_right:
    target_audience = st.selectbox("Target Audience", ["Young Women (18-30)", "Skincare Enthusiasts", "Luxury Shoppers", "Gen-Z"])
    brand_tone = st.selectbox("Brand Tone", ["Luxury", "Casual & Energetic", "Minimalist", "Informative & Scientific"])
    platform = st.selectbox("Platform", ["Instagram Reels", "TikTok", "YouTube Shorts"])
    reel_timing = st.slider("Reel Target Timing (Seconds)", min_value=5, max_value=30, value=15, step=5)

if uploaded_file:
    product_img = Image.open(uploaded_file)
    st.image(product_img, caption="Uploaded Product Photo", width=180)

if st.button("🚀 Run Live Presentation Workflow", type="primary"):
    if not uploaded_file:
        st.error("Please upload a product image to proceed.")
    else:
        try:
            # Step 1: Strategy Retrieval
            with st.spinner("🤖 Step 1/4: Querying RAG Knowledge Base..."):
                rag_context = query_rag(f"{brand_tone} {platform} {product_name}")
                
                st.success("Strategy Brief Ready!")
                st.markdown("### 📋 Campaign Brief & Generated Script")
                brief_markdown = f"""
                * **Platform**: {platform}
                * **Hook**: "Say goodbye to dry skin in under 3 seconds!"
                * **Script**: "Experience ultimate hydration with {brand_name}'s all-new {product_name}. Powered by glass-skin technology for 24-hour glow. Tap link to shop now!"
                * **RAG Rules Applied**: {rag_context}
                """
                st.markdown(brief_markdown)

            st.markdown("---")
            st.subheader("🎬 Generated Media Assets")
            
            col_img, col_audio = st.columns(2)
            
            # Step 2: Free Image Generation
            generated_photo = None
            with col_img:
                st.markdown("#### 🎨 Generated Commercial Asset")
                with st.spinner("Step 2/4: Rendering image via Free Visual Engine..."):
                    img_prompt = f"Studio commercial photography of {brand_name} {product_name}, {user_context}, photorealistic, 8k resolution"
                    try:
                        generated_photo = generate_free_image(img_prompt)
                        st.image(generated_photo, caption="Generated Commercial Visual", use_container_width=True)
                    except Exception as e:
                        st.warning("Visual fallback mode activated.")
                        generated_photo = product_img
                        st.image(generated_photo, caption="Processed Visual Asset", use_container_width=True)

            # Step 3: Audio Synthesis
            with col_audio:
                st.markdown("#### 🎙️ Voiceover Audio Track")
                with st.spinner("Step 3/4: Synthesizing Voiceover Audio..."):
                    audio_script = f"Experience ultimate hydration with {brand_name} {product_name}. Unlock your glow today."
                    audio_fp = generate_voiceover(audio_script)
                    st.audio(audio_fp, format="audio/mp3")

            # Step 4: Video Reel Generation
            st.markdown("---")
            st.markdown("#### 📹 Animated Motion Reel")
            with st.spinner("Step 4/4: Generating dynamic motion video reel..."):
                target_image = generated_photo if generated_photo else product_img
                gif_bytes = animate_image_to_video_gif(target_image)
                
                st.image(gif_bytes, caption="Generated Motion Video Reel", use_container_width=True)
                st.success("Campaign execution complete and ready for presentation!")

        except Exception as global_err:
            st.error(f"Error during presentation execution: {global_err}")
