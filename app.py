import os
import time
import io
import streamlit as st
from PIL import Image, ImageEnhance
from gtts import gTTS

# Google GenAI SDK
from google import genai
from google.genai import types

# RAG Libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Brand Voice & Multi-Modal Ad Generator",
    page_icon="✨",
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
    """Retrieves relevant context from the Brand Knowledge Base safely."""
    try:
        query_vector = embedder.encode([query_text]).astype('float32')
        distances, indices = faiss_index.search(query_vector, k)
        results = [knowledge_base[i] for i in indices[0] if i < len(knowledge_base)]
        return "\n".join(results)
    except Exception:
        return "Focus on clear product positioning, engaging hooks, and strong calls to action."

# -------------------------------------------------------------------
# 3. AGENT DEFINITIONS (NANO BANANA + VEO 2.0 INTEGRATION)
# -------------------------------------------------------------------

class CopywritingAgent:
    """Agent for analyzing product images + RAG rules to generate campaign briefs."""
    def __init__(self, client):
        self.client = client

    def run(self, image: Image.Image, brand: str, product: str, audience: str, tone: str, platform: str, length_sec: int, user_context: str, rag_context: str):
        prompt = f"""
        You are an expert Ad Strategist. Analyze the product image and generate a complete campaign brief.
        
        Inputs:
        - Brand Name: {brand}
        - Product Name: {product}
        - Target Audience: {audience}
        - Tone: {tone}
        - Platform: {platform}
        - Target Video Length: {length_sec} seconds
        - Custom User Notes: {user_context}

        Retrieved Guidelines:
        {rag_context}

        Provide a structured response containing:
        1. **Ad Copy & Caption**: Platform-ready text with hashtags.
        2. **Voiceover Script**: Exact spoken script tuned for a {length_sec}-second video.
        3. **Visual Prompt**: Detailed prompt to generate a studio quality commercial photo.
        4. **Video Motion Prompt**: Scene-by-scene motion prompt for dynamic video.
        """
        
        img_converted = image.convert('RGB')
        img_byte_arr = io.BytesIO()
        img_converted.save(img_byte_arr, format='JPEG')
        image_part = types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type='image/jpeg')

        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        last_exception = None

        for model_name in models_to_try:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[image_part, prompt]
                    )
                    return response.text
                except Exception as e:
                    last_exception = e
                    err_str = str(e)
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    else:
                        break

        raise last_exception

class ImageGenerationAgent:
    """Agent for generating visual assets using Google's Nano Banana model."""
    def __init__(self, client):
        self.client = client

    def run(self, prompt: str, base_image: Image.Image = None):
        # Using Nano Banana native image generation via Gemini Flash
        try:
            # Send prompt to generate image output using Nano Banana model
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )
            )
            
            # Extract generated image part
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        img_bytes = part.inline_data.data
                        return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            st.info(f"Nano Banana API Notice: {e}. Utilizing studio image enhancer as fallback.")

        # Fallback processing if API keys restrict direct image outputs
        if base_image:
            enhanced = base_image.convert('RGB')
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.25)
            enhancer_bright = ImageEnhance.Brightness(enhanced)
            return enhancer_bright.enhance(1.1)
        
        return None

class VoiceGenerationAgent:
    """Agent for converting script into MP3 voiceover."""
    def run(self, script_text: str):
        clean_text = script_text.replace("*", "").replace("#", "").strip()
        tts = gTTS(text=clean_text[:300], lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp

class VideoGenerationAgent:
    """Agent for animating generated product visuals using Google Veo 2.0."""
    def __init__(self, client):
        self.client = client

    def run(self, input_image: Image.Image, motion_prompt: str):
        try:
            img_converted = input_image.convert('RGB')
            img_byte_arr = io.BytesIO()
            img_converted.save(img_byte_arr, format='JPEG')
            
            image_part = types.Part.from_bytes(
                data=img_byte_arr.getvalue(),
                mime_type='image/jpeg'
            )

            # Send image-to-video task to Veo 2.0
            operation = self.client.models.generate_videos(
                model="veo-2.0-generate-001",
                prompt=f"Animate this product photography: {motion_prompt}",
                images=[image_part],
                config=types.GenerateVideosConfig(
                    aspect_ratio="9:16",
                    duration_seconds=5
                )
            )
            
            # Poll status until video rendering completes
            while not operation.done:
                time.sleep(8)
                operation = self.client.operations.get(operation)

            if operation.result and hasattr(operation.result, 'generated_videos'):
                video_bytes = operation.result.generated_videos[0].video.video_bytes
                return video_bytes
            return None
        except Exception as e:
            st.warning(f"Veo Video Generation Status: {e}")
            return None

# -------------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------------------------

with st.sidebar:
    st.subheader("🔑 API Setup")
    gemini_api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("### 🛠️ Active Architecture")
    st.markdown("- **Planner Agent**: Gemini 2.5 Flash")
    st.markdown("- **Visual Agent**: Nano Banana (Gemini Native Image)")
    st.markdown("- **Audio Agent**: gTTS Engine")
    st.markdown("- **Video Agent**: Google Veo 2.0")

st.title("✨ AI Brand Voice & Multi-Modal Ad Generator")
st.caption("Generate Ad Copy, Voiceovers, Nano Banana Visuals, and Veo Video Reels")

col_left, col_right = st.columns([1, 1])

with col_left:
    user_context = st.text_area(
        "Ad Customization / Motion Notes",
        value="Focus on hydration benefits, water splash droplets, cinematic slow zoom..."
    )
    brand_name = st.text_input("Brand Name", value="Sephora")
    product_name = st.text_input("Product Name", value="Hydrate Dewy Serum")
    uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

with col_right:
    target_audience = st.selectbox("Target Audience", ["Young Women (18-30)", "Skincare Enthusiasts", "Luxury Shoppers", "Gen-Z"])
    brand_tone = st.selectbox("Brand Tone", ["Luxury", "Casual & Energetic", "Minimalist", "Informative & Scientific"])
    platform = st.selectbox("Platform", ["Instagram Reels", "TikTok", "YouTube Shorts"])
    reel_timing = st.slider("Reel Timing (Seconds)", min_value=5, max_value=60, value=15, step=5)

if uploaded_file:
    product_img = Image.open(uploaded_file)
    st.image(product_img, caption="Uploaded Product Photo", width=200)

# Execution Workflow
if st.button("🚀 Generate Campaign & Animated Reel", type="primary"):
    if not gemini_api_key:
        st.error("Please provide your Google Gemini API Key in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a product image.")
    else:
        try:
            client = genai.Client(api_key=gemini_api_key)
            
            # Step 1: Strategy & Script
            with st.spinner("🤖 Step 1/4: Analyzing Image & Generating Strategy..."):
                rag_knowledge = query_rag(f"{brand_tone} {platform} {product_name}")
                copywriter = CopywritingAgent(client)
                plan_output = copywriter.run(
                    image=product_img,
                    brand=brand_name,
                    product=product_name,
                    audience=target_audience,
                    tone=brand_tone,
                    platform=platform,
                    length_sec=reel_timing,
                    user_context=user_context,
                    rag_context=rag_knowledge
                )
                
                st.success("Campaign Strategy Generated!")
                st.markdown("### 📋 Campaign Brief & Script")
                st.write(plan_output)

            st.markdown("---")
            st.subheader("🎬 Generated Media Assets")
            
            col_img, col_audio = st.columns(2)
            
            # Step 2: Nano Banana Image Generation
            generated_photo = None
            with col_img:
                st.markdown("#### 🎨 Nano Banana Visual Asset")
                with st.spinner("Step 2/4: Generating image with Nano Banana..."):
                    img_agent = ImageGenerationAgent(client)
                    img_prompt = f"Studio commercial photography of {brand_name} {product_name}, luxury lighting, realistic 8k resolution"
                    generated_photo = img_agent.run(img_prompt, base_image=product_img)
                    if generated_photo:
                        st.image(generated_photo, caption="Nano Banana Generated Asset", use_container_width=True)

            # Step 3: Voiceover Generation
            with col_audio:
                st.markdown("#### 🎙️ Voiceover Audio")
                with st.spinner("Step 3/4: Synthesizing Audio Track..."):
                    try:
                        voice_agent = VoiceGenerationAgent()
                        sample_script = f"Discover deep hydration with the all-new {brand_name} {product_name}. Experience glass skin today."
                        audio_data = voice_agent.run(sample_script)
                        st.audio(audio_data, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Voice Agent Error: {e}")

            # Step 4: Veo 2.0 Video Generation
            st.markdown("---")
            st.markdown("#### 📹 Veo Animated Video Reel")
            with st.spinner("Step 4/4: Animating image into video reel using Veo 2.0..."):
                video_agent = VideoGenerationAgent(client)
                target_image = generated_photo if generated_photo else product_img
                motion_desc = f"Slow camera zoom into {product_name}, dynamic studio lighting and soft water ripple background motion."
                
                video_bytes = video_agent.run(target_image, motion_desc)
                if video_bytes:
                    st.video(video_bytes)
                    st.success("Veo Video generated successfully!")

        except Exception as global_err:
            st.error(f"An unexpected error occurred: {global_err}")
