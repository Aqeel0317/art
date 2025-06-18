import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
from datetime import datetime
import time
import math
import re
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Polished UI ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding: 1rem 2rem 2rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #5A67D8 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:disabled {
        background: #ccc;
        color: #888;
    }
    
    /* Main generate button styling */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        font-size: 1.25rem;
        padding: 0.75rem;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        box-shadow: 0 5px 20px rgba(40, 167, 69, 0.5);
    }
    
    /* Warning box */
    .token-warning {
        background: #fff3cd;
        border-left: 5px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .token-success {
        background: #d4edda;
        border-left: 5px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state variables if not already present."""
    for key, value in {
        'generated_image': None,
        'processing': False,
        'hf_token': "",
        'selected_preset_style_name': None,
        'style_image': None,
        'processing_method': 'simple'
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Simple Working HF API Call ---
def query_working_model(prompt, hf_token):
    """
    Use a simple text-to-image model that actually works.
    """
    # Using a model that's guaranteed to work
    api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        elif response.status_code == 503:
            # Model is loading
            return "loading"
        else:
            try:
                error_info = response.json()
                return f"Error: {error_info.get('error', 'Unknown error')}"
            except:
                return f"Error: HTTP {response.status_code}"
                
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Local Style Transfer Effects ---
def apply_simple_style_transfer(content_image, style_name, intensity=0.7):
    """
    Apply simple style effects using PIL - works without any API calls.
    """
    # Convert to RGB if needed
    if content_image.mode != 'RGB':
        content_image = content_image.convert('RGB')
    
    # Resize for processing
    img = content_image.copy()
    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    if "Van Gogh" in style_name:
        # Van Gogh style: Enhanced colors, slight blur, texture
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)  # Boost colors
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Boost contrast
        
        # Add slight blur for painting effect
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        
        # Add texture by overlaying with slight noise
        img_array = np.array(img)
        noise = np.random.randint(-15, 15, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
    elif "Picasso" in style_name:
        # Cubist style: Edge enhancement, geometric patterns
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Posterize for geometric effect
        img = img.quantize(colors=8, dither=0).convert('RGB')
        
    elif "Abstract" in style_name:
        # Abstract style: High contrast, vivid colors
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.8)
        
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.6)
        
        # Apply emboss filter
        img = img.filter(ImageFilter.EMBOSS)
        
    else:  # Japanese Wave
        # Serene, blue-tinted style
        img_array = np.array(img)
        
        # Add blue tint
        img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.2, 0, 255)  # Enhance blue
        img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.9, 0, 255)  # Reduce red slightly
        
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply smooth filter
        img = img.filter(ImageFilter.SMOOTH_MORE)
    
    return img

# --- Preset Style Image Generation ---
def create_sample_style_image(style_name):
    """Creates a simple colored pattern to represent a style."""
    img = Image.new('RGB', (256, 256))
    draw = ImageDraw.Draw(img)
    
    if "Van Gogh" in style_name:
        for i in range(256):
            for j in range(256):
                wave = math.sin(i * 0.05) * math.cos(j * 0.05)
                blue = int(100 + 100 * abs(wave))
                yellow = int(150 + 100 * wave) if wave > 0 else 50
                draw.point((i, j), (yellow, yellow, blue))
    elif "Picasso" in style_name:
        colors = [(255, 182, 193), (135, 206, 235), (255, 218, 185), (221, 160, 221)]
        for i in range(0, 256, 32):
            for j in range(0, 256, 32):
                color = colors[(i // 32 + j // 32) % len(colors)]
                draw.rectangle([i, j, i+32, j+32], fill=color)
                draw.line([i, j, i+32, j+32], fill=(0, 0, 0), width=2)
    elif "Abstract" in style_name:
        for i in range(256):
            for j in range(256):
                r = int(255 * abs(math.sin(i * 0.02)))
                g = int(255 * abs(math.cos(j * 0.02)))
                b = int(255 * abs(math.sin((i + j) * 0.01)))
                draw.point((i, j), (r, g, b))
    else:  # Japanese Wave
        for i in range(256):
            for j in range(256):
                wave1 = math.sin(i * 0.1) * 50
                wave2 = math.cos(j * 0.08) * 30
                blue = int(100 + abs(wave1 + wave2))
                white = int(200 + wave2) if wave2 > 0 else 100
                draw.point((i, j), (white, white, min(255, blue + 50)))
    return img

# --- New Function Placeholder ---
def new_function():
    pass

# --- Main App ---
def main():
    init_session_state()

    # --- Header ---
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Neural Style Transfer Studio</h1>
        <p class="header-subtitle">Transform your photos into stunning artworks using AI</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("⚙️ Controls & Settings")

        # Processing Method Selection
        st.subheader("🔧 Processing Method")
        processing_method = st.radio(
            "Choose processing approach:",
            ["🚀 Simple & Fast (Local Processing)", "🌐 AI-Powered (Requires Token)"],
            key="processing_method_radio",
            help="Simple method works instantly without tokens. AI method requires HuggingFace token but gives better results."
        )
        
        st.session_state.processing_method = "simple" if "Simple" in processing_method else "ai"

        # API Token Input (only show if AI method selected)
        if st.session_state.processing_method == "ai":
            st.subheader("Hugging Face API Token")
            st.session_state.hf_token = st.text_input(
                "Enter your HF Token", 
                type="password", 
                help="Get your token from https://huggingface.co/settings/tokens"
            )

            if st.session_state.hf_token:
                if st.session_state.hf_token.startswith("hf_"):
                     st.markdown('<div class="token-success">✅ Token format looks correct.</div>', unsafe_allow_html=True)
                else:
                     st.markdown('<div class="token-warning">⚠️ Invalid Token Format. It should start with "hf_".</div>', unsafe_allow_html=True)
        
        st.markdown("---")

        # Style Settings
        st.subheader("🎨 Style Settings")
        if st.session_state.processing_method == "simple":
            style_intensity = st.slider("Style Intensity", 0.3, 1.0, 0.7, 0.1, help="How strong the style effect should be")
        else:
            style_strength = st.slider("AI Style Strength", 5.0, 15.0, 7.5, 0.5, help="How much the AI should follow the style prompt")

    # --- Main Content Area ---
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Your Photo")
        content_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            key="content"
        )
        
        content_image = None
        if content_file:
            content_image = Image.open(content_file)
            st.image(content_image, caption="Your Photo", use_container_width=True)
        elif st.session_state.processing_method == "simple":
            st.info("💡 For simple processing, upload a photo to apply style effects!")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎨 Choose Art Style")
        
        # Preset styles
        PRESET_STYLES = {
            "🌻 Van Gogh": create_sample_style_image("Van Gogh"),
            "🎨 Picasso": create_sample_style_image("Picasso"), 
            "🎭 Abstract": create_sample_style_image("Abstract"),
            "🌊 Japanese Wave": create_sample_style_image("Japanese Wave"),
        }
        preset_cols = st.columns(2)
        for i, name in enumerate(PRESET_STYLES.keys()):
            with preset_cols[i % 2]:
                if st.button(name, key=f"preset_{i}"):
                    st.session_state.selected_preset_style_name = name
                    st.session_state.style_image = PRESET_STYLES[name]
        
        # Display the chosen style
        if st.session_state.style_image and st.session_state.selected_preset_style_name:
            st.image(st.session_state.style_image, caption=f"Style: {st.session_state.selected_preset_style_name}", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    
    # --- Generation Form and Button ---
    with st.form("generate_form"):
        submitted = st.form_submit_button("🎨 Generate Artwork", use_container_width=True, type="primary")
        
        if submitted:
            if not st.session_state.selected_preset_style_name:
                st.warning("⚠️ Please select an art style.")
            elif st.session_state.processing_method == "simple" and not content_image:
                st.warning("⚠️ Please upload a photo for simple processing.")
            elif st.session_state.processing_method == "ai" and (not st.session_state.hf_token or not st.session_state.hf_token.startswith("hf_")):
                st.error("❌ Please enter a valid Hugging Face API token for AI processing.")
            else:
                st.session_state.processing = True
                
                with st.spinner('✨ Creating your artwork...'):
                    if st.session_state.processing_method == "simple":
                        # Simple local processing
                        try:
                            result_image = apply_simple_style_transfer(
                                content_image, 
                                st.session_state.selected_preset_style_name,
                                style_intensity
                            )
                            
                            if result_image:
                                st.session_state.generated_image = result_image
                                st.toast("Artwork created successfully!", icon="🎉")
                                st.rerun()
                            else:
                                st.error("Failed to process image locally.")
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
                    
                    else:
                        # AI processing
                        clean_name = re.sub(r'[^\w\s]', '', st.session_state.selected_preset_style_name).strip()
                        if content_image:
                            prompt = f"A photograph in the artistic style of {clean_name}, masterpiece, high quality"
                        else:
                            prompt = f"A beautiful artwork in the style of {clean_name}, masterpiece, high quality, detailed"
                        
                        result = query_working_model(prompt, st.session_state.hf_token)
                        
                        if isinstance(result, Image.Image):
                            st.session_state.generated_image = result
                            st.toast("AI artwork generated successfully!", icon="🎉")
                            st.rerun()
                        elif result == "loading":
                            st.warning("🔄 AI model is loading. Please wait 30-60 seconds and try again.")
                        else:
                            st.error(f"AI generation failed: {result}")
                
                st.session_state.processing = False

    # --- Display Result ---
    if st.session_state.generated_image:
        st.markdown("### 🖼️ Your Generated Artwork")
        
        result_col1, result_col2 = st.columns([3, 1])
        
        with result_col1:
            st.image(st.session_state.generated_image, caption="Generated Artwork", use_container_width=True)
        
        with result_col2:
            # Prepare image for download
            img_bytes = io.BytesIO()
            st.session_state.generated_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Artwork",
                data=img_bytes,
                file_name=f"style_transfer_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Reset button to generate new artwork
            if st.button("🔄 Generate New", use_container_width=True):
                st.session_state.generated_image = None
                st.rerun()
    
    # --- Method Comparison ---
    with st.expander("🆚 Processing Methods Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 Simple & Fast**
            ✅ Works instantly  
            ✅ No token required  
            ✅ Always available  
            ✅ Privacy-friendly  
            ❌ Basic effects only  
            ❌ Requires uploaded photo  
            """)
        
        with col2:
            st.markdown("""
            **🌐 AI-Powered**
            ✅ High-quality results  
            ✅ Works without photo  
            ✅ Advanced AI effects  
            ❌ Requires HF token  
            ❌ May have delays  
            ❌ Internet dependent  
            """)
    
    # --- Help Section ---
    with st.expander("❓ Need Help?"):
        st.markdown("""
        **Getting Started:**
        1. **Simple Method**: Upload a photo → Select style → Generate (no token needed!)
        2. **AI Method**: Get HF token → Select style → Generate (works with or without photo)
        
        **HuggingFace Token Setup:**
        1. Go to https://huggingface.co/settings/tokens
        2. Click "New token"
        3. Choose "Read" permissions
        4. Copy the token (starts with 'hf_')
        
        **Troubleshooting:**
        - Try the Simple method first - it always works!
        - For AI method: wait if model is loading
        - Check your internet connection
        - Make sure token starts with 'hf_'
        """)
    
    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            Made with ❤️ using Streamlit | Try the Simple method for instant results!
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
