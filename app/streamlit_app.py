"""
Streamlit application for Handwritten Formula to LaTeX conversion.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import our modules
from src.inference import LatexOCRInference


# Page configuration
st.set_page_config(
    page_title="Handwritten LaTeX OCR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .latex-output {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 1.1rem;
    }
    .rendered-formula {
        background-color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str, checkpoint_path: str = None):
    """Load model with caching."""
    try:
        engine = LatexOCRInference(
            model_name=model_name,
            adapter_path=checkpoint_path
        )
        return engine
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def render_latex(latex: str, fontsize: int = 20) -> Image.Image:
    """
    Render LaTeX formula to image using matplotlib.
    
    Args:
        latex: LaTeX string
        fontsize: Font size for rendering
    
    Returns:
        PIL Image of rendered formula
    """
    # Clean up LaTeX
    latex = latex.strip()
    
    # Add math mode if not present
    if not latex.startswith('$') and not latex.startswith('\\['):
        latex = f'${latex}$'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_axis_off()
    
    try:
        # Render LaTeX
        ax.text(
            0.5, 0.5,
            latex,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='center',
            horizontalalignment='center',
            usetex=False  # Use matplotlib's mathtext
        )
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0.1, dpi=150, facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        
    except Exception as e:
        # If rendering fails, show error message
        ax.text(
            0.5, 0.5,
            f"Rendering error: {str(e)[:50]}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='center',
            color='red'
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0.1, dpi=150, facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
    
    plt.close(fig)
    return img


def main():
    # Header
    st.markdown('<div class="main-header">📝 Handwritten Formula → LaTeX</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Convert handwritten mathematical formulas to LaTeX code</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = {
            "SmolVLM-256M (Fast)": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "Qwen2-VL-2B (Accurate)": "Qwen/Qwen2-VL-2B-Instruct",
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        model_name = model_options[selected_model]
        
        # Checkpoint path
        checkpoint_path = st.text_input(
            "Checkpoint Path (optional)",
            value="",
            help="Path to fine-tuned model checkpoint"
        )
        
        if not checkpoint_path:
            checkpoint_path = None
        
        # Generation settings
        st.subheader("Generation Settings")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=512,
            value=256,
            step=32
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        )
        
        use_one_shot = st.checkbox(
            "Use One-shot Prompting",
            value=False,
            help="Include an example in the prompt"
        )
        
        # Load model button
        if st.button("🔄 Load/Reload Model"):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image of a handwritten formula",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            help="Supported formats: PNG, JPG, JPEG, BMP, WEBP"
        )
        
        # Camera input
        camera_image = st.camera_input(
            "Or take a photo",
            help="Use your camera to capture a handwritten formula"
        )
        
        # Use camera image if no file uploaded
        if camera_image is not None:
            image_source = camera_image
        elif uploaded_file is not None:
            image_source = uploaded_file
        else:
            image_source = None
        
        if image_source is not None:
            # Display uploaded image
            image = Image.open(image_source).convert("RGB")
            st.image(image, caption="Input Image", width="stretch")
            
            # Convert button
            if st.button("🔮 Convert to LaTeX", type="primary"):
                with st.spinner("Loading model and processing..."):
                    # Load model
                    engine = load_model(model_name, checkpoint_path)
                    
                    if engine is not None:
                        # Run inference
                        with st.spinner("Generating LaTeX..."):
                            latex_output = engine.predict(
                                image=image,
                                use_one_shot=use_one_shot,
                                max_new_tokens=max_tokens,
                                temperature=temperature
                            )
                        
                        # Store result in session state
                        st.session_state['latex_output'] = latex_output
                        st.session_state['has_result'] = True
    
    with col2:
        st.subheader("📥 Output")
        
        if st.session_state.get('has_result', False):
            latex_output = st.session_state.get('latex_output', '')
            
            # LaTeX code output
            st.markdown("**LaTeX Code:**")
            st.code(latex_output, language="latex")
            
            # Copy button
            st.button(
                "📋 Copy to Clipboard",
                on_click=lambda: st.write(
                    f'<script>navigator.clipboard.writeText("{latex_output}")</script>',
                    unsafe_allow_html=True
                )
            )
            
            # Rendered output
            st.markdown("**Rendered Formula:**")
            try:
                rendered_img = render_latex(latex_output)
                st.image(rendered_img, caption="Rendered LaTeX", width="stretch")
            except Exception as e:
                st.error(f"Could not render LaTeX: {e}")
                st.info("The LaTeX code is still valid, but rendering failed.")
            
            # Download buttons
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "💾 Download LaTeX",
                    data=latex_output,
                    file_name="formula.tex",
                    mime="text/plain"
                )
            with col_b:
                if 'rendered_img' in dir():
                    buf = io.BytesIO()
                    rendered_img.save(buf, format='PNG')
                    st.download_button(
                        "🖼️ Download Image",
                        data=buf.getvalue(),
                        file_name="formula.png",
                        mime="image/png"
                    )
        else:
            st.info("👆 Upload an image and click 'Convert to LaTeX' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Built by Zinoviev Dmitry using Streamlit and HuggingFace Transformers
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
