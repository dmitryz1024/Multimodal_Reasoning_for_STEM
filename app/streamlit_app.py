"""
Streamlit application for Handwritten Formula to LaTeX conversion.
"""

import io
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from huggingface_hub import login

matplotlib.use("Agg")

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.inference import LatexOCRInference


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        login(hf_token)
        print("Successfully logged in to Hugging Face")
    except Exception as exc:
        print(f"Failed to login to Hugging Face: {exc}")


DEFAULT_TRAINED_CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "checkpoints" / "sft_latex_ocr_only" / "final",
    REPO_ROOT / "checkpoints" / "local_gpu_run" / "final",
    REPO_ROOT / "checkpoints" / "sft_latex_ocr_mathwriting" / "final",
]
NO_FORMULA_MESSAGE = "No LaTeX-convertible formula detected."


def get_default_checkpoint_path() -> str:
    env_path = os.getenv("DEFAULT_CHECKPOINT_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    for candidate in DEFAULT_TRAINED_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return str(candidate.resolve())

    return ""


def is_latex_formula(text: str) -> bool:
    """
    Check if the generated text appears to be a LaTeX mathematical formula.
    """
    text = text.strip()
    if not text or len(text) < 3:
        return False

    latex_indicators = [
        "\\",
        "$",
        "^",
        "_",
        "\\frac",
        "\\int",
        "\\sum",
        "\\prod",
        "\\sqrt",
        "\\log",
        "\\ln",
        "\\sin",
        "\\cos",
        "\\tan",
        "\\alpha",
        "\\beta",
        "\\gamma",
        "\\delta",
        "\\epsilon",
        "\\theta",
        "\\pi",
        "\\sigma",
        "\\phi",
        "\\omega",
        "\\leq",
        "\\geq",
        "\\neq",
        "\\approx",
        "\\infty",
        "\\partial",
        "\\nabla",
        "\\in",
        "\\notin",
        "\\subset",
        "\\supset",
    ]
    has_latex_symbols = any(indicator in text for indicator in latex_indicators)

    words = text.split()
    regular_words = [word for word in words if word.isalpha() and len(word) > 3]
    has_regular_words = len(regular_words) > 2

    if has_latex_symbols and not has_regular_words:
        return True
    if has_regular_words and not has_latex_symbols:
        return False
    if len(text) > 200:
        return False

    math_patterns = ["=", "+", "-", "*", "/", "(", ")", "[", "]", "{", "}", "|"]
    math_chars = sum(1 for char in text if char in math_patterns)
    return math_chars > len(text) * 0.3


st.set_page_config(
    page_title="Handwritten LaTeX OCR",
    page_icon="LaTeX",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_name: str, checkpoint_path: str | None = None):
    try:
        return LatexOCRInference(model_name=model_name, adapter_path=checkpoint_path)
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def render_latex(latex: str, fontsize: int = 20) -> Image.Image:
    latex = latex.strip()
    if not latex.startswith("$") and not latex.startswith("\\["):
        latex = f"${latex}$"

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_axis_off()

    try:
        ax.text(
            0.5,
            0.5,
            latex,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="center",
            horizontalalignment="center",
            usetex=False,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150, facecolor="white")
        buf.seek(0)
        img = Image.open(buf)
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Rendering error: {str(exc)[:50]}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="center",
            color="red",
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150, facecolor="white")
        buf.seek(0)
        img = Image.open(buf)

    plt.close(fig)
    return img


def main():
    default_checkpoint_path = get_default_checkpoint_path()

    st.markdown('<div class="main-header">Handwritten Formula to LaTeX</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Convert handwritten mathematical formulas to LaTeX code</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")

        model_options = {
            "Qwen2-VL-2B (Accurate)": "Qwen/Qwen2-VL-2B-Instruct",
            "SmolVLM-256M (Fast)": "HuggingFaceTB/SmolVLM-256M-Instruct",
        }
        selected_model = st.selectbox("Select Model", options=list(model_options.keys()), index=0)
        model_name = model_options[selected_model]

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value=default_checkpoint_path,
            help="Path to a fine-tuned model checkpoint. Use a trained checkpoint for the final internship demo.",
        )
        if not checkpoint_path:
            checkpoint_path = None

        if checkpoint_path:
            st.caption("A fine-tuned checkpoint is selected for inference.")
        else:
            st.warning("No fine-tuned checkpoint selected. For the final demo, point this to a trained checkpoint.")

        st.subheader("Generation Settings")
        max_tokens = st.slider("Max Tokens", min_value=64, max_value=512, value=256, step=32)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        use_one_shot = st.checkbox("Use One-shot Prompting", value=False)

        if st.button("Load/Reload Model"):
            st.cache_resource.clear()
            st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader(
            "Upload an image of a handwritten formula",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
        )
        camera_image = st.camera_input("Or take a photo")

        if camera_image is not None:
            image_source = camera_image
        elif uploaded_file is not None:
            image_source = uploaded_file
        else:
            image_source = None

        if image_source is not None:
            image = Image.open(image_source).convert("RGB")
            st.image(image, caption="Input Image", width="stretch")

            if st.button("Convert to LaTeX", type="primary"):
                with st.spinner("Loading model and processing..."):
                    engine = load_model(model_name, checkpoint_path)
                    if engine is not None:
                        with st.spinner("Generating LaTeX..."):
                            latex_output = engine.predict(
                                image=image,
                                use_one_shot=use_one_shot,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                            )

                        if not is_latex_formula(latex_output):
                            latex_output = NO_FORMULA_MESSAGE

                        st.session_state["latex_output"] = latex_output
                        st.session_state["has_result"] = True

    with col2:
        st.subheader("Output")
        if st.session_state.get("has_result", False):
            latex_output = st.session_state.get("latex_output", "")
            st.markdown("**LaTeX Code:**")
            st.code(latex_output, language="latex")

            st.button(
                "Copy to Clipboard",
                on_click=lambda: st.write(
                    f'<script>navigator.clipboard.writeText("{latex_output}")</script>',
                    unsafe_allow_html=True,
                ),
            )

            if latex_output != NO_FORMULA_MESSAGE:
                st.markdown("**Rendered Formula:**")
                try:
                    rendered_img = render_latex(latex_output)
                    st.image(rendered_img, caption="Rendered LaTeX", width="stretch")
                except Exception as exc:
                    st.error(f"Could not render LaTeX: {exc}")
                    st.info("The LaTeX code is still valid, but rendering failed.")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button(
                        "Download LaTeX",
                        data=latex_output,
                        file_name="formula.tex",
                        mime="text/plain",
                    )
                with col_b:
                    if "rendered_img" in locals():
                        buf = io.BytesIO()
                        rendered_img.save(buf, format="PNG")
                        st.download_button(
                            "Download Image",
                            data=buf.getvalue(),
                            file_name="formula.png",
                            mime="image/png",
                        )
            else:
                st.info("Upload an image with a handwritten formula to convert it into LaTeX.")
        else:
            st.info("Upload an image and click 'Convert to LaTeX' to see results.")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Built by Zinoviev Dmitry using Streamlit and HuggingFace Transformers
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
