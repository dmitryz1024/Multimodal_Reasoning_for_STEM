from setuptools import setup, find_packages

setup(
    name="multimodal_reasoning_for_stem",
    version="0.1.0",
    description="Vision-Language Model for Handwritten Formula to LaTeX Conversion",
    author="Dmitry Zinoviev",
    author_email="dmkozinovev@edu.hse.ru",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "datasets>=2.14.0",
        "Pillow>=10.0.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pytest",
        ],
    },
)
