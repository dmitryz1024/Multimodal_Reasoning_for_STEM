#!/bin/bash
# Run Streamlit application

set -e

echo "=============================================="
echo "Starting Handwritten LaTeX OCR App"
echo "=============================================="

# Default port
PORT="${PORT:-8501}"

# Run Streamlit
streamlit run app/streamlit_app.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --browser.gatherUsageStats false

