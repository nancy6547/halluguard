FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HALLUGUARD_MODEL=cross-encoder/nli-deberta-v3-base
WORKDIR /home/user/app

RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user "streamlit>=1.35" "halluguard-nli @ git+https://github.com/nancy6547/halluguard"

# Download the model at build time so the first visitor doesn't wait for it.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-base')"

COPY --chown=user app.py .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
