FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="CLIPer CAID linker predictor"
LABEL org.opencontainers.image.description="Stage4 CNN concat_window linker prediction from precomputed ProstT5 embeddings"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

COPY submission/caid/requirements-caid.txt /app/requirements-caid.txt
RUN pip install --no-cache-dir -r /app/requirements-caid.txt

COPY cliper /app/cliper
COPY disorder/feature_modeling.py /app/disorder/feature_modeling.py
COPY disorder/feature_io.py /app/disorder/feature_io.py
COPY disorder/__init__.py /app/disorder/__init__.py
COPY configs/caid_stage4_predict.yaml /app/configs/caid_stage4_predict.yaml
COPY submission/caid/predict.sh /app/predict.sh

RUN chmod +x /app/predict.sh

ENV PYTHONPATH=/app

# Mount at runtime:
#   /input/targets.fasta
#   /input/embeddings  (one file per protein: {id}.npy or {id}.h5)
#   /models/best.pt
#   /output

ENTRYPOINT ["/app/predict.sh"]
