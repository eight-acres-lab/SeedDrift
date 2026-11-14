# SeedDrift API (own algorithm) + web page
FROM python:3.11-slim
WORKDIR /opt/seeddrift
RUN pip install --no-cache-dir "numpy>=1.26" biopython==1.88 "fastapi>=0.110" "uvicorn>=0.29" python-multipart "pyrodigal>=3.5,<4"
COPY seeddrift ./seeddrift
COPY services ./services
COPY web ./web
COPY data ./data
ENV PYTHONPATH=/opt/seeddrift SEEDDRIFT_DATA=/data
EXPOSE 8600
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8600"]
