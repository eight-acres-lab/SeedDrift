# Dependency service: ESMFold (needs an NVIDIA GPU and the NVIDIA container toolkit)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN pip install --no-cache-dir transformers==4.49.0 "fastapi>=0.110" "uvicorn>=0.29"
WORKDIR /opt/seeddrift
COPY seeddrift ./seeddrift
COPY services ./services
ENV PYTHONPATH=/opt/seeddrift SEEDDRIFT_DATA=/data
EXPOSE 8602
CMD ["uvicorn", "services.esmfold.app:app", "--host", "0.0.0.0", "--port", "8602"]
