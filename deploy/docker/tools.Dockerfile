# Dependency service: DIAMOND, Foldseek, COILS, pyhmmer (+ the data commands)
FROM mambaorg/micromamba:1.5.10
COPY --chown=$MAMBA_USER:$MAMBA_USER deploy/envs/tools.yml /tmp/tools.yml
RUN micromamba install -y -n base -f /tmp/tools.yml && micromamba clean -a -y
WORKDIR /opt/seeddrift
COPY --chown=$MAMBA_USER:$MAMBA_USER seeddrift ./seeddrift
COPY --chown=$MAMBA_USER:$MAMBA_USER services ./services
COPY --chown=$MAMBA_USER:$MAMBA_USER data ./data
ENV PYTHONPATH=/opt/seeddrift SEEDDRIFT_DATA=/data
EXPOSE 8601
CMD ["uvicorn", "services.tools.app:app", "--host", "0.0.0.0", "--port", "8601"]
