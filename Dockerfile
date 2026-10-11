FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    swig \
    libopenbabel-dev \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --no-deps fairchem-core==2.16.0 --ignore-requires-python
RUN pip3 install -r requirements.txt 
RUN pip3 install --no-deps upet@git+https://github.com/lab-cosmo/upet.git --ignore-requires-python
RUN pip3 install --no-deps git+https://github.com/WillBaldwin0/graph_electrostatics.git --ignore-requires-python
# Optional AIMNet2 models. aimnet 0.0.1 is the last release that runs on Python 3.10; its model files
# are fetched at build time because aimnet stores them inside site-packages, which the app user cannot write.
RUN pip3 install --no-deps aimnet==0.0.1 && \
    python -c "from aimnet.calculators.model_registry import get_model_path; [get_model_path(m) for m in ('aimnet2', 'aimnet2_b973c')]"

EXPOSE 8501
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app/src


COPY --chown=user . $HOME/app/src
COPY --chown=user . $HOME/app
COPY --chown=user . $HOME

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health


ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none"]