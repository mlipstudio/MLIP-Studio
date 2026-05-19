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