FROM continuumio/miniconda3

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# Run package updates, install packages, and then clean up to reduce layer size
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git wget libatomic1 gfortran perl m4 cmake pkg-config \
    libopenblas-dev libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update Python in the base environment to 3.11
RUN conda install python==3.11 \
    && conda clean -afy

RUN pip install numpy==1.24.4 \
    scipy==1.10.1 \
    matplotlib==3.7.2 \
    proxsuite \
    pin==2.6.18 \
    pybind11 \
    && rm -rf ~/.cache/pip

RUN conda install -c conda-forge \
    xtensor \
    xtensor-blas \
    xtensor-python \
    && conda clean -afy

# Clone the repository
RUN git clone https://github.com/shiqingw/Differentiable-Optimization-Helper.git\
    && cd Differentiable-Optimization-Helper\
    && pip install .

# Spin the container
CMD ["tail", "-f", "/dev/null"]