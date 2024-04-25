ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
# ARG OS_TYPE=aarch64

FROM ubuntu:${UBUNTU_VER}

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# System packages 
RUN apt-get update \
    && apt-get install -yq curl wget jq vim software-properties-common lsb-release net-tools\
    # update cmake
    && apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update \
    && apt-get install -y cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE

# Install miniconda to /miniconda
RUN wget http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

# Run package updates, install packages, and then clean up to reduce layer size
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git wget libatomic1 gfortran perl m4 pkg-config \
    liblapack-dev libopenblas-dev libgl1-mesa-glx libpoco-dev libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update Python in the base environment to 3.11
RUN conda install python==3.11 \
    && conda clean -afy

# Install the required Python packages
RUN pip install numpy==1.24.4 \
    scipy==1.10.1 \
    matplotlib==3.7.2 \
    proxsuite \
    pin==2.6.18 \
    mujoco \
    && rm -rf ~/.cache/pip

# Install pybind11, xtensor, xtensor-blas
RUN conda install -c conda-forge \
    pybind11 \
    xtensor \
    xtensor-blas \
    # xtensor-python \ 
    && conda clean -afy

# Install xtensor-python from source
RUN git clone https://github.com/shiqingw/xtensor-python.git \
    && cd xtensor-python \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make install \
    && cd ../.. \
    && rm -rf xtensor-python

# Install Differentiable-Optimization-Helper
RUN git clone https://github.com/shiqingw/Differentiable-Optimization-Helper.git\
    && cd Differentiable-Optimization-Helper \
    && pip install -e .

# Install FR3Py
RUN git clone https://github.com/Rooholla-KhorramBakht/FR3Py.git \
    && cd FR3Py \
    && pip install -e .\
    && cd ..

# Install libfranka
RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.10.0 \
    && cd libfranka \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. \
    && cmake --build . \
    && cpack -G DEB \
    && dpkg -i libfranka*.deb \
    && cd ../.. \
    && rm -rf libfranka

# Install LCM
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd ../lcm-python\
    && python -m pip install .\
    && cd ../.. \
    && rm -rf lcm

# Install C++ Bridge
RUN cd FR3Py/fr3_bridge \
&& mkdir build && cd build \
&& cmake .. \
&& make -j $(( $(nproc) - 1 )) \
&& make \
&& make install 

# # Setting up the real-time kernel
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#     build-essential bc curl ca-certificates gnupg2 libssl-dev lsb-release libelf-dev bison flex dwarves zstd libncurses-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN curl -SLO https://www.kernel.org/pub/linux/kernel/v5.x/linux-5.9.1.tar.xz \
#     && curl -SLO https://www.kernel.org/pub/linux/kernel/v5.x/linux-5.9.1.tar.sign \
#     && curl -SLO https://www.kernel.org/pub/linux/kernel/projects/rt/5.9/patch-5.9.1-rt20.patch.xz \
#     && curl -SLO https://www.kernel.org/pub/linux/kernel/projects/rt/5.9/patch-5.9.1-rt20.patch.sign \
#     && xz -d *.xz \
#     && tar xf linux-*.tar \
#     && cd linux-*/ \
#     && patch -p1 < ../patch-*.patch \
#     && make olddefconfig \
#     && make menuconfig \
#     && make -j$(nproc) deb-pkg \
#     && sudo dpkg -i ../linux-headers-*.deb ../linux-image-*.deb 

# Display
RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


# Spin the container
CMD ["tail", "-f", "/dev/null"]