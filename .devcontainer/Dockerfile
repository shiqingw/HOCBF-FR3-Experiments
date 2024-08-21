# ARG UBUNTU_VER=22.04
# FROM ubuntu:${UBUNTU_VER}

FROM ros:humble

# Use bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# System packages 
RUN apt-get update -y \
    && apt-get upgrade -y

RUN apt -y purge binutils \
    && apt -y remove make \
    && apt -y autoremove 
    
RUN apt-get install -y vim \
    net-tools \
    build-essential \
    git \
    liblapack-dev \
    libopenblas-dev \
    libeigen3-dev \
    libboost-all-dev \
    libglib2.0-dev \
    python3-pip \
    libpoco-dev \
    python3-tk

# Install minimal LaTeX packages for English
RUN apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-base \
    cm-super \
    dvipng \
    && texhash

# Display
RUN apt-get install -y -q --no-install-recommends \
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

# Install the required Python packages
RUN pip3 install numpy \
    scipy==1.11.4 \
    matplotlib \
    proxsuite \
    pin \
    mujoco \
    osqp \
    ipykernel \
    empy==3.3.4 \
    catkin_pkg \
    lark \
    PyQt6 \
    "pybind11[global]"

# Install xtl
RUN git clone https://github.com/xtensor-stack/xtl.git \
    && cd xtl \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtl

# # Install xsimd
RUN git clone https://github.com/xtensor-stack/xsimd.git \
    && cd xsimd \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xsimd

# Install xtensor
RUN git clone https://github.com/xtensor-stack/xtensor.git \
    && cd xtensor \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor

# Install xtensor-blas
RUN git clone https://github.com/shiqingw/xtensor-blas.git \
    && cd xtensor-blas \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor-blas

# Install xtensor-python
RUN git clone https://github.com/shiqingw/xtensor-python.git \
    && cd xtensor-python \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor-python

# Install scs
RUN git clone https://github.com/cvxgrp/scs.git \
    && cd scs \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf scs

# Install liegroups
RUN git clone https://github.com/utiasSTARS/liegroups.git \
    && cd liegroups \
    && pip install -e . 

# Install libfranka
RUN git clone --recursive https://github.com/frankaemika/libfranka --branch 0.13.3 \
    && cd libfranka \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF .. \
    && cmake --build . \
    && cpack -G DEB \
    && dpkg -i libfranka*.deb \
    && cd / \
    && rm -rf libfranka

# Install LCM (checkout at an older version)
RUN git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm \
    && git checkout 6f2f4c948667452aea021c0afefbd512f91283bc \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd /lcm/lcm-python \
    && pip install .\
    && cd / \
    && rm -rf lcm

# Install FR3Py
RUN git clone --recursive https://github.com/Rooholla-KhorramBakht/FR3Py.git \
    && cd FR3Py \
    && pip install -e .

# Install FR3Py-Gripper
RUN cd FR3Py/fr3_gripper \
    && pip install .
    
# Install C++ Bridge
RUN cd FR3Py/fr3_bridge \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j $(( $(nproc) - 1 )) \
    && make \
    && make install 

# Install vicon_bridge2
RUN mkdir vicon_ros2 \
    && cd vicon_ros2 \
    && mkdir src \
    && cd src \
    && git clone https://github.com/Rooholla-KhorramBakht/vicon_bridge2.git \
    && cd .. \
    && source /opt/ros/humble/setup.sh \
    && colcon build
    
# Install Scaling-Functions-Helper
RUN git clone https://github.com/shiqingw/Scaling-Functions-Helper.git\
    && cd Scaling-Functions-Helper \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install \
    && cd .. \
    && pip install -e .
    
# Install HOCBF-Helper
RUN git clone https://github.com/shiqingw/HOCBF-Helper.git\
    && cd HOCBF-Helper \
    && pip install -e .

# Install FR3-Vicon-Visualizer
RUN git clone https://github.com/shiqingw/FR3-Vicon-Visualizer.git \
    && cd FR3-Vicon-Visualizer \
    && pip install -e .

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/pip

# Spin the container
CMD ["tail", "-f", "/dev/null"]