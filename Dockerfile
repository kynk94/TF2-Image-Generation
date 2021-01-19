FROM tensorflow/tensorflow:nightly-gpu

ARG user_name=noname
ARG user_uid=1000
ARG user_gid=$user_uid

COPY requirements.txt .
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
            /etc/apt/sources.list.d/cuda.list \
            /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        cmake \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        yasm \
        pkg-config \
        libgl1-mesa-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        tzdata \
        && \
# ==================================================================
# python packages
# ------------------------------------------------------------------
    $PIP_INSTALL \
        pip \
        && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        scipy \
        scikit-image \
        # scikit-learn \
        matplotlib \
        tqdm \
        opencv-python \
        pillow \
        # imutils \
        # dlib \
        && \
    $PIP_INSTALL -r requirements.txt \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* requirements.txt

ARG TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN groupadd --gid $user_gid $user_name \
    && useradd --uid $user_uid --gid $user_gid -m $user_name \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $user_name ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$user_name \
    && chmod 0440 /etc/sudoers.d/$user_name
USER $user_name

EXPOSE 6006
