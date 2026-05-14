# OctoLong ms-swift training image.
#
# Self-contained CUDA + ms-swift image for multi-node SFT with the
# iNeil77/ms-swift fork (branch v4.2.0-multiturn-sft) which adds the
# `octolong` template + agent_template that produce byte-equivalent output to
# Qwen3's HF chat_template.
#
# Pinned stack (all compatible with cu130 / torch 2.11):
#   CUDA 13.1.2, Python 3.12, torch 2.11.0, vllm 0.20.2, lmdeploy 0.13.0,
#   flash-attn 2.8.3, deepspeed 0.19.0 (CPU ops prebuilt), liger_kernel,
#   transformers 4.56-5.8, trl <1.0, peft <0.20, datasets >=3.0,<4.8.5,
#   xformers 0.0.34, flash-linear-attention + tilelang + causal-conv1d +
#   mamba-ssm for hybrid linear-attention models.
#
# Build (BuildKit-enabled, with apt + pip + ccache mounts that DON'T bloat
# the final image). The Dockerfile references no local files in the build
# context, so any non-empty directory works as the context.
#
#   cd /path/to/ms-swift
#   docker buildx build --progress=plain \
#       -t ineil77/ms-swift:v4.2.0-multiturn-sft-extras \
#       --load \
#       -f docker/Dockerfile docker/
#
# Smoke test:
#   docker run --gpus all --rm -it ineil77/ms-swift:v4.2.0-multiturn-sft-extras \
#       python -c "import swift; from swift.template.register import TEMPLATE_MAPPING; \
#                  from swift.agent_template import agent_template_map; \
#                  print('octolong template registered:', 'octolong' in TEMPLATE_MAPPING); \
#                  print('octolong agent_template registered:', 'octolong' in agent_template_map)"
#
# Notes on size and rebuild speed:
#   * The base nvidia/cuda:13.1.2-cudnn-devel image is required (~3.95 GB
#     content) because Triton (used by liger-kernel, flash-linear-attention,
#     tilelang) JIT-compiles to PTX at runtime and needs ptxas from the
#     CUDA toolkit. A multi-stage runtime-only final layer would break this.
#   * Each compile-heavy step (deepspeed, flash-attn, causal-conv1d,
#     mamba-ssm) ends with an explicit cleanup that removes /tmp build
#     trees, ccache by-products, and any pip-cached wheels. ccache itself
#     uses a BuildKit cache mount so it persists across rebuilds without
#     bloating the published image.
#   * BuildKit cache mounts (--mount=type=cache) keep apt and pip download
#     caches between builds for the same daemon, but those caches do NOT
#     end up in any image layer.

# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# Global env. PIP_NO_CACHE_DIR=1 prevents pip from leaving a per-user wheel
# cache in /root/.cache/pip — that cache would otherwise be persisted in
# every layer that runs pip.
# ---------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# ---------------------------------------------------------------------------
# Build-time CUDA + arch + ccache configuration. ccache is wired at the apt
# layer below; setting CC/CXX up front lets every subsequent compile pick it
# up automatically.
# ---------------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/lib/ccache:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 10.3+PTX" \
    MAX_JOBS=160 \
    NVCC_THREADS=8 \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=20G
# ccache is wired via the /usr/lib/ccache/{cc,c++,gcc,g++} shim symlinks (first
# entry on PATH). nvcc's -ccbin needs a single-binary path, so we deliberately
# leave CC/CXX unset; the shims are picked up automatically when torch's
# cpp_extension shells out to the host compiler.

# ---------------------------------------------------------------------------
# 1. System packages + Python 3.12 (deadsnakes ppa). Uses BuildKit apt
# cache mounts so apt-get's archive doesn't end up in the final layer.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git git-lfs gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ninja-build cmake pkg-config ccache \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libffi-dev liblzma-dev libxml2-dev libxslt1-dev \
        libsndfile1 libsox-dev ffmpeg \
        libopenmpi-dev openmpi-bin \
        libaio-dev libzstd-dev zstd \
        iproute2 iputils-ping net-tools rsync openssh-client \
        vim less htop tini \
        python3.12 python3.12-dev python3.12-venv && \
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.12 - && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 100 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 && \
    git lfs install && \
    rm -rf /var/log/* /usr/share/doc /usr/share/man /var/tmp/*

# ---------------------------------------------------------------------------
# 2. PyTorch + audio/vision (cu130 wheels).
# Uses a BuildKit pip cache mount so subsequent builds reuse the wheel
# downloads. The cache lives outside the image.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel packaging && \
    python -m pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0

# ---------------------------------------------------------------------------
# 3. Core ML dependency stack (versions aligned with modelscope swift4.1.x).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        "transformers>=4.56,<5.9" \
        "datasets>=3.0,<4.8.5" \
        "accelerate>=0.34" \
        "peft>=0.11,<0.20" \
        "trl>=0.15,<1.0" \
        "safetensors>=0.4" \
        "huggingface-hub>=0.24" \
        "modelscope>=1.23" \
        awscli pynvml py-cpuinfo deepspeed-kernels \
        tiktoken sentencepiece einops rouge tqdm matplotlib \
        "PyYAML>=5.4" json_repair addict attrdict binpacking dacite \
        fastapi "gradio<6.0" "uvicorn[standard]" openai oss2 \
        tensorboard wandb swanlab \
        scipy scikit-learn pandas pillow soundfile librosa \
        qwen_vl_utils qwen_omni_utils \
        icecream py-spy nvitop pre-commit \
        math_verify \
        liger-kernel \
        "ray[default]>=2.10"

# ---------------------------------------------------------------------------
# 4. DeepSpeed 0.19.0 with CPU ops prebuilt (CCL/sparse/stochastic disabled).
# Source build inside /tmp; cleanup at end strips ~250 MB of intermediate
# objects.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    DS_BUILD_OPS=1 \
    DS_BUILD_CCL_COMM=0 \
    DS_BUILD_SPARSE_ATTN=0 \
    DS_BUILD_STOCHASTIC_TRANSFORMER=0 \
    DS_BUILD_EVOFORMER_ATTN=0 \
    pip install --no-build-isolation \
        deepspeed==0.19.0 \
        --config-settings "--global-option=build_ext" \
        --config-settings "--global-option=-j${MAX_JOBS}" && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/torch_extensions

# ---------------------------------------------------------------------------
# 5. xformers (cu130 wheel; no-deps prevents pip from re-pinning torch).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        xformers==0.0.34

# ---------------------------------------------------------------------------
# 6. flash-attn 2.8.3 (sdist-only on PyPI; compiles ~10 min with ccache hit
# on rebuild). Cleanup removes the unpacked source tree pip leaves under
# /tmp/pip-* on success.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    python -m pip install "flash-attn==2.8.3" --no-build-isolation && \
    rm -rf /tmp/* /var/tmp/*

# ---------------------------------------------------------------------------
# 7. Hybrid linear-attention kernels.
#
# Required by:
#   * Qwen3.5 / Qwen3.5-MoE  -- transformers' modeling_qwen3_5 imports
#                              fla.ops.gated_delta_rule.chunk_gated_delta_rule
#                              and fla.modules.FusedRMSNormGated. ms-swift's
#                              swift/model/models/qwen.py raises ImportError
#                              when --sequence_parallel_size > 1 if missing.
#   * Mamba family            -- swift/model/models/mamba.py error string
#                              "Remember installing causal-conv1d>=1.2.0
#                               and mamba-ssm".
#
# tilelang accelerates a subset of fla's ops on Hopper/Blackwell when
# present; fla detects it at runtime. abi3 wheel ships pre-built so install
# is instant. We pin to the [conv1d] extra of fla so its dispatch picks up
# our installed causal-conv1d version.
#
# Inter-dependencies:
#   - causal-conv1d         -> torch (already installed, cu130).
#   - mamba-ssm             -> torch + triton>=3.5.0 + causal-conv1d>=1.2.0
#                              via the [causal-conv1d] extra.
#   - flash-linear-attention[conv1d] -> causal-conv1d>=1.4.0.
#   - tilelang              -> torch (abi3 wheel, no compile).
#
# Install order:
#   1. causal-conv1d FIRST (downstream packages need it during build).
#   2. tilelang (cheap wheel; install before fla so fla auto-detects it).
#   3. flash-linear-attention[conv1d] and mamba-ssm in either order.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.ccache \
    python -m pip install --no-build-isolation \
        "causal-conv1d>=1.4.0" && \
    python -m pip install \
        tilelang && \
    python -m pip install --no-build-isolation \
        "flash-linear-attention[conv1d]" \
        mamba-ssm && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/torch_extensions

# ---------------------------------------------------------------------------
# 8. vllm + lmdeploy (no-deps to keep our torch/transformers pins).
#    vllm uses flashinfer-python and flashinfer-cubin at runtime.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation \
        "vllm==0.20.2" \
        "lmdeploy==0.13.0" \
        "flashinfer-python==0.6.8.post1" \
        "flashinfer-cubin==0.6.8.post1" && \
    python -m pip install \
        msgspec prometheus-client outlines fairscale partial-json-parser \
        cloudpickle pyzmq

# ---------------------------------------------------------------------------
# 9. Multimodal preprocessor + media-codec stack.
#
# Required by ms-swift template paths in swift/template/templates/*.py
# and swift/template/vision_utils.py. None compile CUDA; all PyPI-only:
#
#   * qwen-vl-utils, qwen-omni-utils  -- already installed in step 3.
#   * keye-vl-utils                    -- Kuai Keye-VL preprocessing.
#   * minicpmo                         -- MiniCPM-O multimodal helper.
#   * openai-whisper                   -- Llama3.1-Omni audio frontend
#                                          (import name: 'whisper').
#   * qwen-asr                         -- Qwen3-ASR audio frontend.
#   * decord, av (PyAV)                -- video readers used by
#                                          template/vision_utils.py.
#   * moviepy                          -- legacy video reader fallback.
#   * audioread                        -- librosa fallback decoder.
#
# NOT installed (vendor-only on GitHub or in checkpoint dirs; PyPI names
# squatted): llava, mplug-owl2, valley-eagle, deepseek-vl/vl2, janus,
# emu3, omni-speech. Layer those in a per-model derived Dockerfile.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --ignore-installed blinker && \
    python -m pip install \
        keye-vl-utils \
        minicpmo \
        openai-whisper \
        qwen-asr \
        decord \
        av \
        moviepy \
        audioread

# ---------------------------------------------------------------------------
# 10. ms-swift fork at the multiturn-sft branch.
# Installed in editable mode so users can patch swift/template/* at runtime
# (e.g. by mounting a different branch of the fork over /opt/ms-swift).
# ---------------------------------------------------------------------------
ARG SWIFT_REPO=https://github.com/iNeil77/ms-swift.git
ARG SWIFT_BRANCH=v4.2.0-multiturn-sft
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 --branch ${SWIFT_BRANCH} ${SWIFT_REPO} /opt/ms-swift && \
    cd /opt/ms-swift && \
    python -m pip install -e . && \
    python -c "from swift.template.register import TEMPLATE_MAPPING; \
               from swift.agent_template import agent_template_map; \
               assert 'octolong' in TEMPLATE_MAPPING, 'octolong template not registered'; \
               assert 'octolong' in agent_template_map, 'octolong agent_template not registered'; \
               print('octolong template + agent_template registered OK')"

# ---------------------------------------------------------------------------
# 11. Final cleanup pass.
# Strip *.pyc bytecode-cache files (they're rebuilt on first run anyway),
# discard pip's per-user wheel cache (PIP_NO_CACHE_DIR should already
# prevent it but belt-and-suspenders), wipe /tmp and /var/tmp, and remove
# any stray torch JIT extension caches that source builds may have left.
# ---------------------------------------------------------------------------
RUN find /usr/local/lib/python3.12 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true ; \
    find /usr/local/lib/python3.12 -name "*.pyc" -delete 2>/dev/null || true ; \
    rm -rf \
        /root/.cache/pip \
        /root/.cache/torch_extensions \
        /tmp/* /var/tmp/* \
        /usr/share/doc /usr/share/man /usr/share/locale \
        /var/log/*

# ---------------------------------------------------------------------------
# 12. Sensible runtime defaults.
# ---------------------------------------------------------------------------
ENV TOKENIZERS_PARALLELISM=false \
    NCCL_NVLS_ENABLE=0 \
    NCCL_P2P_LEVEL=NVL \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    PYTHONPATH=/opt/ms-swift

WORKDIR /opt/ms-swift

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
