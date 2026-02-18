# ⚠️ Installation Roadblock: Mamba-SSM Compatibility Issue

## 1. Problem Summary

The attempt to install `mamba-ssm` via a direct wheel URL failed with a **404 Not Found** error. This indicates that the specific pre-compiled binary (wheel) requested does not exist on the official repository's server.

```bash
.venv ❯ uv pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
× Failed to download ... HTTP status client error (404 Not Found)

```

## 2. Root Cause Analysis

### A. Major Version Mismatch (The "Bleeding Edge" Problem)

Your environment is running **PyTorch 2.10.0+cu128**.

* **PyTorch 2.10.0** is an extremely new (or nightly/dev) version.
* The official Mamba developers usually provide pre-compiled wheels for stable versions (e.g., PyTorch 2.1, 2.2, or 2.3).
* Because your version is too new, a matching pre-compiled wheel has not been built yet.

### B. CUDA Capability Conflict

You are using **CUDA 12.8/13.1**, but the wheel you attempted to download was named `+cu118` (CUDA 11.8).

* Mamba relies on highly optimized CUDA kernels (C++ code) that are strictly tied to specific versions of the CUDA Toolkit and PyTorch ABI.
* A wheel built for CUDA 11.8 **cannot** run on a CUDA 12.8/13.1 environment due to binary incompatibility.

### C. Build-from-Source Failure

When `uv` or `pip` cannot find a pre-compiled wheel, it attempts to "Build from Source." On **Arch Linux**, this often fails because:

1. **Compiler Mismatch:** Arch uses GCC 14+, but CUDA 12.8/13.1 typically requires GCC 12 or 13.
2. **Environment Isolation:** The build process cannot find the CUDA paths (`/opt/cuda`) inside the isolated temporary build environment.

---

## 3. Why we cannot proceed with the current setup

We have reached a state where:

1. **No Pre-built Binaries exist** for PyTorch 2.10 + CUDA 12.8+.
2. **Manual Compilation fails** due to the strict version requirements of the `causal-conv1d` and `mamba-ssm` custom kernels against the Arch Linux system libraries.
