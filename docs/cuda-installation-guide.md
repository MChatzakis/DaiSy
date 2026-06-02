# CUDA Installation Guide

This guide provides detailed instructions for installing CUDA to enable GPU acceleration with the SING algorithm.

## Linux (Ubuntu)

### 1. Install Required Packages

```bash
sudo apt update
sudo apt install curl
```

### 2. Install NVIDIA Drivers

```bash
sudo ubuntu-drivers autoinstall
```

**Note**: For specific driver versions, use `sudo apt install nvidia-driver-XXX` (e.g., `nvidia-driver-535`) after checking available drivers with `ubuntu-drivers devices`.

Reboot after driver installation:

```bash
sudo reboot
```

### 3. Add NVIDIA GPG Key

```bash
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg
```

### 4. Add CUDA Repository

```bash
echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-repository.list
```

### 5. Install CUDA Toolkit

```bash
sudo apt update
sudo apt install cuda
```

### 6. Set Up Environment Variables

```bash
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

### 7. Verification

```bash
nvcc --version
nvidia-smi
```

## Windows

### 1. Check System Compatibility

- **NVIDIA GPU**: Ensure you have an NVIDIA GPU (check Device Manager > Display adapters)
- **Windows Version**: Verify compatibility with CUDA Toolkit version

### 2. Update NVIDIA Drivers

1. Download latest drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers)
2. Select your GPU model and operating system
3. Run installer with "Express Installation"

### 3. Download and Install CUDA Toolkit

1. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select Windows, x86_64, and your Windows version 
3. Download the `exe (local)` installer (recommended)
4. Run installer as administrator
5. Choose "Express" installation for typical setups

### 4. Verify Installation

Open Command Prompt and run:

```cmd
nvcc --version
nvidia-smi
```

Check environment variables in System Properties > Environment Variables:

- `CUDA_PATH` should point to CUDA installation
- `PATH` should include `%CUDA_PATH%\bin`

## macOS

NVIDIA discontinued CUDA support for macOS. Consider using:

- **Metal Performance Shaders** for GPU acceleration on Apple Silicon
- **OpenCL** for cross-platform GPU computing
- **Docker** with Linux containers for CUDA development

## Troubleshooting

### Common Issues

**Driver conflicts**:

```bash
# Remove existing drivers (Linux)
sudo apt purge nvidia-*
sudo apt autoremove
# Reinstall following steps above
```

**Path issues**:

```bash
# Verify CUDA installation location
ls /usr/local/cuda/bin/nvcc  # Linux
# or check Windows environment variables
```

**Version mismatches**:

- Ensure driver version supports CUDA toolkit version
- Check compatibility matrix on NVIDIA documentation

### Getting Help

- Check [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- Review [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/)
- Search [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
