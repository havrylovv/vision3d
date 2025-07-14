# Vision 3D installation  

- Vision3D has been developed and tested with Python 3.10, PyTorch 2.4.0, CUDA 11.8, and NVIDIA V100. The code is expected to work with newer versions as well.

1. Create conda environment
```bash 
conda create --name vis3d python=3.10
conda activate vis3d
```

2. Install torch
```bash 
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Install dependencies
```bash 
pip install -r requirements.txt
```

4. Build custom CUDA kernels for Deformable Attention
```bash 
cd vision3d/models/ops
bash make.sh
```
5. Install `vision3d` package
 ```bash 
pip install -e .
```
