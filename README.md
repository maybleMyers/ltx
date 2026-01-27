# to run the gui:
```bash
pip install uv  
uv sync  
uv run python lt1.py
```

Use with the official ltxv2 models and full gemma text encoder from the main ltx page. This repository is under active development and alot of features are quite broken but the basics should work well. If you need some helps to get it going I will try...  
  
# These are working settings to get a 609 frame video with a 24gb gpu
  
![LTX-2 Video Generator](packages/screenshot.jpg)  
  
   
      
         
This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides

## Model Links

### LTX-2 Core Models (Lightricks)
Download from [Lightricks/LTX-2 on HuggingFace](https://huggingface.co/Lightricks/LTX-2):
| File | Description |
|------|-------------|
| `ltx-2-19b-dev.safetensors` | Main 19B dev checkpoint |
| `ltx-2-19b-distilled.safetensors` | Distilled model |
| `ltx-2-19b-distilled-lora-384.safetensors` | Distilled LoRA |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | 2x spatial upscaler |

### Text Encoder (Gemma)
| Model | Link |
|-------|------|
| gemma-3-12b-it-qat-q4_0-unquantized | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

### Video Frame Interpolation

#### GIMM-VFI
Download from [GSean/GIMM-VFI on HuggingFace](https://huggingface.co/GSean/GIMM-VFI):
| File | Description |
|------|-------------|
| `gimmvfi_r_arb.pt` | GIMM-VFI-R (RAFT-based) |
| `gimmvfi_r_arb_lpips.pt` | GIMM-VFI-R-P (RAFT + Perceptual) |
| `gimmvfi_f_arb.pt` | GIMM-VFI-F (FlowFormer-based) |
| `gimmvfi_f_arb_lpips.pt` | GIMM-VFI-F-P (FlowFormer + Perceptual) |
| `flowformer_sintel.pth` | FlowFormer optical flow (also on [Google Drive](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_)) |
| `raft-things.pth` | RAFT optical flow (also from [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)) |

#### BiM-VFI
| File | Link |
|------|------|
| `bim_vfi.pth` | [Google Drive](https://drive.google.com/file/d/18Wre7XyRtu_wtFRzcsit6oNfHiFRt9vC/view?usp=sharing) |

Place VFI checkpoints in `GIMM-VFI/pretrained_ckpt/`

### Upscalers
| Model | Link |
|-------|------|
| RealESRGAN_x2plus.pth | [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| RealESRGAN_x4plus.pth | [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| 003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth | [SwinIR GitHub Release](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth) |
| basicvsr_plusplus_reds4.pth | [OpenMMLab](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) |

Place upscaler checkpoints in `GIMM-VFI/pretrained_ckpt/`

### Depth Estimation
| Model | Link |
|-------|------|
| ZoeDepth (Intel/zoedepth-nyu-kitti) | [HuggingFace](https://huggingface.co/Intel/zoedepth-nyu-kitti) (auto-downloaded by transformers) |

## Troubleshooting

### CUDA / GPU Issues

**CUDA out of memory**
- Enable **CPU Offloading** in Model Settings
- Enable **Block Swap** for DiT and Text Encoder to reduce VRAM usage
- Reduce **DiT Blocks in GPU** (try 10-15 for 24GB VRAM)
- Reduce **Text Encoder Blocks in GPU** (try 4-6)
- Lower resolution or frame count
- Use FP8 quantized checkpoints (`ltx-2-19b-dev-fp8.safetensors`)

**CUDA version mismatch / not detected**
- Ensure CUDA >= 12.7 is installed
- Check PyTorch CUDA version matches system: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- Reinstall PyTorch with correct CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)

### Model Loading Errors

**FileNotFoundError: Checkpoint not found**
- Verify model paths in the GUI match actual file locations
- Check that models are downloaded completely (not corrupted/partial)
- Use absolute paths if relative paths fail

**Error loading Gemma text encoder**
- Ensure you have the full unquantized Gemma model, not GGUF format
- Accept the license on HuggingFace before downloading
- Check the `gemma-3-12b-it-qat-q4_0-unquantized` folder contains `config.json` and model files

**GIMM-VFI / BiM-VFI model errors**
- Ensure all checkpoints are in `GIMM-VFI/pretrained_ckpt/`
- For FlowFormer variants, verify `flowformer_sintel.pth` is present
- For RAFT variants, verify `raft-things.pth` is present

### Generation Issues

**Black or corrupted output video**
- Check input image/video dimensions are divisible by 32
- Frame count must be divisible by 8, plus 1 (e.g., 9, 17, 25, 33...)
- Try reducing inference steps or changing the seed

**Very slow generation**
- Enable block swap to trade speed for VRAM
- Disable CPU offloading if you have sufficient VRAM
- Use distilled model for faster inference (fewer steps needed)

**Prompt enhancement not working**
- Verify Gemma model path is correct
- Check "Enhance Prompt" is enabled
- Gemma requires significant VRAM; enable text encoder block swap

### Installation Issues

**uv sync fails**
- Update uv: `pip install -U uv`
- Clear cache: `uv cache clean`
- Try with fresh venv: `uv venv && uv sync`

**Import errors / missing modules**
- Run from the project root directory
- Ensure virtual environment is activated: `uv run python lt1.py`
- Check Python version >= 3.12

**Gradio UI not loading**
- Check for port conflicts (default 7860)
- Try specifying a different port in launch options
- Disable any VPN/proxy that might block localhost

### Video Frame Interpolation Issues

**Interpolation produces artifacts**
- Try a different model variant (RAFT vs FlowFormer)
- Use perceptual variants (-P) for better quality
- Reduce interpolation multiplier for fast motion scenes

**Upscaler produces blurry results**
- SwinIR-L generally produces sharper results than RealESRGAN
- BasicVSR++ is optimized for video temporal consistency
- Check input video quality - upscalers can't recover lost detail

### Common Fixes

```bash
# Clear PyTorch cache
rm -rf ~/.cache/torch

# Clear HuggingFace cache (redownloads models)
rm -rf ~/.cache/huggingface

# Check GPU memory usage
nvidia-smi

# Monitor GPU during generation
watch -n 1 nvidia-smi
```
