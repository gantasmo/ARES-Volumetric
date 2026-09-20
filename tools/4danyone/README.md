# 4DAnyone on Windows and Turing

`windows-turing.patch` is the change set that makes ant-research/4DAnyone
(`e38f210827f7b3effbe5b573ea07cfcf17e72dca`) run on this machine: Windows, two RTX 2080 Ti
(sm_75, no bfloat16 tensor cores, 11 GB each). Apply it to a clone in `tools/ext/4danyone`.

| file | change |
| --- | --- |
| `fdanyone/io.py` | `fcntl` is Linux only: the output lock uses `msvcrt.locking` on Windows, and `O_NOFOLLOW` is optional |
| `fdanyone/precision.py` (new) | the compute dtypes and the second DiT device, read from the environment |
| `fdanyone/model/loader.py` | DiT, pose encoder and VAE dtypes come from `precision`; with a split device the blocks are placed per card and the Turbo LoRA fuses afterwards, in place |
| `fdanyone/model/inference.py`, `model/vae.py` | autocast and the noise tensor follow those dtypes |
| `fdanyone/model/conditioning.py` | the pose encoder's batch limit is settable (`FDANYONE_POSE_BATCH`); its output is cast back to bfloat16 for the CPU feature bank; the pose feature shape follows the configured window instead of the 121-frame one |
| `fdanyone/model/turbo_lora.py` | each Turbo delta merges on the device of the weight it belongs to, so a DiT split over two cards fuses on the GPUs |
| `fdanyone/config.py` | window length, raster and tiled VAE read the environment |
| `fdanyone/motion/result.py`, `fdanyone/video.py` | the frame count follows the configured window |
| `fdanyone/vendor/diffsynth/models/wan_video_dit.py` | the three internal chunk budgets are settable (`FDANYONE_TEMP_BUDGET_MB`), and the feed-forward runs in token chunks under the same budget: its hidden tensor is 14336 wide and was the single biggest allocation |

## Environment

| variable | value here | what it does |
| --- | --- | --- |
| `FDANYONE_DIT_DTYPE` | `float16` | DiT and pose encoder weights and autocast; the release is bfloat16, which Turing has no tensor cores for |
| `FDANYONE_VAE_DTYPE` | `float16` | Wan2.2 VAE weights and autocast |
| `FDANYONE_SPLIT_DEVICE` | `cuda:1` | the second half of the 30 DiT blocks lives there, so the 5B DiT spans both 11 GB cards; activations follow the blocks through forward pre-hooks |
| `FDANYONE_POSE_BATCH` | `3` | full-resolution skeleton videos encoded at once (the release encodes 6, about 3.9 GB of input) |
| `FDANYONE_SPLIT_AT` | unset (15) | index of the first block on the second card, when the halves need to be uneven |
| `FDANYONE_FRAMES` | `45` | generated window; the release is 121 and needs more than 11 GB. `(frames - 1) % 4 == 0` |
| `FDANYONE_HEIGHT`, `FDANYONE_WIDTH` | `1280`, `704` | generated raster, multiples of 32 |
| `FDANYONE_TEMP_BUDGET_MB` | `256` | the DiT's internal chunk budgets, 1536 in the release |
| `FDANYONE_TILED_VAE` | `1` | tile the VAE encode and decode |

Run with `--gpu_ids "[0]"` so the pipeline keeps one worker (the split already uses both cards)
and `--attention_backend sdpa` (FlashAttention-3 and SageAttention need newer hardware).

## Install

```sh
git clone https://github.com/ant-research/4DAnyone.git tools/ext/4danyone
cd tools/ext/4danyone
git submodule update --init --depth 1
git apply ../../4danyone/windows-turing.patch
uv venv --python 3.12 venv
uv pip install --python venv/Scripts/python.exe -r requirements.txt
python scripts/download_model.py --model_dir models        # 21 GB
```

torch 2.8 and torchvision 0.23 are linked into `venv/Lib/site-packages` from an environment that
already has them (directory junctions for `torch`, `torchgen`, `functorch`, `torchvision`, with the
two `.dist-info` folders copied so uv sees them as installed).

`SMPLX_NEUTRAL.npz` is installed from `3DAIGC/LHMPP-Prior`
(`human_model_files/smplx/SMPLX_NEUTRAL.npz`) through `fdanyone.download.install_smplx`.
