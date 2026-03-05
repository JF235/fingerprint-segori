"""
Batch inference pipeline for pyfing's LEADER model.

Supports single-GPU, multi-GPU (DDP), and CPU inference with two execution
strategies:
  - full_gpu: model forward + minutiae extraction on GPU, async file saving
  - hybrid:   model forward on GPU, minutiae extraction + saving on CPU workers

Usage:
    from pyfing.pytorch.api import run_inference
    run_inference('images/', 'output/', gpus=2, batch_size=8)
"""

import glob
import logging
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Import LeaderNet directly from its module to avoid pulling in Keras/TensorFlow.
# algorithms.py imports from pyfing.minutiae which does `import keras` at the
# top level — that triggers TF even if we never use it.
from .leader_model import LeaderNet

logger = logging.getLogger("pyfing.leader")

torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"


def _load_state(model: torch.nn.Module, weights_path: str, device: str) -> torch.nn.Module:
    """Load weights into *model*, move to *device*, and set eval mode."""
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _default_weights_path() -> str:
    """Lazy lookup — registry import is deferred so TF is never loaded at CLI startup."""
    from .registry import get_model_spec
    return str(get_model_spec("leader").torch_weights)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintDataset(Dataset):
    """Loads greyscale fingerprint images for LEADER batch inference.

    Images are returned as float32 tensors in the [0, 255] range (no
    normalisation) because that is what LeaderNet was trained on.
    """

    def __init__(self, image_paths: list[str], dpi: int = 500, max_dim: int = 1024):
        self.image_paths = image_paths
        self.dpi = dpi
        self.max_dim = max_dim

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img_pil = Image.open(img_path).convert("L")

            # Pre-scale for DPI (LEADER model trained at 500 DPI)
            if self.dpi != 500:
                f = 500.0 / self.dpi
                new_w = max(1, int(round(img_pil.width * f)))
                new_h = max(1, int(round(img_pil.height * f)))
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Cap maximum dimension
            if img_pil.width > self.max_dim or img_pil.height > self.max_dim:
                logger.warning(
                    "Image %s size %s exceeds max_dim %d; resizing.",
                    os.path.basename(img_path), img_pil.size, self.max_dim,
                )
                img_pil.thumbnail((self.max_dim, self.max_dim), Image.Resampling.LANCZOS)

            # Keep [0, 255] float32 — LeaderNet expects this range
            img_np = np.array(img_pil, dtype=np.float32)

            return {
                "image": torch.from_numpy(img_np).unsqueeze(0),  # [1, H, W]
                "path": img_path,
                "original_shape": img_np.shape,  # (H, W)
            }
        except Exception as e:
            logger.warning("Could not load %s: %s", img_path, e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Path discovery
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_EXTS = {".png", ".bmp", ".wsq", ".tif", ".tiff", ".jpg", ".jpeg"}


def find_image_paths(input_path: str, recursive: bool = True) -> list[str]:
    """Collect image paths from a file, directory, or .txt/.list file."""
    paths: list[str] = []

    if os.path.isfile(input_path):
        _, ext = os.path.splitext(input_path)
        if ext.lower() in {".txt", ".list"}:
            with open(input_path) as f:
                paths = [line.strip() for line in f if line.strip()]
        elif ext.lower() in _SUPPORTED_EXTS:
            paths = [input_path]
        else:
            raise ValueError(
                f"Unsupported file type: {input_path!r}. "
                f"Supported: {sorted(_SUPPORTED_EXTS)}"
            )
    elif os.path.isdir(input_path):
        for ext in ("png", "bmp", "tif", "tiff", "jpg", "jpeg"):
            pattern = (
                f"{input_path}/**/*.{ext}" if recursive else f"{input_path}/*.{ext}"
            )
            paths.extend(glob.glob(pattern, recursive=recursive))
    else:
        raise ValueError(f"Input path does not exist: {input_path!r}")

    if not paths:
        raise ValueError(f"No images found in: {input_path!r}")

    return sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def _round32(n: int) -> int:
    """Round *n* up to the next multiple of 32."""
    return (n + 31) // 32 * 32


def dynamic_padding_collate(batch):
    """Pad images within a batch to the same (mult-of-32) size.

    The batch max H and W are each rounded up to a multiple of 32 so that
    LeaderNet's U-Net-style encoder can downsample evenly.  White (255) is
    used as the padding value because fingerprint images have white backgrounds.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None

    max_h = _round32(max(b["image"].shape[1] for b in batch))
    max_w = _round32(max(b["image"].shape[2] for b in batch))

    images, paths, orig_shapes = [], [], []
    for b in batch:
        img = b["image"]          # [1, H, W]
        _, h, w = img.shape
        pad_h, pad_w = max_h - h, max_w - w
        # F.pad order: last dim first → (left_W, right_W, top_H, bottom_H)
        padded = F.pad(img, (0, pad_w, 0, pad_h), value=255.0)
        images.append(padded)
        paths.append(b["path"])
        orig_shapes.append(b["original_shape"])

    batch_tensors = torch.stack(images)  # [B, 1, max_h, max_w]
    orig_hs = torch.tensor([s[0] for s in orig_shapes])
    orig_ws = torch.tensor([s[1] for s in orig_shapes])

    return batch_tensors, paths, (orig_hs, orig_ws)


# ─────────────────────────────────────────────────────────────────────────────
# Minutiae extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_minutiae_numpy(
    out: np.ndarray,         # [H, W, 4] float32
    threshold: float,
) -> np.ndarray:
    """Return [N, 4] array: [x, y, direction_rad, quality].

    Channel layout:
        out[..., 0]: position confidence
        out[..., 1]: direction in radians (CCW, pyfing convention)
        out[..., 2]: type confidence (>= type_threshold → Ending)
        out[..., 3]: NMS-filtered position score
    """
    ys, xs = np.where(out[..., 3] >= threshold)
    if xs.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    directions = out[ys, xs, 1]
    qualities  = out[ys, xs, 3]
    result = np.column_stack([xs, ys, directions, qualities]).astype(np.float32)
    return result[np.argsort(-qualities)]


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_minutiae(
    minutiae_np: np.ndarray,   # [N, 4]: x, y, direction_rad, quality
    input_path: str,
    output_path: str,
    input_base_path: str | None = None,
):
    """Write a .min file in the standard #MIN X Y ANGLE QUALITY format.

    Angle convention (pyfing LEADER): CCW radians → CCW degrees.
    Formula: ``round(deg(direction_rad) % 360)``  (no negation, unlike FingerNet).
    """
    if input_base_path and os.path.isdir(input_base_path):
        rel_path = os.path.relpath(input_path, input_base_path)
        rel_dir  = os.path.dirname(rel_path)
        filename = os.path.basename(rel_path)
    else:
        rel_dir  = ""
        filename = os.path.basename(input_path)

    base_name = os.path.splitext(filename)[0]
    out_path  = os.path.join(output_path, "minutiae", rel_dir, f"{base_name}.min")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if minutiae_np.size == 0:
        # Still write a valid (empty) .min file
        np.savetxt(
            out_path, np.empty((0, 4), dtype=int),
            fmt="%d", header="X Y ANGLE QUALITY", comments="#MIN ", delimiter=" ",
        )
        return

    xs          = minutiae_np[:, 0].astype(int)
    ys          = minutiae_np[:, 1].astype(int)
    # CCW degrees (pyfing uses CCW radians — direct conversion, no negation)
    angle_deg   = np.round(np.rad2deg(minutiae_np[:, 2]) % 360).astype(int)
    quality_int = np.round(minutiae_np[:, 3] * 100).astype(int)

    out_array = np.column_stack([xs, ys, angle_deg, quality_int])
    np.savetxt(
        out_path, out_array,
        fmt="%d", header="X Y ANGLE QUALITY", comments="#MIN ", delimiter=" ",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CPU worker functions
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess_and_save_batch(
    raw_outputs_cpu: torch.Tensor,   # [B, 4, H, W] NCHW
    batch_paths: list[str],
    batch_orig_shapes: tuple,        # (orig_hs, orig_ws) tensors
    output_path: str,
    threshold: float,
    input_base_path: str | None,
):
    """CPU worker: extract minutiae from raw model output and save (hybrid strategy)."""
    worker_id = threading.get_ident()
    logger.debug("CPU worker %d processing %d images", worker_id, len(batch_paths))
    try:
        # NCHW → NHWC
        out_nhwc = raw_outputs_cpu.permute(0, 2, 3, 1).numpy()  # [B, H, W, 4]
        orig_hs, orig_ws = batch_orig_shapes

        for i, img_path in enumerate(batch_paths):
            orig_h = orig_hs[i].item()
            orig_w = orig_ws[i].item()
            out_crop = out_nhwc[i, :orig_h, :orig_w, :]
            minutiae = _extract_minutiae_numpy(out_crop, threshold)
            save_minutiae(minutiae, img_path, output_path, input_base_path)

    except Exception as e:
        warnings.warn(
            f"Post-processing failed for batch starting with "
            f"{os.path.basename(batch_paths[0])}: {e}"
        )


def _save_results_chunk(
    chunk: list[dict],
    output_path: str,
    input_base_path: str | None,
):
    """CPU worker: save a pre-extracted chunk of results (full_gpu strategy)."""
    for item in chunk:
        try:
            save_minutiae(
                item["minutiae"], item["input_path"], output_path, input_base_path
            )
        except Exception as e:
            logger.warning(
                "Failed to save %s: %s", os.path.basename(item.get("input_path", "?")), e
            )


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ddp(rank: int, world_size: int, gpu_id: int, timeout_minutes: int = 30):
    """Setup DDP for a specific GPU.
    
    Args:
        rank: Process rank in DDP (0 to world_size-1)
        world_size: Total number of processes
        gpu_id: Physical CUDA device ID to use
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
        device_id=torch.device(f"cuda:{gpu_id}"),
    )


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _ddp_launch_target(rank: int, world_size: int, config: dict):
    """DDP worker process entry point.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Configuration dict containing 'gpu_ids' list
    """
    gpu_ids = config.get("gpu_ids", list(range(world_size)))
    gpu_id = gpu_ids[rank]  # Map rank to actual physical GPU ID
    runner = InferenceRunner(config)
    runner.setup(rank=rank, world_size=world_size, gpu_id=gpu_id)
    runner.run()


# ─────────────────────────────────────────────────────────────────────────────
# InferenceRunner
# ─────────────────────────────────────────────────────────────────────────────

class InferenceRunner:
    def __init__(self, config: dict):
        self.config = config
        self.rank = -1
        self.world_size = 1
        self.device = "cpu"
        self.is_main = True
        self.model = None
        self.dataloader = None

    def setup(self, rank: int = -1, world_size: int = 1, gpu_id: int = 0):
        self.rank       = rank
        self.world_size = world_size
        self.is_main    = rank <= 0  # rank 0 for DDP, -1 for single/cpu

        # 1. Device & DDP
        if world_size > 1:
            _setup_ddp(rank, world_size, gpu_id)
            self.device = f"cuda:{gpu_id}"
        elif self.config.get("gpus") and torch.cuda.is_available():
            # Single GPU: use the specified GPU ID directly
            gpu_id = self.config.get("gpu_ids", [0])[0]
            torch.cuda.set_device(gpu_id)
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cpu"

        if self.is_main:
            logger.info("Runner device: %s", self.device)
            os.makedirs(os.path.join(self.config["output_path"], "minutiae"), exist_ok=True)

        if world_size > 1:
            dist.barrier()

        # 2. Model
        weights = self.config.get("weights_path") or _default_weights_path()
        self.model = _load_state(LeaderNet(), weights, self.device)
        if self.config.get("compile_model"):
            if self.is_main:
                logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # 3. DataLoader
        dataset = FingerprintDataset(
            self.config["image_paths"],
            dpi=self.config.get("dpi", 500),
            max_dim=self.config.get("max_image_dim", 1024),
        )
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=sampler,
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            persistent_workers=(self.config.get("num_workers", 4) > 0),
            collate_fn=dynamic_padding_collate,
        )

    def run(self):
        strategy = self.config.get("strategy", "full_gpu")
        if self.is_main:
            logger.info("Strategy: %s", strategy)

        if strategy == "hybrid":
            self._run_hybrid()
        elif strategy == "full_gpu":
            self._run_full_gpu()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        if self.world_size > 1:
            dist.barrier()
            _cleanup_ddp()

        if self.is_main:
            logger.info("✓ LEADER inference complete.")

    # ── Hybrid: GPU infer → CPU post-process + save ──────────────────────────

    def _run_hybrid(self):
        num_cpu = self.config.get("num_cpu_workers", 4)
        threshold = self.config.get("threshold", 0.6)
        input_base = self.config.get("input_base_path")

        with ThreadPoolExecutor(max_workers=num_cpu) as executor:
            futures = []
            max_queue = 2 * num_cpu

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                for batch_tensors, batch_paths, batch_orig_shapes in tqdm(
                    self.dataloader, desc=desc, disable=not self.is_main
                ):
                    if batch_tensors is None:
                        continue

                    batch_tensors = batch_tensors.to(self.device)
                    raw = self.model(batch_tensors)            # [B, 4, H, W]
                    raw_cpu = raw.detach().cpu()

                    future = executor.submit(
                        _postprocess_and_save_batch,
                        raw_cpu, batch_paths, batch_orig_shapes,
                        self.config["output_path"], threshold, input_base,
                    )
                    futures.append(future)

                    if len(futures) >= max_queue:
                        futures.pop(0).result()

            if self.is_main:
                logger.info("Inference done; finalising CPU workers…")
            for f in tqdm(futures, desc="Finalising", disable=not self.is_main):
                f.result()

    # ── Full GPU: GPU infer + extract → CPU save ─────────────────────────────

    def _run_full_gpu(self):
        num_save = self.config.get("num_cpu_workers", 4)
        chunk_size = self.config["batch_size"] * 10
        threshold = self.config.get("threshold", 0.6)
        input_base = self.config.get("input_base_path")

        with ThreadPoolExecutor(max_workers=num_save) as save_executor:
            futures = []
            pending_chunk: list[dict] = []

            with torch.no_grad():
                desc = f"GPU {self.rank}" if self.world_size > 1 else "Processing"
                for batch_tensors, batch_paths, batch_orig_shapes in tqdm(
                    self.dataloader, desc=desc, disable=not self.is_main
                ):
                    if batch_tensors is None:
                        continue

                    batch_tensors = batch_tensors.to(self.device)
                    raw = self.model(batch_tensors)            # [B, 4, H, W]
                    # NCHW → NHWC on GPU, then to numpy
                    out_nhwc = raw.permute(0, 2, 3, 1).cpu().numpy()

                    orig_hs, orig_ws = batch_orig_shapes
                    for i, img_path in enumerate(batch_paths):
                        orig_h = orig_hs[i].item()
                        orig_w = orig_ws[i].item()
                        out_crop = out_nhwc[i, :orig_h, :orig_w, :]
                        minutiae = _extract_minutiae_numpy(out_crop, threshold)
                        pending_chunk.append({"input_path": img_path, "minutiae": minutiae})

                    if len(pending_chunk) >= chunk_size:
                        futures.append(
                            save_executor.submit(
                                _save_results_chunk, pending_chunk,
                                self.config["output_path"], input_base,
                            )
                        )
                        pending_chunk = []

            if pending_chunk:
                futures.append(
                    save_executor.submit(
                        _save_results_chunk, pending_chunk,
                        self.config["output_path"], input_base,
                    )
                )

            if self.is_main:
                logger.info("Inference done; waiting for save workers…")
            for f in tqdm(futures, desc="Saving", disable=not self.is_main):
                f.result()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    input_path: str,
    output_path: str,
    weights_path: str | None = None,
    gpus: int | list[int] | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    dpi: int = 500,
    threshold: float = 0.6,
    type_threshold: float = 0.5,
    recursive: bool = True,
    strategy: str = "full_gpu",
    num_cpu_workers: int = 4,
    compile_model: bool = False,
    max_image_dim: int = 1024,
):
    """Run LEADER batch inference.

    Args:
        input_path:     Image file, directory, or .txt/.list file of paths.
        output_path:    Base output directory; ``minutiae/`` subdir is created.
        weights_path:   Path to LEADER .pth weights (None = bundled default).
        gpus:           GPU config — 0/None: CPU; 1: one GPU; N: N-GPU DDP;
                        [i,j]: specific physical GPUs (after CUDA_VISIBLE_DEVICES remap).
        batch_size:     Images per GPU per step.
        num_workers:    DataLoader worker processes per GPU.
        dpi:            DPI of input images (model trained at 500).
        threshold:      Minutia quality threshold (default 0.6).
        type_threshold: Ending/Bifurcation threshold (kept for API consistency).
        recursive:      Recurse into subdirectories.
        strategy:       ``'full_gpu'`` or ``'hybrid'``.
        num_cpu_workers: Thread workers for saving (full_gpu) or post-proc (hybrid).
        compile_model:  Use ``torch.compile`` (experimental).
        max_image_dim:  Resize images larger than this (post-DPI-scaling).
    """
    image_paths = find_image_paths(input_path, recursive)

    if os.path.isdir(input_path):
        input_base_path = input_path
    elif os.path.isfile(input_path):
        input_base_path = os.path.dirname(os.path.abspath(input_path))
    else:
        input_base_path = None

    config = dict(locals())   # capture all parameters
    config["image_paths"]    = image_paths
    config["input_base_path"] = input_base_path

    use_cpu = not gpus or not torch.cuda.is_available()
    world_size = 0
    gpu_ids = []
    
    if not use_cpu:
        if isinstance(gpus, int):
            world_size = gpus
            # Use first N GPUs: [0, 1, ..., N-1]
            gpu_ids = list(range(gpus))
        elif isinstance(gpus, list):
            world_size = len(gpus)
            # Use specific GPU IDs as provided
            gpu_ids = gpus
        else:
            raise TypeError(f"gpus must be int or list[int], got {type(gpus)}")
    
        # Validate GPU IDs
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(
                    f"Requested GPU {gpu_id} but only {available_gpus} GPUs available "
                    f"(IDs: 0-{available_gpus-1})"
                )
    
    config["gpu_ids"] = gpu_ids

    is_ddp = not use_cpu and world_size > 1

    if use_cpu:
        logger.info("Running LEADER on CPU")
        config["gpus"] = False
        runner = InferenceRunner(config)
        runner.setup()
        runner.run()
    elif is_ddp:
        logger.info(
            "Running LEADER on %d GPU(s) (DDP) with GPU IDs: %s",
            world_size, gpu_ids,
        )
        mp.spawn(_ddp_launch_target, nprocs=world_size, args=(world_size, config), join=True)
    else:
        logger.info(
            "Running LEADER on single GPU: %d",
            gpu_ids[0],
        )
        config["gpus"] = True
        runner = InferenceRunner(config)
        runner.setup(rank=-1, world_size=1, gpu_id=gpu_ids[0])
        runner.run()
