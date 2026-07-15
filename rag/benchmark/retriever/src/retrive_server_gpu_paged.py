"""Run the existing retriever server with a paged single-GPU FAISS loader.

FAISS' ``index_cpu_to_all_gpus`` takes a direct-clone shortcut when only one
GPU is visible.  For a large ``IndexFlat`` stored as FP16 on GPU, that shortcut
converts the complete CPU matrix in one temporary allocation.  The HotpotQA
index needs about 30 GiB for that allocation in addition to its final GPU
storage, which exceeds an 80 GiB H100 during conversion.

This wrapper leaves the existing server and vendored FlashRAG implementation
untouched.  It creates an empty GPU flat index and calls its public ``add``
method instead.  FP32 is the default so retrieval retains the original index
precision; optional FP16 storage uses FAISS' built-in 256 MiB paging path.
"""

from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


def cpu_index_to_gpu_paged(
    cpu_index: Any,
    faiss_module: Any,
    *,
    device: int = 0,
    use_float16: bool = False,
) -> tuple[Any, Any]:
    """Move a CPU ``IndexFlat`` to one GPU through FAISS' paged add path."""

    if not isinstance(cpu_index, faiss_module.IndexFlat):
        raise TypeError(
            "Paged single-GPU loading currently supports faiss.IndexFlat only; "
            f"got {type(cpu_index).__name__}"
        )

    resources = faiss_module.StandardGpuResources()
    config = faiss_module.GpuIndexFlatConfig()
    config.device = device
    config.useFloat16 = use_float16

    if cpu_index.metric_type == faiss_module.METRIC_INNER_PRODUCT:
        gpu_index = faiss_module.GpuIndexFlatIP(
            resources, cpu_index.d, config
        )
    elif cpu_index.metric_type == faiss_module.METRIC_L2:
        gpu_index = faiss_module.GpuIndexFlatL2(
            resources, cpu_index.d, config
        )
    else:
        raise ValueError(
            "Paged single-GPU loading supports inner-product and L2 metrics; "
            f"got metric_type={cpu_index.metric_type}"
        )

    if cpu_index.ntotal:
        # rev_swig_ptr exposes the existing CPU index memory.  Reshaping this
        # contiguous float32 view does not duplicate the ~64 GiB matrix.
        vectors = faiss_module.rev_swig_ptr(
            cpu_index.get_xb(), int(cpu_index.ntotal) * int(cpu_index.d)
        ).reshape(int(cpu_index.ntotal), int(cpu_index.d))
        # GpuIndexFlat.add reserves final storage and, in FP16 mode, delegates
        # the transfer/conversion to GpuIndex's bounded paging implementation.
        gpu_index.add(vectors)

    return gpu_index, resources


def install_paged_single_gpu_loader() -> None:
    """Patch DenseRetriever only inside this server process."""

    import faiss
    from flashrag.retriever import DenseRetriever

    def load_index(self: Any) -> None:
        if self.index_path is None or not Path(self.index_path).exists():
            raise FileNotFoundError(f"Index file {self.index_path} does not exist")

        cpu_index = faiss.read_index(self.index_path)
        if not self.use_faiss_gpu:
            self.index = cpu_index
            return

        gpu_count = faiss.get_num_gpus()
        if gpu_count != 1:
            raise RuntimeError(
                "Paged retriever expects exactly one visible GPU; "
                f"FAISS reports {gpu_count}. Set CUDA_VISIBLE_DEVICES to the "
                "dedicated retriever GPU."
            )

        use_float16 = os.environ.get("FAISS_GPU_USE_FLOAT16", "0") == "1"
        log.info(
            "Loading FAISS IndexFlat through bounded %s transfer "
            "(vectors=%s, dim=%s, visible_gpu=0)",
            "FP16" if use_float16 else "FP32",
            cpu_index.ntotal,
            cpu_index.d,
        )
        gpu_index, resources = cpu_index_to_gpu_paged(
            cpu_index, faiss, device=0, use_float16=use_float16
        )
        # Keep the provider alive explicitly for the lifetime of the index.
        self._faiss_gpu_resources = resources
        self.index = gpu_index
        log.info("Paged FAISS GPU transfer complete (vectors=%s)", self.index.ntotal)

    DenseRetriever.load_index = load_index


def main() -> None:
    install_paged_single_gpu_loader()
    server_path = Path(__file__).with_name("retrive_server.py")
    runpy.run_path(str(server_path), run_name="__main__")


if __name__ == "__main__":
    main()
