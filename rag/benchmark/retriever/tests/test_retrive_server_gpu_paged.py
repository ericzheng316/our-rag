import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "retrive_server_gpu_paged.py"
)
SPEC = importlib.util.spec_from_file_location("retrive_server_gpu_paged", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Vectors:
    def __init__(self, rows, columns):
        self.shape = (rows, columns)
        self.size = rows * columns
        self.reshape_calls = []

    def reshape(self, *shape):
        self.reshape_calls.append(shape)
        return self


class _IndexFlat:
    def __init__(self, vectors, metric_type):
        self.vectors = vectors
        self.ntotal, self.d = self.vectors.shape
        self.metric_type = metric_type

    def get_xb(self):
        return self.vectors


class _GpuIndex:
    def __init__(self, resources, dimension, config):
        self.resources = resources
        self.d = dimension
        self.config = config
        self.added = None

    def add(self, vectors):
        self.added = vectors


class _Config:
    device = None
    useFloat16 = None


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0
    METRIC_L2 = 1
    IndexFlat = _IndexFlat
    GpuIndexFlatConfig = _Config
    GpuIndexFlatIP = _GpuIndex
    GpuIndexFlatL2 = _GpuIndex

    @staticmethod
    def StandardGpuResources():
        return object()

    @staticmethod
    def rev_swig_ptr(pointer, size):
        assert pointer.size == size
        return pointer.reshape(-1)


class PagedGpuLoaderTests(unittest.TestCase):
    def test_inner_product_index_defaults_to_fp32_and_uses_view(self):
        vectors = _Vectors(3, 8)
        cpu_index = _IndexFlat(vectors, _FakeFaiss.METRIC_INNER_PRODUCT)

        gpu_index, resources = MODULE.cpu_index_to_gpu_paged(
            cpu_index, _FakeFaiss, device=2
        )

        self.assertIs(gpu_index.resources, resources)
        self.assertEqual(gpu_index.config.device, 2)
        self.assertFalse(gpu_index.config.useFloat16)
        self.assertIs(gpu_index.added, vectors)
        self.assertEqual(vectors.reshape_calls, [(-1,), (3, 8)])

    def test_l2_index_can_explicitly_enable_fp16(self):
        cpu_index = _IndexFlat(_Vectors(2, 4), _FakeFaiss.METRIC_L2)

        gpu_index, _ = MODULE.cpu_index_to_gpu_paged(
            cpu_index, _FakeFaiss, use_float16=True
        )

        self.assertIsInstance(gpu_index, _GpuIndex)
        self.assertTrue(gpu_index.config.useFloat16)

    def test_rejects_non_flat_index(self):
        with self.assertRaisesRegex(TypeError, "IndexFlat"):
            MODULE.cpu_index_to_gpu_paged(object(), _FakeFaiss)

    def test_rejects_unsupported_metric(self):
        cpu_index = _IndexFlat(_Vectors(2, 4), 99)
        with self.assertRaisesRegex(ValueError, "metric_type=99"):
            MODULE.cpu_index_to_gpu_paged(cpu_index, _FakeFaiss)


if __name__ == "__main__":
    unittest.main()
