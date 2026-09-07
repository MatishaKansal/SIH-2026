from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR  # noqa: E402
from xlsr import EMBEDDING_DIM, XLSREmbeddingExtractor  # noqa: E402


@unittest.skipUnless((MODEL_DIR / "pytorch_model.bin").is_file(), "local XLS-R weights are unavailable")
class TestFrozenXLSR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extractor = XLSREmbeddingExtractor(MODEL_DIR, device="cpu")

    def test_cpu_processor_model_and_single_embedding_contract(self) -> None:
        audio = np.sin(np.linspace(0, 100, 16_000, dtype=np.float32))
        first = self.extractor.extract(audio, 16_000)
        second = self.extractor.extract(audio, 16_000)
        self.assertEqual(str(self.extractor.device), "cpu")
        self.assertEqual(first.shape, (EMBEDDING_DIM,))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        self.assertFalse(self.extractor.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.extractor.model.parameters()))

    def test_short_padded_and_batch_inference(self) -> None:
        short = np.linspace(-0.05, 0.05, 8_000, dtype=np.float32)
        padded = np.pad(short, (0, 8_000))
        batch = self.extractor.extract_batch([short, padded], 16_000, valid_lengths=[8_000, 8_000])
        self.assertEqual(batch.shape, (2, EMBEDDING_DIM))
        self.assertTrue(np.isfinite(batch).all())


@unittest.skipUnless(torch.cuda.is_available() and (MODEL_DIR / "pytorch_model.bin").is_file(), "CUDA or local XLS-R weights unavailable")
class TestCudaXLSR(unittest.TestCase):
    def test_cuda_inference(self) -> None:
        extractor = XLSREmbeddingExtractor(MODEL_DIR, device="cuda")
        self.assertEqual(extractor.extract(np.zeros(16_000, dtype=np.float32)).shape, (EMBEDDING_DIM,))
