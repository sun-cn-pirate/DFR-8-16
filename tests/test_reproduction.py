from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DFR-source"))

from MVTec import NormalDataset, TestDataset  # noqa: E402
from anoseg_dfr import AnoSegDFR  # noqa: E402
from main import MVTEC_CATEGORIES, load_summary, select_categories, write_summary  # noqa: E402


class DatasetCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "bottle"
        (self.root / "train" / "good").mkdir(parents=True)
        (self.root / "test" / "good").mkdir(parents=True)
        (self.root / "test" / "broken").mkdir(parents=True)
        (self.root / "ground_truth" / "broken").mkdir(parents=True)

        grayscale = np.full((32, 24), 127, dtype=np.uint8)
        Image.fromarray(grayscale).save(self.root / "train" / "good" / "000.png")
        Image.fromarray(grayscale).save(self.root / "test" / "good" / "000.png")
        Image.fromarray(grayscale).save(self.root / "test" / "broken" / "001.png")
        mask = np.zeros((32, 24), dtype=np.uint8)
        mask[4:12, 5:14] = 255
        Image.fromarray(mask).save(
            self.root / "ground_truth" / "broken" / "001_mask.png"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_grayscale_training_image_is_converted_to_rgb(self) -> None:
        dataset = NormalDataset(str(self.root / "train" / "good"))
        self.assertEqual(tuple(dataset[0].shape), (3, 256, 256))

    def test_good_and_anomalous_masks_are_loaded(self) -> None:
        dataset = TestDataset(str(self.root / "test"))
        samples = {Path(dataset[index][2]).parent.name: dataset[index] for index in range(len(dataset))}
        self.assertEqual(float(np.asarray(samples["good"][1]).sum()), 0.0)
        self.assertGreater(float(np.asarray(samples["broken"][1]).sum()), 0.0)

    def test_visualization_artifacts_use_modern_image_dtypes(self) -> None:
        dfr = object.__new__(AnoSegDFR)
        dfr.cfg = argparse.Namespace(save_path=str(Path(self.temp_dir.name) / "outputs"))
        dfr.subpath = "bottle/test-model"
        scores = np.linspace(0, 1, 256 * 256, dtype=np.float32).reshape(256, 256)
        binary = (scores > 0.5).astype(np.uint8)
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[64:96, 64:96] = 1
        image_path = str(self.root / "test" / "broken" / "001.png")
        dfr.save_seg_results(scores, binary, mask, image_path)
        result_root = Path(dfr.cfg.save_path) / "Results" / dfr.subpath
        self.assertTrue(any((result_root / "score_map").iterdir()))
        self.assertTrue(any((result_root / "gt_pred_score_map").iterdir()))


class CliTests(unittest.TestCase):
    def test_all_selects_every_mvtec_category(self) -> None:
        self.assertEqual(select_categories(["all"]), MVTEC_CATEGORIES)

    def test_unknown_category_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_categories(["not_a_category"])

    def test_summary_files_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = argparse.Namespace(seed=0, epochs=1)
            row = {
                "category": "bottle",
                "det_pr": 0.9,
                "det_auc": 0.91,
                "seg_pr": 0.8,
                "seg_auc": 0.81,
                "seg_pro": 0.7,
                "seg_iou": 0.6,
            }
            write_summary([row], Path(temp_dir), args)
            self.assertTrue((Path(temp_dir) / "dfr_mvtec_summary.csv").is_file())
            self.assertTrue((Path(temp_dir) / "dfr_mvtec_summary.md").is_file())
            self.assertTrue((Path(temp_dir) / "environment.json").is_file())
            loaded = load_summary(Path(temp_dir))
            self.assertEqual(loaded[0]["category"], "bottle")
            self.assertAlmostEqual(float(loaded[0]["seg_auc"]), 0.81)


class CheckpointTests(unittest.TestCase):
    def test_training_checkpoint_restores_epoch_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = torch.nn.Linear(2, 1)
            dfr = object.__new__(AnoSegDFR)
            dfr.model_path = temp_dir
            dfr.autoencoder = model
            dfr.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            dfr.n_dim = 1
            dfr.data_name = "synthetic"
            dfr.device = torch.device("cpu")
            dfr.cfg = argparse.Namespace(
                seed=123,
                epochs=9,
                data_root=Path(temp_dir),
                cnn_layers=("relu1_1", "relu1_2"),
            )
            expected = model.weight.detach().clone()
            dfr.save_model(epoch=7)
            checkpoint = torch.load(
                Path(temp_dir) / "autoencoder.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["checkpoint_version"], 2)
            self.assertEqual(checkpoint["seed"], 123)
            self.assertEqual(checkpoint["config"]["data_root"], temp_dir)
            self.assertEqual(
                checkpoint["config"]["cnn_layers"],
                ["relu1_1", "relu1_2"],
            )
            with torch.no_grad():
                model.weight.zero_()
            epoch = dfr.load_training_checkpoint()
            self.assertEqual(epoch, 7)
            self.assertTrue(torch.equal(model.weight.detach(), expected))


if __name__ == "__main__":
    unittest.main()
