"""
Comprehensive test suite for SAR change-detection math and HDF5 operations.

Validates:
1. Log-ratio dB calculation on known arrays: 10 * log10(post / pre).
2. Absolute change thresholding (|diff_db| > threshold).
3. NoData and NaN/Inf isolation: ensuring nodata pixels are not counted in metrics.
4. Shape compatibility checks.
5. Synthetic pair generation properties.
6. HDF5 dataset discovery, slicing, geotransform bounds extraction, and error handling.
"""

from pathlib import Path
import h5py
import numpy as np
import pytest

from sar_processor import (
    CHANGE_THRESHOLD_DB,
    compute_change_detection,
    generate_synthetic_sar_pair,
    inspect_h5_metadata,
    load_h5_slice,
)


class TestChangeDetectionMath:
    def test_known_array_doubling_power(self):
        """Doubling linear power must equal +3.0103 dB."""
        pre = np.full((10, 10), 1.0, dtype=np.float32)
        post = np.full((10, 10), 2.0, dtype=np.float32)

        res = compute_change_detection(pre, post, threshold_db=3.0)
        expected_db = 10.0 * np.log10(2.0)  # ~3.0103
        assert np.isclose(res["mean_db"], expected_db, atol=1e-3)
        assert res["valid_pixel_count"] == 100
        # Since 3.0103 > 3.0, 100% of pixels should be classified as changed
        assert res["changed_area_pct"] == 100.0

    def test_known_array_halving_power(self):
        """Halving linear power (e.g. water specular drop) must equal -3.0103 dB."""
        pre = np.full((10, 10), 1.0, dtype=np.float32)
        post = np.full((10, 10), 0.5, dtype=np.float32)

        res = compute_change_detection(pre, post, threshold_db=3.0)
        expected_db = 10.0 * np.log10(0.5)  # ~ -3.0103
        assert np.isclose(res["mean_db"], expected_db, atol=1e-3)
        assert res["valid_pixel_count"] == 100
        assert res["changed_area_pct"] == 100.0

    def test_identical_rasters_zero_change(self):
        """Identical rasters must yield 0.0 dB change and 0% changed area."""
        arr = np.full((20, 20), 0.25, dtype=np.float32)
        res = compute_change_detection(arr, arr, threshold_db=3.0)
        assert res["mean_db"] == 0.0
        assert res["changed_area_pct"] == 0.0
        assert res["valid_pixel_count"] == 400

    def test_custom_threshold(self):
        """Custom threshold of 1.0 dB should flag smaller backscatter shifts."""
        pre = np.full((10, 10), 1.0, dtype=np.float32)
        post = np.full((10, 10), 1.3, dtype=np.float32)  # ~ +1.139 dB

        # With 3 dB threshold: 0% changed
        res_3 = compute_change_detection(pre, post, threshold_db=3.0)
        assert res_3["changed_area_pct"] == 0.0

        # With 1 dB threshold: 100% changed
        res_1 = compute_change_detection(pre, post, threshold_db=1.0)
        assert res_1["changed_area_pct"] == 100.0


class TestNoDataAndMasking:
    def test_zeros_excluded_as_nodata(self):
        """Boundary zero pixels must be completely excluded from metrics."""
        pre = np.full((10, 10), 1.0, dtype=np.float32)
        post = np.full((10, 10), 2.0, dtype=np.float32)

        # Set outer row and col to 0 (nodata boundary)
        pre[0, :] = 0.0
        post[0, :] = 0.0

        res = compute_change_detection(pre, post, nodata_value=0.0)
        # 90 valid pixels, 10 nodata
        assert res["valid_pixel_count"] == 90
        assert res["total_pixel_count"] == 100

    def test_nan_and_inf_excluded(self):
        """NaN and Inf values must not corrupt calculation or count as valid."""
        pre = np.full((10, 10), 1.0, dtype=np.float32)
        post = np.full((10, 10), 2.0, dtype=np.float32)

        pre[1, 1] = np.nan
        post[2, 2] = np.inf

        res = compute_change_detection(pre, post)
        assert res["valid_pixel_count"] == 98
        assert np.isfinite(res["mean_db"])

    def test_all_nodata_graceful_handling(self):
        """When all pixels are nodata, return 0.0 metrics without crashing."""
        pre = np.zeros((10, 10), dtype=np.float32)
        post = np.zeros((10, 10), dtype=np.float32)

        res = compute_change_detection(pre, post, nodata_value=0.0)
        assert res["valid_pixel_count"] == 0
        assert res["mean_db"] == 0.0
        assert res["changed_area_pct"] == 0.0


class TestShapeAndOrientation:
    def test_mismatched_shapes_raise_error(self):
        pre = np.ones((10, 10), dtype=np.float32)
        post = np.ones((12, 10), dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_change_detection(pre, post)

    def test_image_base64_output(self):
        pre, post = generate_synthetic_sar_pair(seed=42)
        res = compute_change_detection(pre, post, mode="synthetic")
        assert res["image_base64"].startswith("data:image/png;base64,")
        assert len(res["image_base64"]) > 1000
        assert res["mode"] == "synthetic"


class TestHDF5Operations:
    def test_inspect_and_load_real_sample(self):
        sample_path = (
            Path(__file__).resolve().parent.parent
            / "sample_data"
            / "nisar_gcov_20240115_pre.h5"
        )
        if not sample_path.exists():
            pytest.skip("Sample HDF5 not yet generated")

        info = inspect_h5_metadata(str(sample_path))
        assert "datasets" in info
        ds_names = [d["path"] for d in info["datasets"]]
        assert any("science/LSAR/GCOV/grids/frequencyA/HHHH" in n for n in ds_names)

        arr, meta = load_h5_slice(str(sample_path))
        assert arr.ndim == 2
        assert meta["bounds"] is not None
        assert meta["bounds"]["west"] < meta["bounds"]["east"]
        assert meta["bounds"]["south"] < meta["bounds"]["north"]
        assert "crs" in meta

    def test_missing_dataset_raises_informative_error(self, tmp_path):
        dummy_h5 = tmp_path / "dummy.h5"
        with h5py.File(dummy_h5, "w") as f:
            f.create_dataset("test/data", data=np.zeros((10, 10)))

        with pytest.raises(FileNotFoundError, match="Available datasets"):
            load_h5_slice(str(dummy_h5), "nonexistent/dataset/path")
