"""
Generate a reproducible pair of real NISAR GCOV HDF5 files for scientific validation and testing.

Follows the official NISAR GCOV product specification:
- HDF5 group: science/LSAR/GCOV/grids/frequencyA/
- Datasets: HHHH, HVHV (calibrated linear power gamma0)
- Metadata: geotransform (west, px_x, 0, north, 0, -px_y), CRS (EPSG:4326)
- Simulates an actual flood change scenario (Sacramento River Valley, CA):
  * Pre-event (dry/normal): agricultural fields (~ -10 dB, 0.1 linear)
  * Post-event (flooded): specular water reflection (~ -22 dB, 0.006 linear)
  * Permanent river: low backscatter in both (~ -22 dB)
  * Urban areas: stable high backscatter (~ +3 dB, 2.0 linear)
  * Swath borders: nodata (0.0) to test proper boundary masking
"""

from pathlib import Path
import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "sample_data"
DATA_DIR.mkdir(exist_ok=True)


def create_nisar_gcov_pair():
    pre_path = DATA_DIR / "nisar_gcov_20240115_pre.h5"
    post_path = DATA_DIR / "nisar_gcov_20240215_post.h5"

    height, width = 256, 256

    # Georeference: Sacramento Valley bounding box (-121.8 to -121.4 W, 38.3 to 38.7 N)
    west = -121.8
    north = 38.7
    pixel_size_x = 0.4 / width
    pixel_size_y = 0.4 / height
    geotransform = [west, pixel_size_x, 0.0, north, 0.0, -pixel_size_y]
    crs_wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]"

    rng = np.random.default_rng(2026)

    # Base agricultural terrain (~ -11 dB = 0.079 in linear power)
    base_terrain = rng.gamma(shape=4.0, scale=0.02, size=(height, width)).astype(np.float32)

    # Permanent river corridor (running north-south with a meander)
    yy, xx = np.mgrid[0:height, 0:width]
    river_center = 128 + 20 * np.sin(yy / 25.0)
    river_mask = np.abs(xx - river_center) < 5

    # Urban cluster (east side, high backscatter ~ +2 dB = 1.58 linear)
    urban_mask = ((yy - 60)**2 + (xx - 190)**2) < 25**2

    # Flood inundation basin (central agricultural fields, inundated in post-event)
    flood_basin_mask = ((yy - 150)**2 + (xx - 120)**2) < 45**2

    # Swath border nodata (outer 8 pixels)
    border_mask = (yy < 8) | (yy >= height - 8) | (xx < 8) | (xx >= width - 8)

    # --- Pre-event raster ---
    pre_array = base_terrain.copy()
    pre_array[river_mask] = 0.006  # permanent water (~ -22 dB)
    pre_array[urban_mask] = 1.6    # urban (~ +2 dB)
    # in pre-event, flood basin is normal agriculture
    pre_array[border_mask] = 0.0   # nodata

    # --- Post-event raster ---
    post_array = base_terrain.copy()
    post_array[river_mask] = 0.006 # permanent water remains water
    post_array[urban_mask] = 1.6   # urban remains stable
    post_array[flood_basin_mask] = 0.005 # FLOODED: backscatter drops to ~ -23 dB (specular reflection)
    post_array[border_mask] = 0.0  # nodata

    # HV polarization (cross-pol, generally ~6 dB lower)
    pre_hv = pre_array * 0.25
    post_hv = post_array * 0.25

    def write_h5(filepath, hh, hv, date_str):
        with h5py.File(filepath, "w") as f:
            group = f.create_group("science/LSAR/GCOV/grids/frequencyA")
            group.attrs["geotransform"] = geotransform
            group.attrs["crs"] = crs_wkt
            group.attrs["acquisition_date"] = date_str
            group.attrs["platform"] = "NISAR"
            group.attrs["processing_level"] = "GCOV"

            ds_hh = group.create_dataset("HHHH", data=hh, dtype=np.float32)
            ds_hh.attrs["units"] = "gamma0 linear power"
            ds_hh.attrs["geotransform"] = geotransform
            ds_hh.attrs["crs"] = crs_wkt
            ds_hh.attrs["nodata"] = 0.0

            ds_hv = group.create_dataset("HVHV", data=hv, dtype=np.float32)
            ds_hv.attrs["units"] = "gamma0 linear power"
            ds_hv.attrs["geotransform"] = geotransform
            ds_hv.attrs["crs"] = crs_wkt
            ds_hv.attrs["nodata"] = 0.0

    write_h5(pre_path, pre_array, pre_hv, "2024-01-15T14:00:00Z")
    write_h5(post_path, post_array, post_hv, "2024-02-15T14:00:00Z")

    print(f"Generated NISAR GCOV test pair:\n  Pre: {pre_path}\n  Post: {post_path}")
    return pre_path, post_path


if __name__ == "__main__":
    create_nisar_gcov_pair()
