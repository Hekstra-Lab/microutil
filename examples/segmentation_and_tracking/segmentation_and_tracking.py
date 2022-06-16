#!/bin/python

import sys
from time import perf_counter

import xarray as xr
from cellpose.models import Cellpose

import microutil as mu

#########
# SETUP #
#########

t0 = perf_counter

# Path to zarr containing the dataset and the position in the dataset to process
_, dataset_path, array_id = sys.argv


position_slice = slice(int(array_id), int(array_id) + 1)

ds = xr.open_zarr(dataset_path).sel(S=position_slice).squeeze()

cp_batch_size = 256  # 64 for 4MP images 256 for 1MP on 16G V100 GPU
bt_search_radius = 40

segmentation_imgs = ds["segmentation_imgs"].load()

assert (
    segmentation_imgs.ndim == 3
), f"Need 3D (TYX) array but found array with shape {segmentation_imgs.shape}"

t1 = perf_counter()
print(f"Setup complete -- {t1-t0:0.2f} seconds", flush=True)

############
# CELLPOSE #
############

print("Starting Cellpose segmentation", flush=True)

t0 = perf_counter()

model = Cellpose(model_type="cyto2", gpu=True)
channels = [[0, 0]]

image_list = [im.data for im in segmentation_imgs]

masks, flows, styles = model.cp.eval(
    image_list,
    batch_size=cp_batch_size,
    channels=channels,
    diameter=15,
    flow_threshold=0.6,
    cellprob_threshold=-1,
    normalize=False,
)

print("Saving Cellpose outputs", flush=True)

cp_ds = mu.cellpose.make_cellpose_dataset(masks, flows, styles)
mu.save_dataset(cp_ds, dataset_path, position_slice)
t1 = perf_counter()
print(f"Cellpose segmentation complete -- {t1-t0:0.2f} seconds", flush=True)

##########
# BTRACK #
##########

print("Starting btrack", flush=True)
t0 = perf_counter()

config_file = "cell_config.json"
tracks_out = dataset_path + f"pos_{array_id}_tracks.h5"
updated_masks = mu.btrack.gogogo_btrack(
    cp_ds['cp_masks'].data, config_file, bt_search_radius, tracks_out
)

bt_ds = xr.Dataset({"labels": xr.DataArray(updated_masks, dims=list("TYX"))})
mu.save_dataset(bt_ds, dataset_path, position_slice)

t1 = perf_counter()
print(f"Cell tracking complete -- {t1-t0:0.2f} seconds", flush=True)
print("Done", flush=True)
