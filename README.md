# microutil

## About

`microutil` was developed in the [Hekstra Lab](https://hekstralab.fas.harvard.edu/) to enable large scale processing
of timelapse timelapse microscopy datasets. Many of the functions here wrap [`scikit-image`](https://scikit-image.org/)
functions to map them over each image in the dataset in parallel.

## Conventions

We like to

`import microutil as mu`.

Microutil typically operates on [xarray](https://xarray.pydata.org/en/stable/) datasets and data arrays. The standard use case is for 6D data with
the following naming convention:
- `S` : Position number
- `T` : Time
- `C` : Channel
- `Z` : Z stack slices
- `Y` : First image dimension
- `X` : Second image dimension

For the later parts of the pipeline, `microutil` will expect an xarray dataset with the same dimension names
as above and the following conventions for naming the variables in your dataset:

- `images` : Data array containing your raw images
- `masks` : Binary mask indicating which pixels are cells (1) and which are background (0)
- `labels` : Integer mask where each cell has a unique integer value and the background is still 0.


## Workflow

### Loading

Loading microsocpy images into python was one of the first tasks we tackled with microutil. Especially
if you use micromanager to acquire data, then the following will get you set up to start with `microutil`:

```
data = mu.load_mm_frames("/path/to/micromanager/data/")
```

This will lazily load your images and assemble them into a data array with the right dimensions and
coordinates as determined by the metadata.

### Semantic Segmentation

Semantic segmentation is the process of determining which pixels are cells and which pixels are background.
For semantic segmentation you can

- Use thresholding based on Otsu's method. We have sped up and parallelized this method in `mu.segmentation.calc_thresholds`.
- Use a neural network. If you happen to be working on yeast, you use the Unet from from
[YeaZ](https://www.nature.com/articles/s41467-020-19557-4) by downloading
[the appropriate weights](https://github.com/lpbsscientist/YeaZ-GUI#installation-steps) and calling

`predictions = mu.segmentation.apply_unet(images, "path/to/unet/weights.h5")`

### Instance Segmentation

At this point you will need to start using a dataset rather than a data array. This change can be as
simple as `ds = xr.Dataset({"images":your_data_array, "mask":your_semantic_seg_results})`.

Once you have determined the pixels that correspond to cells vs. background, you can break them up into
individual cells with `mu.individualize(ds)`. It is probably worth reading the docstring on this one
since there are a few variations accessible from this one function, though all fundamentally use the
watershed algorithm. This will create a variable in `ds` called `labels` that contains the labelled regions.

### Tracking

Once the cells are invidivudally labelled, we can track individuals through time with the Hungarian algorithm.
Our implementation uses the fairly high performance [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to solve the matching
problem. We compute the cost matrix by comparing the centers of mass of cells at successive time points as well as their
areas and they extent to which they overlap with one another.

*Note*: We currently do not support the case where cells disappear from the field of view. This implies that the number of
cells at time `t+1` must be greater than or equal to the number cells at time `t`. We have made a convenient
[`napari`](https://napari.org/) wrapper `mu.correct_decreasing_cell_frames` which will show you problematic frames and
allow you to interactively correct them.

### Analyzing

The submodule `mu.single_cell` computes some relevant quantities (such as center of mass and area) for each labelled region
in each image. These functions return an array with the same leading dimensions as your labels (typically (`S` and `T`)
and then an addition dimension for each cell which is called `CellID` by default and is padded with nans to keep an array
format when there are different numbers of cells at each position and time.

## Related projects

`microutil` is like philosophy in that whenever any part of it becomes particularly useful, that part spins
out and becomes it's own project. Typically we leave the original versions here for backwards compatibility
but any new users should prefer the newer standalone versions. Below are some such newer versions.

- `mu.load_mm_frames` -> [AICSImagio `TiffGlobReader`](https://allencellmodeling.github.io/aicsimageio/aicsimageio.readers.html#module-aicsimageio.readers.tiff_glob_reader)

  `TiffGlobReader` implements a more general logic for reading multi-file tiff datasets. It also has a handy
  static method that trivially handles micromanager images. It also is part of a AICSImageio which is trying to
  offer a unified API for microscopy datasets in python.

- `mu.single_cell` -> [`dask_regionprops`](https://github.com/jrussell25/dask-regionprops)

  Scikit-imgae [`regionprops`](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops)
  can compute many generic properties of labelled regions. It also ended up being more
  natural to store these properties in a dataframe where each row is a single region from a single image than in a
  padded array. As the name suggests, `dask_regionprops` uses dask to compute these properties in parallel and supports
  lazily computing them for larger than memory datasets.

- `mu.apply_unet`, `mu.individualize` -> [`yeast_mrcnn`](https://github.com/Hekstra-Lab/yeast-mrcnn)

  Modern neural architectures, especially [Mask-RCNN](https://arxiv.org/abs/1703.06870), can perform instance segmentation
  of many different classes without and intermediate semantic segmentation step. `yeast_mrcnn` uses `pytorch` and pre-trained
  models from `torchvision` to do this for yeast microscopy images. It also implements an efficient storage scheme for the labelled
  images.


## Contributions

We greatly welcome issues and pull requests if you find this library useful and want to help improve it.
