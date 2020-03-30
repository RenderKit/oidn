Training
========

The Intel Open Image Denoise source distribution includes a Python-based neural
network training toolkit (located in the `training` directory), which can be
used to train the denoising filter models using image datasets provided by the
user. The toolkit consists of multiple command-line scripts (e.g. dataset
preprocessing, training, inference, image comparison) that together can be used
to train and evaluate models. These scripts are the following:

-   `preprocess.py`: Preprocesses training and validation datasets.

-   `train.py`: Trains a model using preprocessed datasets.

-   `infer.py`: Performs inference on a set of images using the specified
    training result.

-   `export.py`: Exports a training result to the runtime model weights format.

-   `visualize.py`: Invokes TensorBoard for visualizing statistics of a training result.

-   `find_lr.py`: Tool for finding the optimal minimum and maximum learning
    rates.

-   `split_exr.py`: Splits a multi-channel EXR image into multiple feature
    images.

-   `compare_exr.py`: Compares two EXR images using the specified quality metrics.

-   `convert_exr.py`: Converts an EXR image to another format (including tonemapping).


Prerequisites
-------------

Before you can run the training toolkit you need the following prerequisites:

-   Python 3.7 or later

-   [PyTorch](https://pytorch.org/) 1.4 or later

-   [NumPy](https://numpy.org/) 1.17 or later

-   [OpenImageIO](http://openimageio.org/) 2.1 or later

-   [TensorBoard](https://www.tensorflow.org/tensorboard) 2.1 or later (*optional*)

The training toolkit has been tested only on Linux, thus other operating systems
are currently not supported.

Datasets
--------

A dataset should consist of a collection of noisy and corresponding noise-free
reference images. It is possible to have more than one noisy version of the
same image in the dataset, e.g. rendered at different samples per pixel and/or
using different seeds.

The training toolkit expects to have all datasets (e.g. training, validation)
in the same parent directory (e.g. `data`). Each dataset is stored in its own
subdirectory (e.g. `train`, `valid`), which can have an arbitrary name.

The images must be stored in [OpenEXR](https://www.openexr.com/) format (`.exr`
files) and the filenames must have a specific format but the files can be stored
in an arbitrary directory structure inside the dataset directory. The only
restriction is that all versions of an image (noisy images and the reference
image) must be located in the same directory. Each feature of an image (e.g.
color, albedo) must be stored in a separate image file, i.e. multi-channel EXR
image files are not supported. If you have multi-channel EXRs, you can split
them into separate images per feature using the included `split_exr.py` tool.

The filename of an image must consist of a name (any valid filename character
except `_` is allowed), the number of samples per pixel or whether it is the
reference (e.g. `0128spp`, `ref`), the identifier (ID) of the feature (e.g.
`hdr`, `alb`), and the file extension (`.exr`). This format as a regular
expression is the following:

```regexp
[^_]+_([0-9]+(spp)?|ref|reference|gt|target)\.(hdr|ldr|alb|nrm)\.exr
```

The number of samples per pixel should be padded with leading zeros to have a
fixed number of digits. If the reference image is not explicitly named as such
(e.g. `ref`, `reference`), the image with the most samples per pixel will be
considered the reference.

The following image features are supported:

Feature                ID        Channels
---------------------- --------- -----------------------------------
color (HDR)            `hdr`     3
color (LDR)            `ldr`     3
albedo                 `alb`     3
normal                 `nrm`     3
---------------------- --------- -----------------------------------
: Supported image features, their IDs, and their number of channels.

The following directory tree demonstates an example root dataset directory
(`data`) containing one dataset (`rt_train`) with HDR color and albedo
feature images:

```
data
`-- rt_train
    |-- scene1
    |   |-- view1_0001.alb.exr
    |   |-- view1_0001.hdr.exr
    |   |-- view1_0004.alb.exr
    |   |-- view1_0004.hdr.exr
    |   |-- view1_8192.alb.exr
    |   |-- view1_8192.hdr.exr
    |   |-- view2_0001.alb.exr
    |   |-- view2_0001.hdr.exr
    |   |-- view2_8192.alb.exr
    |   `-- view2_8192.hdr.exr
    |-- scene2_000008spp.alb.exr
    |-- scene2_000008spp.hdr.exr
    |-- scene2_000064spp.alb.exr
    |-- scene2_000064spp.hdr.exr
    |-- scene2_reference.alb.exr
    `-- scene2_reference.hdr.exr
```

### Preprocessing (preprocess.py)

Training and validation datasets can be used only after preprocessing them
using the `preprocess.py` script. This will convert the specified training
(`-t` or `--train_data` option) and validation datasets (`-v` or
`--valid_data` option) located in the root dataset directory (`-D` or
`--data_dir` option) to a format that can be loaded more efficiently during
training. All preprocessed datasets will be stored in a root preprocessed
dataset directory (`-P` or `--preproc_dir` option).

The preprocessing script requires the set of image features to include in the
preprocessed dataset, as command-line arguments. Only these specified features
will be available for training. Preprocessing also depends on the filter that
will be trained (e.g. determines which HDR/LDR transfer function has to be
used), which should be also specified (`-f` or `--filter` option). The
alternative is to manually specify the transfer function (`-x` or `--transfer`
option) and other filter-specific parameters, which could be useful for training
custom filters.

For example, to preprocess the training and validation datasets (`rt_train` and
`rt_valid`) with HDR color, albedo, and normal image features, for training the
`RT` filter, the following command can be used:

```
./preprocess.py hdr alb nrm --filter RT --train_data rt_train --valid_data rt_valid
```

For more details about using the preprocessing script, including other options,
please have a look at the help message:

```
./preprocess.py -h
```

Training (train.py)
-------------------

After preprocessing the datasets, it is possible to start training a model using
the `train.py` script. Similar to the preprocessing script, the input features
must be specified (could be a subset of the preprocessed features), and the
dataset names, directory paths, and the filter can be also passed.

The script will produce a training *result*, the name of which can be either
specified (`-r` or `--result` option) or automatically generated (by default).
Each result is stored in its own subdirectory, and these are located in the same
parent directory (`-R` or `--results_dir` option).


