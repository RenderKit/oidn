Training
========

The Intel Open Image Denoise source distribution includes a Python-based neural
network training toolkit (located in the `training` directory), which can be
used to train the denoising filter models with image datasets provided by the
user. This is an advanced feature of the library which usage requires some
background knowledge of machine learning and basic familiarity with deep
learning frameworks and toolkits (e.g. PyTorch or TensorFlow, TensorBoard).

The training toolkit consists of the following command-line scripts:

-   `preprocess.py`: Preprocesses training and validation datasets.

-   `train.py`: Trains a model using preprocessed datasets.

-   `infer.py`: Performs inference on a dataset using the specified training
    result.

-   `export.py`: Exports a training result to the runtime model weights format.

-   `find_lr.py`: Tool for finding the optimal minimum and maximum learning
    rates.

-   `visualize.py`: Invokes TensorBoard for visualizing statistics of a training
    result.

-   `split_exr.py`: Splits a multi-channel EXR image into multiple feature
    images.

-   `convert_image.py`: Converts a feature image to a different image format.

-   `compare_image.py`: Compares two feature images using the specified quality
    metrics.


Prerequisites
-------------

Before you can run the training toolkit you need the following prerequisites:

-   Linux (other operating systems are currently not supported)

-   Python 3.7 or later

-   [PyTorch](https://pytorch.org/) 1.8 or later

-   [NumPy](https://numpy.org/) 1.19 or later

-   [OpenImageIO](http://openimageio.org/) 2.1 or later

-   [TensorBoard](https://www.tensorflow.org/tensorboard) 2.4 or later (*optional*)

Devices
-------

Most scripts in the training toolkit support selecting what kind of device
(e.g. CPU, GPU) to use for the computations (`--device` or `-d` option). If
multiple devices of the same kind are available (e.g. multiple GPUs), the user
can specify which one of these to use (`--device_id` or `-k` option).
Additionally, some scripts, like `train.py`, support data-parallel execution
on multiple devices for faster performance (`--num_devices` or `-n` option).

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
files), and the filenames must have a specific format but the files can be
stored in an arbitrary directory structure inside the dataset directory. The
only restriction is that all versions of an image (noisy images and the
reference image) must be located in the same subdirectory. Each feature of an
image (e.g. color, albedo) must be stored in a separate image file, i.e.
multi-channel EXR image files are not supported. If you have multi-channel EXRs,
you can split them into separate images per feature using the included
`split_exr.py` tool.

An image filename must consist of a base name, a suffix with the number of
samples per pixel or whether it is the reference image (e.g. `_0128spp`,
`_ref`), the feature type extension (e.g. `.hdr`, `.alb`), and the image format
extension (`.exr`). The exact filename format as a regular expression is the
following:

```regexp
.+_([0-9]+(spp)?|ref|reference|gt|target)\.(hdr|ldr|sh1[xyz]|alb|nrm)\.exr
```

The number of samples per pixel should be padded with leading zeros to have a
fixed number of digits. If the reference image is not explicitly named as such
(i.e. has the number of samples instead), the image with the most samples per
pixel will be considered the reference.

The following image features are supported:

Feature Description                               Channels     File extension
------- ----------------------------------------- ------------ -------------------------------------
`hdr`   color (HDR)                               3            `.hdr.exr`
`ldr`   color (LDR)                               3            `.ldr.exr`
`sh1`   color (normalized L1 spherical harmonics) 3 × 3 images `.sh1x.exr`, `.sh1y.exr`, `.sh1z.exr`
`alb`   albedo                                    3            `.alb.exr`
`nrm`   normal                                    3            `.nrm.exr`
------- ----------------------------------------- ------------ -------------------------------------
: Image features supported by the training toolkit.

The following directory tree demonstrates an example root dataset directory
(`data`) containing one dataset (`rt_train`) with HDR color and albedo
feature images:

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

Preprocessing (preprocess.py)
-----------------------------

Training and validation datasets can be used only after preprocessing them
using the `preprocess.py` script. This will convert the specified training
(`--train_data` or `-t` option) and validation datasets (`--valid_data` or
`-v` option) located in the root dataset directory (`--data_dir` or `-D`
option) to a format that can be loaded more efficiently during training.
All preprocessed datasets will be stored in a root preprocessed dataset
directory (`--preproc_dir` or `-P` option).

The preprocessing script requires the set of image features to include in the
preprocessed dataset as command-line arguments. Only these specified features
will be available for training but it is not required to use all of them at the
same time. Thus, a single preprocessed dataset can be reused for training
multiple models with different combinations of the preprocessed features.

By default, all input features are assumed to be noisy, including the auxiliary
features (e.g. albedo, normal), each having versions at different samples per
pixel. However, it is also possible to train with noise-free auxiliary features,
in which case the reference auxiliary features are used instead of the various
noisy ones (`--clean_aux` option).

Preprocessing also depends on the filter that will be trained (e.g. determines
which HDR/LDR transfer function has to be used), which should be also specified
(`--filter` or `-f` option). The alternative is to manually specify the transfer
function (`--transfer` or `-x` option) and other filter-specific parameters,
which could be useful for training custom filters.

For example, to preprocess the training and validation datasets (`rt_train` and
`rt_valid`) with HDR color, albedo, and normal image features, for training the
`RT` filter, the following command can be used:

    ./preprocess.py hdr alb nrm --filter RT --train_data rt_train --valid_data rt_valid

It is possible to preprocess the same dataset multiple times, with possibly
different combinations of features and options. The training script will use
the most suitable and most recent preprocessed version depending on the training
parameters.

For more details about using the preprocessing script, including other options,
please have a look at the help message:

    ./preprocess.py -h

Training (train.py)
-------------------

The filters require separate trained models for each supported combination of
input features. Thus, depending on which combinations of features the user wants
to support for a particular filter, one or more models have to be trained.

After preprocessing the datasets, it is possible to start training a model using
the `train.py` script. Similar to the preprocessing script, the input features
must be specified (could be a subset of the preprocessed features), and the
dataset names, directory paths, and the filter can be also passed.

The tool will produce a training *result*, the name of which can be either
specified (`--result` or `-r` option) or automatically generated (by default).
Each result is stored in its own subdirectory, and these are located in a common
parent directory (`--results_dir` or `-R` option). If a training result already
exists, the tool will resume training that result from the latest checkpoint.

The default training hyperparameters should work reasonably well in general,
but some adjustments might be necessary for certain datasets to attain optimal
performance, most importantly: the number of epochs (`--num_epochs` or `-e`
option), the global mini-batch size (`--batch_size` or `-b` option), and the
learning rate. The training tool uses a one-cycle learning rate schedule with
cosine annealing, which can be configured by setting the base learning rate
(`--learning_rate` or `--lr` option), the maximum learning rate
(`--max_learning_rate` or `--max_lr` option), and the percentage of the cycle
spent increasing the learning rate (`--learning_rate_warmup` or `--lr_warmup`
option).

Example usage:

    ./train.py hdr alb --filter RT --train_data rt_train --valid_data rt_valid --result rt_hdr_alb

For finding the optimal learning rate range, we recommend using the included
`find_lr.py` script, which trains one epoch using an increasing learning rate
and logs the resulting losses in a comma-separated values (CSV) file. Plotting
the loss curve can show when the model starts to learn (the base learning
rate) and when it starts to diverge (the maximum learning rate).

The model is evaluated with the validation dataset at regular intervals
(`--num_valid_epochs` option), and checkpoints are also regularly created
(`--num_save_epochs` option) to save training progress. Also, some statistics
are logged (e.g. training and validation losses, learning rate) per epoch,
which can be later visualized with TensorBoard by running the `visualize.py`
script, e.g.:

    ./visualize.py --result rt_hdr_alb

Training is performed with mixed precision (FP16 and FP32) by default, if it
supported by the hardware, which makes training faster and use less memory.
However, in some rare cases this might cause some convergence issues. The
training precision can be manually set to FP32 if necessary (`--precision` or
`-p` option).

Inference (infer.py)
--------------------

A training result can be tested by performing inference on an image dataset
(`--input_data` or `-i` option) using the `infer.py` script. The dataset
does *not* have to be preprocessed. In addition to the result to use, it is
possible to specify which checkpoint to load as well (`-e` or `--num_epochs`
option). By default the latest checkpoint is loaded.

The tool saves the output images in a separate directory (`--output_dir` or
`-O` option) in the requested formats (`--format` or `-F` option). It also
evaluates a set of image quality metrics (`--metric` or `-M` option), e.g. PSNR,
SSIM, for images that have reference images available. All metrics are computed
in tonemapped non-linear sRGB space. Thus, HDR images are first tonemapped
(with Naughty Dog's Filmic Tonemapper from John Hable's *Uncharted 2: HDR
Lighting* presentation) and converted to sRGB before evaluating the metrics.

Example usage:

    ./infer.py --result rt_hdr_alb --input_data rt_test --format exr png --metric ssim

The inference tool supports prefiltering of auxiliary features as well, which
can be performed by specifying the list of training results for each feature to
prefilter (`--aux_results` or `-a` option). This is primarily useful for
evaluating the quality of models trained with clean auxiliary features.

Exporting Results (export.py)
-----------------------------

The training result produced by the `train.py` script cannot be immediately used
by the main library. It has to be first exported to the runtime model weights
format, a *Tensor Archive* (TZA) file. Running the `export.py` script for a
training result (and optionally a checkpoint epoch) will create a binary `.tza`
file in the directory of the result, which can be either used at runtime through
the API or it can be included in the library build by replacing one of the
built-in weights files.

Example usage:

    ./export.py --result rt_hdr_alb

Image Conversion and Comparison
-------------------------------

In addition to the already mentioned `split_exr.py` script, the toolkit contains
a few other image utilities as well.

`convert_image.py` converts a feature image to a different image format (and/or a
different feature, e.g. HDR color to LDR), performing tonemapping and other
transforms as well if needed. For HDR images the exposure can be adjusted by
passing a linear exposure scale (`--exposure` or `-E` option). Example usage:

    ./convert_image.py view1_0004.hdr.exr view1_0004.png --exposure 2.5

The `compare_image.py` script compares two feature images (preferably having the
dataset filename format to correctly detect the feature) using the specified
image quality metrics, similar to the `infer.py` tool. Example usage:

    ./compare_image.py view1_0004.hdr.exr view1_8192.hdr.exr --exposure 2.5 --metric mse ssim