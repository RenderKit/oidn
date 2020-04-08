Training
========

The Intel Open Image Denoise source distribution includes a Python-based neural
network training toolkit (located in the `training` directory), which can be
used to train the denoising filter models with image datasets provided by the
user. The toolkit consists of the following command-line scripts:

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
files), and the filenames must have a specific format but the files can be
stored in an arbitrary directory structure inside the dataset directory. The
only restriction is that all versions of an image (noisy images and the
reference image) must be located in the same subdirectory. Each feature of an
image (e.g. color, albedo) must be stored in a separate image file, i.e.
multi-channel EXR image files are not supported. If you have multi-channel EXRs,
you can split them into separate images per feature using the included
`split_exr.py` tool.

An image filename must consist of a name, the number of samples per pixel or
whether it is the reference (e.g. `0128spp`, `ref`), the identifier (ID) of the
feature (e.g. `hdr`, `alb`), and the file extension (`.exr`). The exact format
as a regular expression is the following:

```regexp
.+_([0-9]+(spp)?|ref|reference|gt|target)\.(hdr|ldr|alb|nrm)\.exr
```

The number of samples per pixel should be padded with leading zeros to have a
fixed number of digits. If the reference image is not explicitly named as such
(i.e. has the number of samples instead), the image with the most samples per
pixel will be considered the reference.

The following image features are supported:

Feature                ID        Channels
---------------------- --------- -----------------------------------
color (HDR)            `hdr`     3
color (LDR)            `ldr`     3
albedo                 `alb`     3
normal                 `nrm`     3
---------------------- --------- -----------------------------------
: Supported image features, their IDs, and their number of channels.

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
(`-t` or `--train_data` option) and validation datasets (`-v` or
`--valid_data` option) located in the root dataset directory (`-D` or
`--data_dir` option) to a format that can be loaded more efficiently during
training. All preprocessed datasets will be stored in a root preprocessed
dataset directory (`-P` or `--preproc_dir` option).

The preprocessing script requires the set of image features to include in the
preprocessed dataset as command-line arguments. Only these specified features
will be available for training. Preprocessing also depends on the filter that
will be trained (e.g. determines which HDR/LDR transfer function has to be
used), which should be also specified (`-f` or `--filter` option). The
alternative is to manually specify the transfer function (`-x` or `--transfer`
option) and other filter-specific parameters, which could be useful for training
custom filters.

For example, to preprocess the training and validation datasets (`rt_train` and
`rt_valid`) with HDR color, albedo, and normal image features, for training the
`RT` filter, the following command can be used:

    ./preprocess.py hdr alb nrm --filter RT --train_data rt_train --valid_data rt_valid

For more details about using the preprocessing script, including other options,
please have a look at the help message:

    ./preprocess.py -h

Training (train.py)
-------------------

After preprocessing the datasets, it is possible to start training a model using
the `train.py` script. Similar to the preprocessing script, the input features
must be specified (could be a subset of the preprocessed features), and the
dataset names, directory paths, and the filter can be also passed.

The tool will produce a training *result*, the name of which can be either
specified (`-r` or `--result` option) or automatically generated (by default).
Each result is stored in its own subdirectory, and these are located in a common
parent directory (`-R` or `--results_dir` option). If a training result already
exists, the tool will resume training that result from the latest checkpoint, or
from an earlier checkpoint at the specified epoch (`-c` or `--checkpoint`
option).

The default training hyperparameters should work reasonably well in general,
but some adjustments might be necessary for certain datasets to attain optimal
performance, most importantly: the number of epochs (`-e` or `--epochs` option),
the mini-batch size (`--bs` or `--batch_size` option), and the learning rate.
The training tool uses a cyclical learning rate (CLR) with the `triangular2`
scaling policy and an optional linear ramp-down at the end. The learning rate
schedule can be configured by setting the base learning rate (`--lr` or
`--learning_rate` option), the maximum learning rate (`--max_lr` or
`--max_learning_rate` option), and the total cycle size in number of epochs
(`--lr_cycle_epochs` option). If there is an incomplete cycle at the end, the
learning rate will be linearly ramped down to almost zero.

Example usage:

    ./train.py hdr alb --filter RT --train_data rt_train --valid_data rt_valid --result rt_hdr_alb

For finding the optimal learning rate range we recommend using the included
`find_lr.py` script, which trains one epoch using an increasing learning rate
and logs the resulting losses in a comma-separated values (CSV) file. Plotting
the loss curve can show when the model starts to learn (the base learning
rate) and when it starts to diverge (the maximum learning rate).

The model is evaluated with the validation dataset at regular intervals
(`--valid_epochs` option), and checkpoints are also regularly created
(`--save_epochs` option) to save training progress. Also, some statistics
are logged (e.g. training and validation losses, learning rate) at a specified
frequency (`--log_steps` option), which can be later visualized with TensorBoard
by running the `visualize.py` script, e.g.:

    ./visualize.py --result rt_hdr_alb

Inference (infer.py)
--------------------

A training result can be tested by performing inference on an image dataset
(`-i` or `--input_data` option) using the `infer.py` script. The dataset
does *not* have to be preprocessed. In addition to the result to use, it is
possible to specify which checkpoint to load as well. By default the latest
checkpoint is loaded.

The tool saves the output images in a separate directory (`-O` or
`--output_dir`) in the requested formats (`-F` or `--format` option). It also
evaluates a set of image quality metrics (`-M` or `--metric` option), e.g. SSIM,
MSE, for images that have reference images available. All metrics are computed
in tonemapped non-linear sRGB space. Thus, HDR images are first tonemapped
(with Naughty Dog's Filmic Tonemapper from John Hable's *Uncharted 2: HDR
Lighting* presentation) and converted to sRGB before evaluating the metrics.

Example usage:

    ./infer.py --result rt_hdr_alb --input_data rt_test --format exr png --metric ssim

Exporting Results (export.py)
-----------------------------

The training result produced by the `train.py` script cannot be immediately used
by the main library. It has to be first exported to the runtime model weights
format, a *Tensor Archive* (TZA) file. Running the `export.py` script for a
training result (and optionally a checkpoint) will create a binary `.tza` file
in the directory of the result, which can be either used at runtime through the
API or it can be included in the library build by replacing one of the built-in
weights files.

Example usage:

    ./export.py --result rt_hdr_alb

Image Conversion and Comparison
-------------------------------

In addition to the already mentioned `split_exr.py` script, the toolkit contains
a few other image utilities as well.

`convert_image.py` converts a feature image to a different image format (and/or a
different feature, e.g. HDR color to LDR), performing tonemapping and other
transforms as well if needed. For HDR images the exposure can be adjusted by
passing a linear exposure scale (`-E` or `--exposure` option). Example usage:

    ./convert_image.py view1_0004.hdr.exr view1_0004.png --exposure 2.5

The `compare_image.py` script compares two feature images (preferably having the
dataset filename format to correctly detect the feature) using the specified
image quality metrics, similar to the `infer.py` tool. Example usage:

    ./compare_image.py view1_0004.hdr.exr view1_8192.hdr.exr --exposure 2.5 --metric mse ssim