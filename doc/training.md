Training
========

The Intel Open Image Denoise source distribution includes a Python-based neural
network training toolkit (see the `training` directory), which can be used to
train the denoising filter models using image datasets provided by the user.
The toolkit consists of multiple command-line tools (e.g. dataset preprocessing,
training, inference, image comparison) that together can be used to train and
evaluate models.


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
image files are not supported.

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