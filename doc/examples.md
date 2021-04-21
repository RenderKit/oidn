Examples
========

Intel Open Image Denoise ships with a couple of simple example applications.

oidnDenoise
-----------

`oidnDenoise` is a minimal working example demonstrating how to use Intel Open
Image Denoise, which can be found at `apps/oidnDenoise.cpp`. It uses the C++11
convenience wrappers of the C99 API.

This example is a simple command-line application that denoises the provided
image, which can optionally have auxiliary feature images as well (e.g. albedo
and normal). By default the images must be stored in the [Portable
FloatMap](http://www.pauldebevec.com/Research/HDR/PFM/) (PFM) format, and the
color values must be encoded in little-endian format. To enable other image
formats (e.g. OpenEXR, PNG) as well, the project has to be rebuilt with
OpenImageIO support enabled.

Running `oidnDenoise` without any arguments or the `-h` argument will bring up
a list of command-line options.

oidnBenchmark
-------------

`oidnBenchmark` is a basic command-line benchmarking application for measuring
denoising speed, which can be found at `apps/oidnBenchmark.cpp`.

Running `oidnBenchmark` with the `-h` argument will bring up a list of
command-line options.
