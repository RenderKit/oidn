Overview
========

Intel Open Image Denoise is an open source library of high-performance,
high-quality denoising filters for images rendered with ray tracing.
Intel Open Image Denoise is part of the
[Intel® oneAPI Rendering Toolkit](https://software.intel.com/en-us/oneapi/render-kit)
and is released under the permissive
[Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).

The purpose of Intel Open Image Denoise is to provide an open, high-quality,
efficient, and easy-to-use denoising library that allows one to significantly
reduce rendering times in ray tracing based rendering applications. It filters
out the Monte Carlo noise inherent to stochastic ray tracing methods like path
tracing, reducing the amount of necessary samples per pixel by even multiple
orders of magnitude (depending on the desired closeness to the ground truth).
A simple but flexible C/C++ API ensures that the library can be easily
integrated into most existing or new rendering solutions.

At the heart of the Intel Open Image Denoise library is a collection of
efficient deep learning based denoising filters, which were trained to handle
a wide range of samples per pixel (spp), from 1 spp to almost fully converged.
Thus it is suitable for both preview and final frame rendering. The filters can
denoise images either using only the noisy color (*beauty*) buffer, or, to
preserve as much detail as possible, can optionally utilize auxiliary feature
buffers as well (e.g. albedo, normal). Such buffers are supported by most
renderers as arbitrary output variables (AOVs) or can be usually implemented
with little effort.

Although the library ships with a set of pre-trained filter models, it is not
mandatory to use these. To optimize a filter for a specific renderer, sample
count, content type, scene, etc., it is possible to train the model using the
included training toolkit and user-provided image datasets.

Intel Open Image Denoise supports Intel® 64 architecture compatible CPUs and
Apple Silicon, and runs on anything from laptops, to workstations, to compute
nodes in HPC systems. It is efficient enough to be suitable not only for offline
rendering, but, depending on the hardware used, also for interactive ray tracing.

Intel Open Image Denoise internally builds on top of
[Intel oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN),
and automatically exploits modern instruction sets like Intel SSE4, AVX2, and
AVX-512 to achieve high denoising performance. A CPU with support for at least
SSE4.1 or Apple Silicon is required to run Intel Open Image Denoise.


Support and Contact
-------------------

Intel Open Image Denoise is under active development, and though we do our best
to guarantee stable release versions a certain number of bugs, as-yet-missing
features, inconsistencies, or any other issues are still possible. Should you
find any such issues please report them immediately via the
[Intel Open Image Denoise GitHub Issue Tracker](https://github.com/OpenImageDenoise/oidn/issues)
(or, if you should happen to have a fix for it, you can also send us a pull
request); for missing features please contact us via email at
<openimagedenoise@googlegroups.com>.

Join our [mailing list](https://groups.google.com/d/forum/openimagedenoise/) to
receive release announcements and major news regarding Intel Open Image Denoise.
