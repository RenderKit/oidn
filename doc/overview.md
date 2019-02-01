Open Image Denoise Overview
===========================

Intel® Open Image Denoise is a collection of high-performance,
high-quality denoising filters for images rendered with ray tracing.
Open Image Denoise is part of the
[Intel Rendering Framework](https://software.intel.com/en-us/rendering-framework)
and is released under the permissive
[Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0).

The purpose of Open Image Denoise is to provide an open, high-quality,
efficient, and easy-to-use denoising library that allows one to significantly
reduce rendering times in ray tracing based rendering applications. It filters
out the Monte Carlo noise inherent to stochastic ray tracing methods like path
tracing, reducing the amount of necessary samples per pixel by even multiple
orders of magnitude (depending on the desired closeness to the ground truth).

Open Image Denoise supports Intel® 64 architecture based CPUs and compatible
architectures, and runs on anything from laptops, to workstations, to compute
nodes in HPC systems. It is efficient enough to be suitable not only for
offline rendering, but, depending on the hardware used, also for interactive
ray tracing.

Open Image Denoise internally builds on top of
[Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN)](https://github.com/intel/mkl-dnn),
and fully exploits modern instruction sets like Intel SSE4, AVX2, and AVX-512
to achieve high denoising performance. A CPU with support for at least SSE4.1
is required to run Open Image Denoise.


Support and Contact
-------------------

Open Image Denoise is under active development, and though we do our best to
guarantee stable release versions a certain number of bugs, as-yet-missing
features, inconsistencies, or any other issues are still possible. Should you
find any such issues please report them immediately via the
[Open Image Denoise GitHub Issue Tracker](https://github.com/OpenImageDenoise/oidn/issues)
(or, if you should happen to have a fix for it, you can also send us a pull
request); for missing features please contact us via email at
<openimagedenoise@googlegroups.com>.

For recent news, updates, and announcements, please see our complete
[news/updates] page.

Join our [mailing list](https://groups.google.com/d/forum/openimagedenoise/) to
receive release announcements and major news regarding Open Image Denoise.

