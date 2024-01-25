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
Thus it is suitable for both preview and final-frame rendering. The filters can
denoise images either using only the noisy color (*beauty*) buffer, or, to
preserve as much detail as possible, can optionally utilize auxiliary feature
buffers as well (e.g. albedo, normal). Such buffers are supported by most
renderers as arbitrary output variables (AOVs) or can be usually implemented
with little effort.

Although the library ships with a set of pre-trained filter models, it is not
mandatory to use these. To optimize a filter for a specific renderer, sample
count, content type, scene, etc., it is possible to train the model using the
included training toolkit and user-provided image datasets.

Intel Open Image Denoise supports a wide variety of CPUs and GPUs from different
vendors:

-   Intel® 64 architecture compatible CPUs (with at least SSE4.1)

-   ARM64 (AArch64) architecture CPUs (e.g. Apple silicon CPUs)

-   Intel Xe architecture dedicated and integrated GPUs, including Intel® Arc™
    A-Series Graphics, Intel® Data Center GPU Flex Series, Intel® Data Center
    GPU Max Series, Intel® Iris® Xe Graphics, Intel® Core™ Ultra Processors with
    Intel® Arc™ Graphics, 11th-14th Gen Intel® Core™ processor graphics, and
    related Intel Pentium® and Celeron® processors (Xe-LP, Xe-LPG, Xe-HPG, and
    Xe-HPC microarchitectures)

-   NVIDIA GPUs with Volta, Turing, Ampere, Ada Lovelace, and Hopper
    architectures

-   AMD GPUs with RDNA2 (Navi 21 only) and RDNA3 (Navi 3x) architectures

-   Apple silicon GPUs (M1 and newer)

It runs on most machines ranging from laptops to workstations and compute nodes
in HPC systems. It is efficient enough to be suitable not only for offline
rendering, but, depending on the hardware used, also for interactive or even
real-time ray tracing.

Intel Open Image Denoise exploits modern instruction sets like SSE4, AVX2,
AVX-512, and NEON on CPUs, Intel® Xe Matrix Extensions (Intel® XMX) on Intel
GPUs, and tensor cores on NVIDIA GPUs to achieve high denoising performance.


System Requirements
-------------------

You need an Intel® 64 (with SSE4.1) or ARM64 architecture compatible CPU to run
Intel Open Image Denoise, and you need a 64-bit Windows, Linux, or macOS
operating system as well.

For Intel GPU support, please also install the latest Intel graphics drivers:

-   Windows:
    [Intel® Graphics Driver](https://www.intel.com/content/www/us/en/download/726609/intel-arc-iris-xe-graphics-whql-windows.html)
    31.0.101.4953 or newer

-   Linux:
    [Intel® software for General Purpose GPU capabilities](https://dgpu-docs.intel.com/driver/installation.html)
    release [20230323](https://dgpu-docs.intel.com/releases/stable_602_20230323.html)
    or newer

Using older driver versions is *not* supported and Intel Open Image Denoise
might run with only limited capabilities, have suboptimal performance or might
be unstable. Also, Resizable BAR *must* be enabled in the BIOS for Intel
dedicated GPUs if running on Linux, and strongly recommended if running on
Windows.

For NVIDIA GPU support, please also install the latest
[NVIDIA graphics drivers](https://www.nvidia.com/en-us/geforce/drivers/):

-   Windows: Version 452.39 or newer

-   Linux: Version 450.80.02 or newer

For AMD GPU support, please also install the latest
[AMD graphics drivers](https://www.amd.com/en/support):

-   Windows: AMD Software: Adrenalin Edition 23.4.3 Driver Version 22.40.51.05 or newer

-   Linux: [Radeon Software for Linux](https://www.amd.com/en/support/linux-drivers)
    version 22.40.5 or newer

For Apple GPU support, macOS Ventura or newer is required.

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


Citation
--------

If you use Intel Open Image Denoise in a research publication, please cite the
project using the following BibTeX entry:

```bibtex
@misc{OpenImageDenoise,
  author = {Attila T. {\'A}fra},
  title  = {{Intel\textsuperscript{\textregistered} Open Image Denoise}},
  year   = {2023},
  note   = {\url{https://www.openimagedenoise.org}}
}
```