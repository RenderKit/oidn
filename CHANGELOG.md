Version History
---------------

### Changes in v0.5.0:

-   Added high-quality HDR denoising
-   Improved LDR denoising quality

### Changes in v0.4.0:

-   Renamed "Autoencoder" filter to "RT" (ray tracing)
-   Improved performance on processors with AVX-512 support
-   Added support for Intel(R) Xeon Phi(TM) processor x200 family (formerly Knights Landing)
-   Added support for macOS
-   Fixed crash when creating a device on an unsupported CPU

### Changes in v0.3.0:

-   Major HDR quality improvement (still experimental)
-   Fixed NaNs in the output for certain inputs
-   Windows versions earlier than 7 are now supported

### Changes in v0.2.1:

-   Fixed crash on Windows when running on CPUs with AVX-512 support
-   Added OIDN_DEVICE_TYPE_DEFAULT

### Changes in v0.2.0:

-   Initial alpha release
