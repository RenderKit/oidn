Version History
---------------

-   Fixed wrong HDR output when the input contains infinities/NaNs
-   Added OIDN_STATIC_LIB CMake option for building as a static library
    (requires CMake 3.13.0 or later)
-   Fixed CMake error when adding the library with add_subdirectory() to a project

### Changes in v0.8.1:

-   Fixed wrong path to TBB in the generated CMake configs
-   Fixed wrong rpath in the binaries
-   Fixed compile error on some macOS systems
-   Fixed minor compile issues with Visual Studio
-   Lowered the CPU requirement to SSE4.1
-   Minor example update

### Changes in v0.8.0:

-   Initial beta release
