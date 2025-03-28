# Security Policy
Intel is committed to rapidly addressing security vulnerabilities affecting our customers and providing clear guidance on the solution, impact, severity and mitigation. 

## Reporting a Vulnerability
Please report any security vulnerabilities in this project utilizing the guidelines [here](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).


## Security Considerations
When integrating this library into your application, you are responsible for ensuring overall application security. If you are building and using the **Open Image Denoise (OIDN) library** from source, be aware of potential security risks, including **DLL planting attacks** and other dynamic library loading vulnerabilities.

To help mitigate such risks, we provide the `OIDN_DEPENDENTLOADFLAG` CMake option. This allows you to specify the appropriate **Windows linker flag** based on your security requirements. By default, this option is not set, and you should configure it according to your deployment needs. See the official Microsoft documentation for more details: [DEPENDENTLOADFLAG Linker Option](https://learn.microsoft.com/en-us/cpp/build/reference/dependentloadflag?view=msvc-170).

To enable and set this flag, configure your build with:

cmake -DOIDN_DEPENDENTLOADFLAG=<value> ..

For more information on securing dynamic library loading, refer to Microsoft's official documentation on:  

- [Safe DLL Search Mode](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)  
- [Dynamic-Link Library Security](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-security)  
