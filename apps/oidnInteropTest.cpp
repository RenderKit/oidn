#ifdef _WIN32
    #include "oidnInteropTestDX.h"
#else
    #include "oidnInteropTestVk.h"
#endif

std::string usage() {
    return
        "Usage:\n"
        "   oidnInteropTest - run oidn using directX/Vulkan interop\n"
        "\n"
        "SYNOPSIS\n"
        "   oidnInteropTest --json <path> [OPTION]\n"
        "\n"
        "OPTIONS\n"
        "   --scale <float>  - scaling factor\n"
        "   --out <ext>      - extension of the out files, i.e. <path>.out.<n>.exr, default = out\n"
        "   --help           - print this help\n"
        "   --verbose        - vebose console output\n"
        ;
}

bool has_arg(const std::string& pname, std::vector<std::string>& args) {
    auto p = std::find(args.begin(), args.end(), pname);
    return p != args.end();
}

bool has_param(const std::string& pname, std::vector<std::string>& args) {
    auto p = std::find(args.begin(), args.end(), pname);
    return p != args.end() && (p+1) != args.end();
}

std::string parse(const std::string& pname, std::vector<std::string>& args) {
    auto p = std::find(args.begin(), args.end(), pname);
    if (p == args.end()) {
        throw std::runtime_error("could not parse parameter: " + pname + "\n" + usage());
    }

    ++p;
    if (p == args.end()) {
        throw std::runtime_error("could not parse parameter: " + pname + "\n" + usage());
    }

    return *p;
}

std::string parse_opt(const std::string& pname, std::vector<std::string>& args, std::string def = "") {
    return has_param(pname, args)? parse(pname, args) : def;
}


int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (has_arg("--help", args)) {
        std::cout << usage() << std::endl;
        return 0;
    }

    auto path_in = parse("--in", args);
    auto path_color = path_in + ".hdr.exr";
    auto path_albedo = path_in + ".alb1.exr";
    auto path_normal = path_in + ".nrm1.exr";

    auto path_out = parse_opt("--out", args, "./out.exr");
    auto scale = std::stof(parse_opt("--scale", args, "2.0f"));

    std::cout << "oidn interop test" << std::endl;
    std::cout << "    color:  " << path_color << std::endl;
    std::cout << "    albedo: " << path_albedo << std::endl;
    std::cout << "    normal: " << path_normal << std::endl;
    std::cout << "    output: " << path_out << std::endl;
    std::cout << "    scale:  " << scale << std::endl;


#ifdef _WIN32
    InteropTestDX test;
#else
    InteropTestVk test;
#endif

    std::cout << "Setting up OIDN device" << std::endl;
    test.setup_oidn_device();

    oidn::DeviceType deviceType = test.oidn_device.get<oidn::DeviceType>("type");
    const int versionMajor = test.oidn_device.get<int>("versionMajor");
    const int versionMinor = test.oidn_device.get<int>("versionMinor");
    const int versionPatch = test.oidn_device.get<int>("versionPatch");

    std::cout << std::setw(15) << std::left << "OIDN Device: ";
    switch (deviceType)
    {
    case oidn::DeviceType::Default: std::cout << std::setw(30) << std::left << "default"; break;
    case oidn::DeviceType::CPU:     std::cout << std::setw(30) << std::left << "CPU";     break;
    case oidn::DeviceType::SYCL:    std::cout << std::setw(30) << std::left << "SYCL";    break;
    case oidn::DeviceType::CUDA:    std::cout << std::setw(30) << std::left << "CUDA";    break;
    case oidn::DeviceType::HIP:     std::cout << std::setw(30) << std::left << "HIP";     break;
    case oidn::DeviceType::Metal:   std::cout << std::setw(30) << std::left << "Metal";   break;
    default:
      throw std::invalid_argument("invalid device type");
    }

    std::cout << "Version: " << versionMajor << "." << versionMinor << "." << versionPatch << std::endl;


    auto color = test.load_exr(path_color);
    auto albedo = test.load_exr(path_albedo);
    auto normal = test.load_exr(path_normal);

    uint32_t render_width, render_height, target_width, target_height;
    {
        render_width = color.get_width();
        render_height = color.get_height();
        target_width = render_width * scale;
        target_height = render_height * scale;

        std::cout << render_height << " x " << render_width << " -> "
                  << target_height << " x " << target_width << std::endl;
    }

    ImageBuffer out = test.create_buffer(target_width, target_height, oidn::Format::Half3);

    // Initialize the denoising filter
    std::cout << "Initializing filter" << std::endl;
    oidn::FilterRef filter = test.oidn_device.newFilter("RT");

    auto oidn_color = test.create_oidn_buffer(color);
    auto oidn_albedo = test.create_oidn_buffer(albedo);
    auto oidn_normal = test.create_oidn_buffer(normal);

    auto oidn_out = test.create_oidn_buffer(out);

    filter.setImage("color", oidn_color, color.get_format(), color.get_width(), color.get_height());
    filter.setImage("albedo", oidn_albedo, albedo.get_format(), albedo.get_width(), albedo.get_height());
    filter.setImage("normal", oidn_normal, normal.get_format(), normal.get_width(), normal.get_height());

    filter.setImage("output", oidn_out, out.get_format(), out.get_width(), out.get_height());

    filter.set("hdr", true);
    filter.set("inputScale", scale);
    filter.set("quality", oidn::Quality::Default);

    filter.commit();

    filter.execute();

    test.save_exr(out, "./out.exr");

    return 0;
}