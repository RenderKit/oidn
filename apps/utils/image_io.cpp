// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_io.h"
#include <fstream>

#if defined(OIDN_USE_OPENIMAGEIO)
  #include <OpenImageIO/imageio.h>
#endif

OIDN_NAMESPACE_BEGIN

  namespace
  {
    inline float srgbForward(float y)
    {
      return (y <= 0.0031308f) ? (12.92f * y) : (1.055f * std::pow(y, 1.f/2.4f) - 0.055f);
    }

    inline float srgbInverse(float x)
    {
      return (x <= 0.04045f) ? (x / 12.92f) : std::pow((x + 0.055f) / 1.055f, 2.4f);
    }

    void srgbForward(ImageBuffer& image)
    {
      for (size_t i = 0; i < image.getSize(); ++i)
        image.set(i, srgbForward(image.get(i)));
    }

    void srgbInverse(ImageBuffer& image)
    {
      for (size_t i = 0; i < image.getSize(); ++i)
        image.set(i, srgbInverse(image.get(i)));
    }

    std::string getExtension(const std::string& filename)
    {
      const size_t pos = filename.find_last_of('.');
      if (pos == std::string::npos)
        return ""; // no extension
      else
      {
        std::string ext = filename.substr(pos + 1);
        for (auto& c : ext) c = tolower(c);
        return ext;
      }
    }

    std::shared_ptr<ImageBuffer> loadImagePFM(const DeviceRef& device,
                                              const std::string& filename,
                                              int numChannels,
                                              Format dataType)
    {
      // Open the file
      std::ifstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      // Read the header
      std::string id;
      file >> id;
      int C;
      if (id == "PF")
        C = 3;
      else if (id == "Pf")
        C = 1;
      else
        throw std::runtime_error("invalid PFM image");

      if (numChannels == 0)
        numChannels = C;
      else if (C < numChannels)
        throw std::runtime_error("not enough image channels");

      if (dataType == Format::Undefined)
        dataType = Format::Float;

      int H, W;
      file >> W >> H;

      float scale;
      file >> scale;

      file.get(); // skip newline

      if (file.fail())
        throw std::runtime_error("invalid PFM image");

      if (scale >= 0.f)
        throw std::runtime_error("big-endian PFM images are not supported");
      scale = fabs(scale);

      // Read the pixels
      auto image = std::make_shared<ImageBuffer>(device, W, H, numChannels, dataType);

      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < C; ++c)
          {
            float x;
            file.read((char*)&x, sizeof(float));
            if (c < numChannels)
              image->set((size_t(H-1-h)*W + w) * numChannels + c, x * scale);
          }
        }
      }

      if (file.fail())
        throw std::runtime_error("invalid PFM image");

      return image;
    }

    void saveImagePFM(const std::string& filename, const ImageBuffer& image)
    {
      const int H = image.getH();
      const int W = image.getW();
      const int C = image.getC();

      std::string id;
      if (C == 3)
        id = "PF";
      else if (C == 1)
        id = "Pf";
      else
        throw std::runtime_error("unsupported number of channels");

      // Open the file
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      // Write the header
      file << id << std::endl;
      file << W << " " << H << std::endl;
      file << "-1.0" << std::endl;

      // Write the pixels
      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < C; ++c)
          {
            const float x = image.get((size_t(H-1-h)*W + w) * C + c);
            file.write((char*)&x, sizeof(x));
          }
        }
      }
    }

    std::shared_ptr<ImageBuffer> loadImagePHM(const DeviceRef& device,
                                              const std::string& filename,
                                              int numChannels,
                                              Format dataType)
    {
      // Open the file
      std::ifstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      // Read the header
      std::string id;
      file >> id;
      int C;
      if (id == "PH")
        C = 3;
      else if (id == "Ph")
        C = 1;
      else
        throw std::runtime_error("invalid PHM image");

      if (numChannels == 0)
        numChannels = C;
      else if (C < numChannels)
        throw std::runtime_error("not enough image channels");

      if (dataType == Format::Undefined)
        dataType = Format::Half;

      int H, W;
      file >> W >> H;

      float scale;
      file >> scale;

      file.get(); // skip newline

      if (file.fail())
        throw std::runtime_error("invalid PHM image");

      if (scale >= 0.f)
        throw std::runtime_error("big-endian PHM images are not supported");
      scale = fabs(scale);

      // Read the pixels
      auto image = std::make_shared<ImageBuffer>(device, W, H, numChannels, dataType);

      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < C; ++c)
          {
            half x;
            file.read((char*)&x, sizeof(x));
            if (c < numChannels)
            {
              if (scale == 1.f)
              {
                image->set((size_t(H-1-h)*W + w) * numChannels + c, x);
              }
              else
              {
                const float xs = float(x) * scale;
                image->set((size_t(H-1-h)*W + w) * numChannels + c, xs);
              }
            }
          }
        }
      }

      if (file.fail())
        throw std::runtime_error("invalid PHM image");

      return image;
    }

    void saveImagePHM(const std::string& filename, const ImageBuffer& image)
    {
      const int H = image.getH();
      const int W = image.getW();
      const int C = image.getC();

      std::string id;
      if (C == 3)
        id = "PH";
      else if (C == 1)
        id = "Ph";
      else
        throw std::runtime_error("unsupported number of channels");

      // Open the file
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      // Write the header
      file << id << std::endl;
      file << W << " " << H << std::endl;
      file << "-1.0" << std::endl;

      // Write the pixels
      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < C; ++c)
          {
            const half x = image.get<half>((size_t(H-1-h)*W + w) * C + c);
            file.write((char*)&x, sizeof(x));
          }
        }
      }
    }

    void saveImagePPM(const std::string& filename, const ImageBuffer& image)
    {
      if (image.getC() != 3)
        throw std::invalid_argument("image must have 3 channels");
      const int H = image.getH();
      const int W = image.getW();
      const int C = image.getC();

      // Open the file
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      // Write the header
      file << "P6" << std::endl;
      file << W << " " << H << std::endl;
      file << "255" << std::endl;

      // Write the pixels
      for (int i = 0; i < W*H; ++i)
      {
        for (int c = 0; c < 3; ++c)
        {
          const float x = image.get(i*C+c);
          const int ch = std::min(std::max(int(x * 255.f), 0), 255);
          file.put(char(ch));
        }
      }
    }

  #ifdef OIDN_USE_OPENIMAGEIO
    std::shared_ptr<ImageBuffer> loadImageOIIO(const DeviceRef& device,
                                               const std::string& filename,
                                               int numChannels,
                                               Format dataType)
    {
      auto in = OIIO::ImageInput::open(filename);
      if (!in)
        throw std::runtime_error("cannot open image file: '" + filename + "'");

      const OIIO::ImageSpec& spec = in->spec();
      if (numChannels == 0)
        numChannels = spec.nchannels;
      else if (spec.nchannels < numChannels)
        throw std::runtime_error("not enough image channels");

      if (dataType == Format::Undefined)
        dataType = (spec.channelformat(0) == OIIO::TypeDesc::HALF) ? Format::Half : Format::Float;

      auto image = std::make_shared<ImageBuffer>(device, spec.width, spec.height, numChannels, dataType);
      bool success = in->read_image(0, 0, 0, numChannels, dataType == Format::Half ? OIIO::TypeDesc::HALF : OIIO::TypeDesc::FLOAT, image->getData());
      in->close();

  #if OIIO_VERSION < 10903
      OIIO::ImageInput::destroy(in);
  #endif

      if (!success)
        throw std::runtime_error("failed to read image data");

      return image;
    }

    void saveImageOIIO(const std::string& filename, const ImageBuffer& image)
    {
      auto out = OIIO::ImageOutput::create(filename);
      if (!out)
        throw std::runtime_error("cannot save unsupported image file format: '" + filename + "'");

      OIIO::TypeDesc format;
      switch (image.getDataType())
      {
      case Format::Float:
        format = OIIO::TypeDesc::FLOAT;
        break;
      case Format::Half:
        format = OIIO::TypeDesc::HALF;
        break;
      default:
        throw std::runtime_error("unsupported image data type");
      }

      OIIO::ImageSpec spec(image.getW(),
                           image.getH(),
                           image.getC(),
                           format);

      if (!out->open(filename, spec))
        throw std::runtime_error("cannot create image file: '" + filename + "'");
      bool success = out->write_image(format, image.getData());
      out->close();

  #if OIIO_VERSION < 10903
      OIIO::ImageOutput::destroy(out);
  #endif

      if (!success)
        throw std::runtime_error("failed to write image data");
    }
  #endif

  } // namespace

  std::shared_ptr<ImageBuffer> loadImage(const DeviceRef& device,
                                         const std::string& filename,
                                         int numChannels,
                                         Format dataType)
  {
    const std::string ext = getExtension(filename);
    std::shared_ptr<ImageBuffer> image;

    if (ext == "pfm")
      image = loadImagePFM(device, filename, numChannels, dataType);
    else if (ext == "phm")
      image = loadImagePHM(device, filename, numChannels, dataType);
    else
#if OIDN_USE_OPENIMAGEIO
      image = loadImageOIIO(device, filename, numChannels, dataType);
#else
      throw std::runtime_error("cannot load unsupported image file format: '" + filename + "'");
#endif

    return image;
  }

  void saveImage(const std::string& filename, const ImageBuffer& image)
  {
    const std::string ext = getExtension(filename);
    if (ext == "pfm")
      saveImagePFM(filename, image);
    else if (ext == "phm")
      saveImagePHM(filename, image);
    else if (ext == "ppm")
      saveImagePPM(filename, image);
    else
#if OIDN_USE_OPENIMAGEIO
      saveImageOIIO(filename, image);
#else
      throw std::runtime_error("cannot write unsupported image file format: '" + filename + "'");
#endif
  }

  bool isSrgbImage(const std::string& filename)
  {
    const std::string ext = getExtension(filename);
    return ext != "pfm" && ext != "phm" && ext != "exr" && ext != "hdr";
  }

  std::shared_ptr<ImageBuffer> loadImage(const DeviceRef& device,
                                         const std::string& filename,
                                         int numChannels,
                                         bool srgb,
                                         Format dataType)
  {
    auto image = loadImage(device, filename, numChannels, dataType);
    if (!srgb && isSrgbImage(filename))
      srgbInverse(*image);
    return image;
  }

  void saveImage(const std::string& filename, const ImageBuffer& image, bool srgb)
  {
    if (!srgb && isSrgbImage(filename))
    {
      std::shared_ptr<ImageBuffer> newImage = image.clone();
      srgbForward(*newImage);
      saveImage(filename, *newImage);
    }
    else
    {
      saveImage(filename, image);
    }
  }

OIDN_NAMESPACE_END