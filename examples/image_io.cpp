// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <fstream>
#include <cmath>
#include "image_io.h"

#ifdef HAS_OPEN_EXR
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#endif

namespace oidn {

  namespace
  {
    std::string getExtension(const std::string& filename)
    {
      const size_t pos = filename.find_last_of('.');
      if (pos == std::string::npos)
        return ""; // no extension
      else
        return filename.substr(pos + 1);
    }

#ifdef HAS_OPEN_EXR
    Imf::FrameBuffer frameBufferForEXR(const ImageBuffer& image)
    {
      Imf::FrameBuffer frameBuffer;
      int bytePixelStride = image.getChannels()*sizeof(float);
      int byteRowStride = image.getWidth()*bytePixelStride;
      frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, (char*)&image[0], bytePixelStride, byteRowStride));
      frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, (char*)&image[1], bytePixelStride, byteRowStride));
      frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, (char*)&image[2], bytePixelStride, byteRowStride));
      return frameBuffer;
    }

    ImageBuffer loadImageEXR(const std::string& filename)
    {
      Imf::InputFile inputFile(filename.c_str());
      if (!inputFile.header().channels().findChannel("R") ||
          !inputFile.header().channels().findChannel("G") ||
          !inputFile.header().channels().findChannel("B"))
        throw std::invalid_argument("image must have 3 channels");
      Imath::Box2i dataWindow = inputFile.header().dataWindow();
      ImageBuffer image(dataWindow.max.x-dataWindow.min.x+1, dataWindow.max.y-dataWindow.min.y+1, 3);
      inputFile.setFrameBuffer(frameBufferForEXR(image));
      inputFile.readPixels(dataWindow.min.y, dataWindow.max.y);
      return image;
    }

    void saveImageEXR(const ImageBuffer& image, const std::string& filename)
    {
      if (image.getChannels() != 3)
        throw std::invalid_argument("image must have 3 channels");
      Imf::Header header(image.getWidth(), image.getHeight(), 1, Imath::V2f(0, 0), image.getWidth(), Imf::INCREASING_Y, Imf::ZIP_COMPRESSION);
      header.channels().insert("R", Imf::Channel(Imf::FLOAT));
      header.channels().insert("G", Imf::Channel(Imf::FLOAT));
      header.channels().insert("B", Imf::Channel(Imf::FLOAT));
      Imf::OutputFile outputFile(filename.c_str(), header);
      outputFile.setFrameBuffer(frameBufferForEXR(image));
      outputFile.writePixels(image.getHeight());
    }
#endif

    ImageBuffer loadImagePFM(const std::string& filename)
    {
      // Open the file
      std::ifstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open file '" + filename + "'");

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
      ImageBuffer image(W, H, C);

      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < C; ++c)
          {
            float x;
            file.read((char*)&x, sizeof(float));
            image[((H-1-h)*W + w) * C + c] = x * scale;
          }
        }
      }

      if (file.fail())
        throw std::runtime_error("invalid PFM image");

      return image;
    }

    void saveImagePFM(const std::string& filename, const ImageBuffer& image)
    {
      const int H = image.getHeight();
      const int W = image.getWidth();
      const int C = image.getChannels();

      // Open the file
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open file: '" + filename + "'");

      // Write the header
      file << "PF" << std::endl;
      file << W << " " << H << std::endl;
      file << "-1.0" << std::endl;

      // Write the pixels
      for (int h = 0; h < H; ++h)
      {
        for (int w = 0; w < W; ++w)
        {
          for (int c = 0; c < 3; ++c)
          {
            const float x = image[((H-1-h)*W + w) * C + c];
            file.write((char*)&x, sizeof(float));
          }
        }
      }
    }

    void saveImagePPM(const std::string& filename, const ImageBuffer& image)
    {
      if (image.getChannels() != 3)
        throw std::invalid_argument("image must have 3 channels");
      const int H = image.getHeight();
      const int W = image.getWidth();
      const int C = image.getChannels();

      // Open the file
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open file: '" + filename + "'");

      // Write the header
      file << "P6" << std::endl;
      file << W << " " << H << std::endl;
      file << "255" << std::endl;

      // Write the pixels
      for (int i = 0; i < W*H; ++i)
      {
        for (int c = 0; c < 3; ++c)
        {
          float x = image[i*C+c];
          x = pow(x, 1.f/2.2f);
          int ch = std::min(std::max(int(x * 255.f), 0), 255);
          file.put(char(ch));
        }
      }
    }
  }

  ImageBuffer loadImage(const std::string& filename)
  {
    const std::string ext = getExtension(filename);
#ifdef HAS_OPEN_EXR
    if (ext == "exr")
      return loadImageEXR(filename);
    else
#endif
    if (ext == "pfm")
      return loadImagePFM(filename);
    else
      throw std::runtime_error("unsupported image file format");
  }

  void saveImage(const std::string& filename, const ImageBuffer& image)
  {
    const std::string ext = getExtension(filename);
#ifdef HAS_OPEN_EXR
    if (ext == "exr")
      saveImageEXR(image, filename);
    else
#endif
    if (ext == "pfm")
      saveImagePFM(filename, image);
    else if (ext == "ppm")
      saveImagePPM(filename, image);
    else
      throw std::runtime_error("unsupported image file format");
  }

} // namespace oidn
