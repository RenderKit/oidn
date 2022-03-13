// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "buffer.h"
#include "tensor.h"
#include "image.h"

namespace oidn {

  std::shared_ptr<Tensor> Buffer::newTensor(const TensorDesc& desc, ptrdiff_t relByteOffset)
  {
    size_t byteOffset = relByteOffset >= 0 ? relByteOffset : getByteSize() + relByteOffset;
    return getDevice()->newTensor(this, desc, byteOffset);
  }

  std::shared_ptr<Image> Buffer::newImage(const ImageDesc& desc, ptrdiff_t relByteOffset)
  {
    size_t byteOffset = relByteOffset >= 0 ? relByteOffset : getByteSize() + relByteOffset;
    return std::make_shared<Image>(this, desc, byteOffset);
  }

} // namespace oidn
