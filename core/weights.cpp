// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "weights.h"
#include "tza.h"

namespace oidn {

  Weights::Weights(const Ref<Device>& device, const Data& blob)
    : tensors(parseTZA(device, blob.ptr, blob.size)),
      device(device) {}

  const std::shared_ptr<Tensor>& Weights::get(const std::string& name)
  {
    std::shared_ptr<Tensor>& tensor = tensors[name];

    // If the tensor is not backed by a buffer, it has not been reordered yet.
    // Reordering is necessary even if the tensor has the right shape and layout
    // because it may not be USM allocated.
    if (!tensor->getBuffer())
    {
      switch (tensor->getRank())
      {
      case 1:
        tensor = reorder1D(tensor);
        break;
      case 4:
        tensor = reorder4D(tensor);
        break;
      default:
        throw Exception(Error::InvalidOperation, "invalid weight or bias tensor");
      }
    }

    return tensor;
  }

  std::shared_ptr<Tensor> Weights::reorder1D(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getRank() == 1);

    const int B = device->getTensorBlockSize();
    const int X = round_up(src->getX(), B);

    auto dst = device->newTensor({{X}, TensorLayout::x, device->getTensorDataType()});
    reorder(*src, *dst);
    return dst;
  }

  std::shared_ptr<Tensor> Weights::reorder4D(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getRank() == 4);

    const int B = device->getTensorBlockSize();
    const int O = round_up(src->getO(), B);
    const int I = round_up(src->getI(), B);
    const int H = src->getH();
    const int W = src->getW();

    auto dst = device->newTensor({{O, I, H, W}, device->getWeightsLayout(), device->getTensorDataType()});
    reorder(*src, *dst);
    return dst;
  }

} // namespace oidn