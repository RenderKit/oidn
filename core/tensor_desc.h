// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_layout.h"
#include <vector>

OIDN_NAMESPACE_BEGIN

  // Tensor dimensions
  // Canonical order: CHW / OIHW
  using TensorDims = std::vector<int>;

  std::ostream& operator <<(std::ostream& sm, const TensorDims& dims);

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;                             // logical dimensions
    TensorDims   paddedDims;                       // storage dimensions with zero-padding
    TensorLayout layout = TensorLayout::Undefined; // storage layout
    DataType     dataType = DataType::Undefined;   // element data type

    TensorDesc() = default;

    TensorDesc(TensorDims dims, TensorDims paddedDims, TensorLayout layout, DataType dataType)
      : dims(dims), paddedDims(paddedDims), layout(layout), dataType(dataType)
    {
      assert(isValid());
    }

    TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), paddedDims(dims), layout(layout), dataType(dataType)
    {
      assert(isValid());
    }

    bool isValid() const
    {
      const auto info = getTensorLayoutInfo(layout);

      return getRank() == info.rank &&
             dims.size() == paddedDims.size() &&
             std::mismatch(dims.begin(), dims.end(), paddedDims.begin(),
                           [](int a, int b) { return a <= b; }).first == dims.end() &&
             (info.blockC == 1 ||
               (getRank() == 3 && getPaddedC() % info.blockC == 0) ||
               (getRank() == 4 && getPaddedO() % info.blockC == 0 && getPaddedI() % info.blockC == 0));
    }

    // Returns the number of dimensions
    oidn_inline int getRank() const { return int(dims.size()); }

    // Returns the number of elements in a 1D tensor
    oidn_inline int getX() const
    {
      assert(dims.size() == 1);
      return dims[0];
    }

    oidn_inline int getPaddedX() const
    {
      assert(paddedDims.size() == 1);
      return paddedDims[0];
    }

    // Returns the number of output channels in the tensor
    oidn_inline int getO() const
    {
      assert(dims.size() >= 4);
      return dims[dims.size()-4];
    }

    oidn_inline int getPaddedO() const
    {
      assert(paddedDims.size() >= 4);
      return paddedDims[paddedDims.size()-4];
    }

    // Returns the number of input channels in the tensor
    oidn_inline int getI() const
    {
      assert(dims.size() >= 3);
      return dims[dims.size()-3];
    }

    oidn_inline int getPaddedI() const
    {
      assert(paddedDims.size() >= 3);
      return paddedDims[paddedDims.size()-3];
    }

    // Returns the number of channels in the tensor
    oidn_inline int getC() const
    {
      assert(dims.size() >= 3);
      return dims[dims.size()-3];
    }

    oidn_inline int getPaddedC() const
    {
      assert(paddedDims.size() >= 3);
      return paddedDims[paddedDims.size()-3];
    }

    // Returns the height of the tensor
    oidn_inline int getH() const
    {
      assert(dims.size() >= 2);
      return dims[dims.size()-2];
    }

    // Returns the width of the tensor
    oidn_inline int getW() const
    {
      assert(dims.size() >= 2);
      return dims[dims.size()-1];
    }

    // Returns the number of elements in the tensor
    oidn_inline size_t getNumElements() const
    {
      if (dims.empty())
        return 0;
      size_t num = 1;
      for (size_t i = 0; i < dims.size(); ++i)
        num *= size_t(dims[i]);
      return num;
    }

    // Returns the size in bytes of the tensor
    oidn_inline size_t getByteSize() const
    {
      if (paddedDims.empty())
        return 0;
      size_t num = 1;
      for (size_t i = 0; i < paddedDims.size(); ++i)
        num *= size_t(paddedDims[i]);
      return num * getDataTypeSize(dataType);
    }

    bool operator ==(const TensorDesc& other) const
    {
      return (dims == other.dims) && (paddedDims == other.paddedDims) &&
             (layout == other.layout) && (dataType == other.dataType);
    }

    bool operator !=(const TensorDesc& other) const
    {
      return (dims != other.dims) || (paddedDims != other.paddedDims) ||
             (layout != other.layout) || (dataType != other.dataType);
    }
  };

OIDN_NAMESPACE_END