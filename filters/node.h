// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#pragma once

#include "common.h"
#include <vector>

namespace oidn {

  class Node
  {
  public:
    virtual ~Node() = default;
    virtual void execute() = 0;
    virtual std::shared_ptr<memory> getDst() const { return nullptr; }
  };

  // Node wrapping an MKL-DNN primitive
  class MklNode : public Node
  {
  private:
    std::vector<primitive> net;

  public:
    MklNode(const primitive& prim)
    {
      net.push_back(prim);
    }

    void execute() override
    {
      stream(stream::kind::eager).submit(net).wait();
    }
  };

  // Convolution node
  class Conv : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> weights;
    std::shared_ptr<memory> bias;
    std::shared_ptr<memory> dst;

  public:
    Conv(const convolution_forward::primitive_desc& desc,
         const std::shared_ptr<memory>& src,
         const std::shared_ptr<memory>& weights,
         const std::shared_ptr<memory>& bias,
         const std::shared_ptr<memory>& dst)
      : MklNode(convolution_forward(desc, *src, *weights, *bias, *dst)),
        src(src), weights(weights), bias(bias), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

  // Pooling node
  class Pool : public MklNode
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    Pool(const pooling_forward::primitive_desc& desc,
         const std::shared_ptr<memory>& src,
         const std::shared_ptr<memory>& dst)
      : MklNode(pooling_forward(desc, *src, *dst)),
        src(src), dst(dst) {}

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
