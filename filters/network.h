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

#include "common/tensor.h"
#include "buffer_view.h"
#include "node.h"

#pragma once

namespace oidn {

  template<int K>
  class Network : public Node
  {
  public:
    Network(const std::map<std::string, Tensor>& weight_map);
    void execute() override;

    std::shared_ptr<memory> alloc_tensor(const memory::dims& dims,
                                         memory::format format = memory::format::any,
                                         void* data = nullptr);

    std::shared_ptr<memory> cast_tensor(const memory::dims& dims,
                                        const std::shared_ptr<memory>& src,
                                        size_t src_offset = 0);

    std::shared_ptr<memory> cast_tensor(const memory::dims& dims,
                                        const std::shared_ptr<memory>& src,
                                        const memory::dims& src_offset);

    void zero_tensor(const std::shared_ptr<memory>& dst);

    void weights_reorder(const std::shared_ptr<memory>& user_weights,
                         const std::shared_ptr<memory>& weights);

    memory::dims input_reorder_dims(const memory::dims& src_tz, int spatial_pad);
    std::shared_ptr<Node> add_input_reorder(const BufferView2D& src,
                                            const BufferView2D& src_albedo,
                                            const BufferView2D& src_normal,
                                            int spatial_pad,
                                            const std::shared_ptr<memory>& user_dst = nullptr);

    std::shared_ptr<Node> add_output_reorder(const std::shared_ptr<memory>& src,
                                             const BufferView2D& dst);

    memory::dims conv_dims(const std::string& name, const memory::dims& src_tz);
    std::shared_ptr<Node> add_conv(const std::string& name,
                                   const std::shared_ptr<memory>& src,
                                   bool relu = true);

    memory::dims pool_dims(const memory::dims& src_tz);
    std::shared_ptr<Node> add_pool(const std::shared_ptr<memory>& src,
                                   const std::shared_ptr<memory>& user_dst = nullptr);

    memory::dims unpool_dims(const memory::dims& src_tz);
    std::shared_ptr<Node> add_unpool(const std::shared_ptr<memory>& src,
                                     const std::shared_ptr<memory>& user_dst = nullptr);

    memory::dims concat_dims(const memory::dims& src1_tz, const memory::dims& src2_tz);

  private:
    engine cpu_engine;
    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, Tensor> weight_map;
  };

} // ::oidn
