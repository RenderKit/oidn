// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>

#include "scratch.h"
#include "input_reorder.h"
#include "output_reorder.h"

class MPSGraphNetworkImpl {
public:
    MPSGraphNetworkImpl();
    ~MPSGraphNetworkImpl();

public:
    void init();
    
    bool createGraph();
    
    bool execute(void* data, void* dst);
    
    bool addInput(std::vector<int64_t> shape, std::string name);
    
    bool addConv(std::string src,
                 int groups,
                 std::vector<int64_t> paddings,
                 std::vector<int64_t> stride,
                 std::vector<int64_t> dilation,
                 std::vector<int64_t> weightsShape,
                 void* weights,
                 std::string name);
    
    bool addAdd(std::string src,
                std::vector<int64_t> addShape,
                void* add,
                std::string name);
    
    bool addRelu(std::string src, std::string name);
    
    bool addPool(std::string src,
                 std::vector<int64_t> kernel,
                 std::vector<int64_t> paddings,
                 std::vector<int64_t> stride,
                 std::string name);
    
    bool addUpsample(std::string src, std::vector<int64_t> size, std::string name);
    
    bool addConcat(std::vector<std::string> src, std::string name);
    
    bool setOutputs(std::vector<std::string> outputs);
    
    std::vector<int64_t> getTensorShape(std::string name);

private:
    void* _self;
};
