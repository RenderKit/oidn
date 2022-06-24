// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#import "mps/MPSGraphNetwork-C-Interface.h"

#import <Cocoa/Cocoa.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

API_AVAILABLE(macos(11.0))
@interface MPSGraphNetworkObjC : NSObject {
    MPSGraph* _graph;
    
    id<MTLCommandQueue> _commandQueue;
    
    id<MTLDevice> _device;
    
    NSMutableArray<NSString*>* _inputs;
    NSMutableArray<NSString*>* _outputs;
    
    NSMutableDictionary<NSString*, id>* _tensors;
    
    NSMutableDictionary<NSString*, MPSGraphTensorData*>* _tensorData;
    
    id<MTLBuffer> _inputBuffer;
}

- (id) init;

- (bool) createGraph;

- (bool) cleanup;

- (bool) executeWidthData: (void*) data
                  writeTo: (void*) dst;

- (MPSGraphTensorData*) tensorDataFromTensor: (MPSGraphTensor*) tensor;

- (bool) addInput: (NSArray*) shape
         withName: (NSString*) name;

- (bool) addConv: (NSString*) src
         groups: (NSInteger) groups
         paddings: (NSArray*) paddings
         strides: (NSArray*) strides
         dilation: (NSArray*) dilation
         weightsShape: (NSArray*) weightsShape
         weights: (NSData*) weights
         withName: (NSString*) name;

- (bool) addAdd: (NSString*) src
         addShape: (NSArray*) addShape
         add: (NSData*) add
         withName: (NSString*) name;

- (bool) addRelu: (NSString*) src
         withName: (NSString*) name;

- (bool) addPool: (NSString*) src
         kernel: (NSArray*) kernel
         paddings: (NSArray*) paddings
         strides: (NSArray*) strides
         withName: (NSString*) name;

- (bool) addUpsample: (NSString*) src
         size: (NSArray*) size
         withName: (NSString*) name;

- (bool) addConcat: (NSArray*) src
         withName: (NSString*) name;

- (bool) setOutputs: (NSArray*) outputs;

- (NSArray*) getTensorShape: (NSString*) name;

@end
