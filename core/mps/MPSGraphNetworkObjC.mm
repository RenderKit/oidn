// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#import "MPSGraphNetworkObjC.hpp"
#import <Accelerate/Accelerate.h>

NSMutableArray* vectorToNSArrayNSNumber(std::vector<int64_t> vec) {
  id arr = [NSMutableArray new];
  std::for_each(vec.begin(), vec.end(), ^(int64_t val) {
    NSNumber* nsNum = [NSNumber numberWithLong:val];
    [arr addObject:nsNum];
  });
  
  return arr;
}

NSString* stringToNSString(std::string val) {
  return [NSString stringWithCString:val.c_str()
                            encoding:[NSString defaultCStringEncoding]];
}

@implementation MPSGraphNetworkObjC

MPSGraphNetworkImpl::MPSGraphNetworkImpl() : _self(nullptr) {
}

MPSGraphNetworkImpl::~MPSGraphNetworkImpl() {
  [(id)_self dealloc];
}

void MPSGraphNetworkImpl::init() {
  if (@available(macOS 11.0, *)) {
    _self = [[MPSGraphNetworkObjC alloc] init];
  } else {
    // Fallback on earlier versions
  }
}

bool MPSGraphNetworkImpl::createGraph() {
  return [(id)_self createGraph];
}

bool MPSGraphNetworkImpl::execute(void* data, void* dst) {
  return [(id)_self executeWidthData:data writeTo:dst];
}

bool MPSGraphNetworkImpl::addInput(std::vector<int64_t> shape, std::string name) {
  id nsShape = vectorToNSArrayNSNumber(shape);
  NSString* nsName = stringToNSString(name);
  
  return [(id)_self addInput:nsShape withName: nsName];
}

bool MPSGraphNetworkImpl::addConv(std::string src,
                                  int groups,
                                  std::vector<int64_t> paddings,
                                  std::vector<int64_t> stride,
                                  std::vector<int64_t> dilation,
                                  std::vector<int64_t> weightsShape,
                                  void* weights,
                                  std::string name) {
  
  id nsPaddings = vectorToNSArrayNSNumber(paddings);
  id nsStrides = vectorToNSArrayNSNumber(stride);
  id nsDilation = vectorToNSArrayNSNumber(dilation);
  NSInteger nsGroups = (NSInteger)groups;
  
  NSString* nsSrc = stringToNSString(src);
  NSString* nsName = stringToNSString(name);
  
  id nsWeightsShape = vectorToNSArrayNSNumber(weightsShape);
  
  int weightsSize = 1;
  
  for (int i = 0; i < weightsShape.size(); i++) {
    weightsSize *= weightsShape[i];
  }
  
  weightsSize *= sizeof(float);
  
  NSData* nsWeights = [NSData dataWithBytes: weights
                                     length: weightsSize];
  
  return [(id)_self addConv:nsSrc groups: nsGroups paddings:nsPaddings
                    strides:nsStrides
                   dilation:nsDilation
               weightsShape:nsWeightsShape weights:nsWeights
                   withName: nsName];
}

bool MPSGraphNetworkImpl::addAdd(std::string src,
                                 std::vector<int64_t> addShape,
                                 void* add,
                                 std::string name) {
  
  NSString* nsSrc = stringToNSString(src);
  NSString* nsName = stringToNSString(name);
  
  id nsAddShape = vectorToNSArrayNSNumber(addShape);
  
  int addSize = 1;
  
  for (int i = 0; i < addShape.size(); i++) {
    addSize *= addShape[i];
  }
  
  addSize *= sizeof(float);
  
  NSData* nsAdd = [NSData dataWithBytes: add
                                 length: addSize];
  
  return [(id)_self addAdd:nsSrc
                  addShape:nsAddShape add:nsAdd
                  withName: nsName];
}

bool MPSGraphNetworkImpl::addRelu(std::string src,
                                  std::string name) {
  
  NSString* nsSrc = stringToNSString(src);
  NSString* nsName = stringToNSString(name);
  
  return [(id)_self addRelu:nsSrc withName: nsName];
}

bool MPSGraphNetworkImpl::addPool(std::string src,
                                  std::vector<int64_t> kernel,
                                  std::vector<int64_t> paddings,
                                  std::vector<int64_t> stride,
                                  std::string name) {
  
  NSString* nsSrc = stringToNSString(src);
  NSString* nsName = stringToNSString(name);
  
  id nsPaddings = vectorToNSArrayNSNumber(paddings);
  id nsStrides = vectorToNSArrayNSNumber(stride);
  id nsKernel = vectorToNSArrayNSNumber(kernel);
  
  return [(id)_self addPool:nsSrc kernel:nsKernel paddings:nsPaddings strides:nsStrides withName:nsName];
}

bool MPSGraphNetworkImpl::addUpsample(std::string src, std::vector<int64_t> size, std::string name) {
  NSString* nsSrc = stringToNSString(src);
  NSString* nsName = stringToNSString(name);
  
  id nsSize = vectorToNSArrayNSNumber(size);
  
  return [(id)_self addUpsample:nsSrc size:nsSize withName:nsName];
}

bool MPSGraphNetworkImpl::addConcat(std::vector<std::string> src, std::string name) {
  id nsSrcs = [NSMutableArray new];
  std::for_each(src.begin(), src.end(), ^(std::string val) {
    [nsSrcs addObject:stringToNSString(val)];
  });
  
  NSString* nsName = stringToNSString(name);
  
  return [(id)_self addConcat:nsSrcs withName:nsName];
}

bool MPSGraphNetworkImpl::setOutputs(std::vector<std::string> outputs) {
  id nsOutputs = [NSMutableArray new];
  std::for_each(outputs.begin(), outputs.end(), ^(std::string val) {
    [nsOutputs addObject:stringToNSString(val)];
  });
  
  return [(id)_self setOutputs:nsOutputs];
}

std::vector<int64_t> MPSGraphNetworkImpl::getTensorShape(std::string name) {
  std::vector<int64_t> shape;
  NSString* nsName = stringToNSString(name);
  NSArray* nsShape = [(id)_self getTensorShape:nsName];
  
  for (id i in nsShape) {
    shape.push_back([i integerValue]);
  }
  return shape;
}

- (id) init {
  self = [super init];
  
  return self;
}

- (bool) createGraph {
  [self cleanup];
  
  _graph = [[MPSGraph alloc] init];

  NSArray* devices = [MTLCopyAllDevices() autorelease];
  for (unsigned long i = 0 ; i < [devices count] ; i++) {
    id<MTLDevice>  device = devices[i];
    if(![device isLowPower]) { // exclude Intel GPUs
      _device = [device retain];
      break;
    }
  }
  
  _inputs = [NSMutableArray new];
  _outputs = [NSMutableArray new];
  
  _tensors = [NSMutableDictionary dictionary];
  _tensorData = [NSMutableDictionary dictionary];
  
  return true;
}

- (bool) cleanup {
  _device = nil;
  
  _graph = nil;
  
  _inputs = nil;
  _outputs = nil;
  
  _tensors = nil;
  _tensorData = nil;
  
  _inputs = nil;
  _outputs = nil;
  
  _commandQueue = nil;
  
  return true;
}

- (bool) executeWidthData: (void*) data writeTo: (void*) dst {
  MPSGraphTensorData* tensorData = _tensorData[_inputs[0]];
  MPSNDArray* inputData = [tensorData mpsndarray];
  
  [inputData writeBytes:data strideBytes:nil];
  
  @autoreleasepool {
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] initWithCapacity: [_inputs count]] autorelease];
    
    for (id val in _inputs) {
      [feeds setObject:_tensorData[val] forKey:_tensors[val]];
    }
    
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [[[NSMutableDictionary alloc] initWithCapacity: [_outputs count]] autorelease];
    
    for (id val in _outputs) {
      [results setObject:_tensorData[val] forKey:_tensors[val]];
    }
    
    if (_commandQueue == nil) {
      _commandQueue = [_device newCommandQueue];
    }
    
    [_graph runWithMTLCommandQueue:_commandQueue
                             feeds:feeds
                  targetOperations:nil
                 resultsDictionary:results];
    
    MPSGraphTensorData* output = [[results allValues] objectAtIndex:0];
    
    [[output mpsndarray] readBytes:dst strideBytes:nil];
  }
  
  return true;
}

- (MPSGraphTensorData*) tensorDataFromTensor: (MPSGraphTensor*) tensor {
  MPSNDArrayDescriptor* descr = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32 shape:tensor.shape];
  MPSNDArray* data = [[MPSNDArray alloc] initWithDevice:_device descriptor:descr];
  
  return [[MPSGraphTensorData alloc] initWithMPSNDArray:data];
}

- (bool) addInput: (NSArray*) shape withName: (NSString*) name {
  id tensor = [_graph placeholderWithShape:shape dataType: MPSDataType::MPSDataTypeFloat32 name: name];
  [_tensors setObject:tensor forKey:name];
  [_inputs addObject:name];
  
  int bufferSize = sizeof(float);
  for (id i in shape) {
    bufferSize *= [i integerValue];
  }
  
  _inputBuffer = [_device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
  
  MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:_inputBuffer
                                                                           shape:shape
                                                                        dataType:MPSDataType::MPSDataTypeFloat32];
  
  [_tensorData setObject:tensorData forKey:name];
  
  return true;
}

- (bool) addConv: (NSString*) src groups: (NSInteger) groups
        paddings: (NSArray*) paddings
         strides: (NSArray*) strides
        dilation: (NSArray*) dilation
    weightsShape: (NSArray*) weightsShape
         weights: (NSData*) weights
        withName: (NSString*) name {
  
  MPSGraphConvolution2DOpDescriptor* descr = [MPSGraphConvolution2DOpDescriptor
                                              descriptorWithStrideInX: [strides[0] integerValue]
                                              strideInY:[strides[1] integerValue]
                                              dilationRateInX:[dilation[0] integerValue]
                                              dilationRateInY:[dilation[1] integerValue]
                                              
                                              groups:groups
                                              
                                              paddingLeft:[paddings[0] integerValue]
                                              paddingRight:[paddings[0] integerValue]
                                              paddingTop:[paddings[1] integerValue]
                                              paddingBottom:[paddings[1] integerValue]
                                              
                                              paddingStyle:MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                                              dataLayout:MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNCHW
                                              weightsLayout:MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutOIHW];
  
  id weightsTensor = [_graph constantWithData:weights shape:weightsShape dataType:MPSDataType::MPSDataTypeFloat32];
  
  id srcTensor = [_tensors objectForKey:src];
  id tensor = [_graph convolution2DWithSourceTensor:srcTensor weightsTensor:weightsTensor descriptor:descr name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) addAdd: (NSString*) src
       addShape: (NSArray*) addShape
            add: (NSData*) add
       withName: (NSString*) name {
  
  id constTensor = [_graph constantWithData:add shape:addShape dataType:MPSDataType::MPSDataTypeFloat32];
  
  id srcTensor = [_tensors objectForKey:src];
  id tensor = [_graph additionWithPrimaryTensor:srcTensor secondaryTensor:constTensor name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) addRelu: (NSString*) src
        withName: (NSString*) name {
  
  id srcTensor = [_tensors objectForKey:src];
  id tensor = [_graph reLUWithTensor:srcTensor name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) addPool: (NSString*) src
          kernel: (NSArray*) kernel
        paddings: (NSArray*) paddings
         strides: (NSArray*) strides
        withName: (NSString*) name {
  
  MPSGraphPooling2DOpDescriptor* descr = [[MPSGraphPooling2DOpDescriptor alloc] init];
  
  descr.strideInX = [strides[0] integerValue];
  descr.strideInY = [strides[1] integerValue];
  
  descr.paddingLeft = [paddings[0] integerValue];
  descr.paddingRight = [paddings[0] integerValue];
  descr.paddingTop = [paddings[1] integerValue];
  descr.paddingBottom = [paddings[1] integerValue];
  descr.paddingStyle = MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME;
  
  descr.kernelWidth = [kernel[0] integerValue];
  descr.kernelHeight = [kernel[1] integerValue];
  
  descr.dilationRateInX = 1;
  descr.dilationRateInY = 1;
  
  descr.dataLayout = MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNCHW;
  
  id srcTensor = [_tensors objectForKey:src];
  id tensor = [_graph maxPooling2DWithSourceTensor:srcTensor descriptor:descr name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) addUpsample: (NSString*) src
                size: (NSArray*) size
            withName: (NSString*) name {
  
  id srcTensor = [_tensors objectForKey:src];
  
  id tensor = [_graph resizeTensor:srcTensor size:size
                              mode:MPSGraphResizeMode::MPSGraphResizeNearest
                      centerResult:true alignCorners:false
                            layout:MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNCHW
                              name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) addConcat: (NSArray*) src
          withName: (NSString*) name {
  id srcTensors = [NSMutableArray new];
  
  for (id val in src) {
    [srcTensors addObject:_tensors[val]];
  }
  
  id tensor = [_graph concatTensors:srcTensors dimension:1 name:name];
  
  [_tensors setObject:tensor forKey:name];
  
  return true;
}

- (bool) setOutputs:(NSArray*)outputs {
  [_outputs addObjectsFromArray:outputs];
  
  for (id val in _outputs) {
    [_tensorData setObject:[self tensorDataFromTensor: _tensors[val]] forKey:val];
  }
  
  return true;
}

- (NSArray*) getTensorShape:(NSString*) name {
  id shape = [NSMutableArray new];
  id tensor = [_tensors objectForKey:name];
  [shape addObjectsFromArray: [tensor shape]];
  return shape;
}

@end
