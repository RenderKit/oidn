// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "device.h"
#include "image.h"
#include "data.h"

namespace oidn {

  class Filter : public RefCount
  {
  protected:
    Ref<Device> device;

    ProgressMonitorFunction progressFunc = nullptr;
    void* progressUserPtr = nullptr;

    bool dirty = true;
    bool dirtyParam = true;

  public:
    explicit Filter(const Ref<Device>& device) : device(device) {}

    virtual void setImage(const std::string& name, const std::shared_ptr<Image>& image) = 0;
    virtual void removeImage(const std::string& name) = 0;
    virtual void setData(const std::string& name, const Data& data) = 0;
    virtual void updateData(const std::string& name) = 0;
    virtual void removeData(const std::string& name) = 0;
    virtual void set1i(const std::string& name, int value) = 0;
    virtual int get1i(const std::string& name) = 0;
    virtual void set1f(const std::string& name, float value) = 0;
    virtual float get1f(const std::string& name) = 0;

    void setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr);

    virtual void commit() = 0;
    virtual void execute(bool sync = true) = 0;

    Device* getDevice() { return device.get(); }

  protected:
    void setParam(int& dst, int src);
    void setParam(bool& dst, int src);
    void setParam(std::shared_ptr<Image>& dst, const std::shared_ptr<Image>& src);
    void removeParam(std::shared_ptr<Image>& dst);
    void setParam(Data& dst, const Data& src);
    void removeParam(Data& dst);
  };

} // namespace oidn
