// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "image.h"
#include "data.h"

OIDN_NAMESPACE_BEGIN

  class Filter : public RefCount
  {
  public:
    explicit Filter(const Ref<Device>& device);

    Device* getDevice() const { return device.get(); }

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
    virtual void execute(SyncMode sync = SyncMode::Sync) = 0;

  protected:
    void setParam(int& dst, int src);
    void setParam(bool& dst, int src);
    void setParam(std::shared_ptr<Image>& dst, const std::shared_ptr<Image>& src);
    void removeParam(std::shared_ptr<Image>& dst);
    void setParam(Data& dst, const Data& src);
    void removeParam(Data& dst);

    Ref<Device> device;

    ProgressMonitorFunction progressFunc = nullptr;
    void* progressUserPtr = nullptr;

    bool dirty = true;
    bool dirtyParam = true;
  };

OIDN_NAMESPACE_END
