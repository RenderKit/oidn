// Copyright 2018 Intel Corporation
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
    ~Filter();

    Device* getDevice() const { return device.get(); }

    virtual void setImage(const std::string& name, const Ref<Image>& image) = 0;
    virtual void unsetImage(const std::string& name) = 0;
    virtual void setData(const std::string& name, const Data& data) = 0;
    virtual void updateData(const std::string& name) = 0;
    virtual void unsetData(const std::string& name) = 0;
    virtual void setInt(const std::string& name, int value) = 0;
    virtual int getInt(const std::string& name) = 0;
    virtual void setFloat(const std::string& name, float value) = 0;
    virtual float getFloat(const std::string& name) = 0;

    void setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr);

    virtual void commit() = 0;
    virtual void execute(SyncMode sync = SyncMode::Blocking) = 0;

  protected:
    void setParam(int& dst, int src);
    void setParam(bool& dst, int src);
    void setParam(Quality& dst, Quality src);
    void setParam(Ref<Image>& dst, const Ref<Image>& src);
    void removeParam(Ref<Image>& dst);
    void setParam(Data& dst, const Data& src);
    void removeParam(Data& dst);

    Ref<Device> device;

    ProgressMonitorFunction progressFunc = nullptr;
    void* progressUserPtr = nullptr;

    bool dirty = true;
    bool dirtyParam = true;
  };

OIDN_NAMESPACE_END
