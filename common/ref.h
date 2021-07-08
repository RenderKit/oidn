// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

namespace oidn {

  class RefCount
  {
  private:
    std::atomic<size_t> count;

  public:
    __forceinline RefCount(size_t count = 0) noexcept : count(count) {}
  
    __forceinline size_t incRef() noexcept
    {
      return count.fetch_add(1) + 1;
    }

    __forceinline size_t decRef()
    {
      const size_t newCount = decRefKeep();
      if (newCount == 0)
        destroy();
      return newCount;
    }

    __forceinline size_t decRefKeep() noexcept
    {
      return count.fetch_add(-1) - 1;
    }

    __forceinline void destroy()
    {
      delete this;
    }

  protected:
    // Disable copying
    RefCount(const RefCount&) = delete;
    RefCount& operator =(const RefCount&) = delete;

    virtual ~RefCount() noexcept = default;
  };

  template<typename T>
  class Ref
  {
  private:
    T* ptr;

  public:
    __forceinline Ref() noexcept : ptr(nullptr) {}
    __forceinline Ref(std::nullptr_t) noexcept : ptr(nullptr) {}
    __forceinline Ref(const Ref& other) noexcept : ptr(other.ptr) { if (ptr) ptr->incRef(); }
    __forceinline Ref(Ref&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    __forceinline Ref(T* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    template<typename Y>
    __forceinline Ref(const Ref<Y>& other) noexcept : ptr(other.get()) { if (ptr) ptr->incRef(); }

    template<typename Y>
    __forceinline explicit Ref(Y* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    __forceinline ~Ref() { if (ptr) ptr->decRef(); }

    __forceinline Ref& operator =(const Ref& other)
    {
      if (other.ptr)
        other.ptr->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other.ptr;
      return *this;
    }

    __forceinline Ref& operator =(Ref&& other)
    {
      if (ptr)
        ptr->decRef();
      ptr = other.ptr;
      other.ptr = nullptr;
      return *this;
    }

    __forceinline Ref& operator =(T* other)
    {
      if (other)
        other->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other;
      return *this;
    }

    __forceinline Ref& operator =(std::nullptr_t)
    {
      if (ptr)
        ptr->decRef();
      ptr = nullptr;
      return *this;
    }

    __forceinline operator bool() const noexcept { return ptr != nullptr; }

    __forceinline T& operator  *() const noexcept { return *ptr; }
    __forceinline T* operator ->() const noexcept { return  ptr; }

    __forceinline T* get() const noexcept { return ptr; }

    __forceinline T* detach() noexcept
    {
      T* res = ptr;
      ptr = nullptr;
      return res;
    }

    friend __forceinline bool operator < (const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   <  b.ptr;   }

    friend __forceinline bool operator ==(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   == nullptr; }
    friend __forceinline bool operator ==(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr == b.ptr;   }
    friend __forceinline bool operator ==(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   == b.ptr;   }

    friend __forceinline bool operator !=(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   != nullptr; }
    friend __forceinline bool operator !=(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr != b.ptr;   }
    friend __forceinline bool operator !=(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   != b.ptr;   }
  };

  template<typename T, typename... Args>
  __forceinline Ref<T> makeRef(Args&&... args)
  {
    return Ref<T>(new T(std::forward<Args>(args)...));
  }

  template<typename T, typename U>
  __forceinline Ref<T> staticRefCast(const Ref<U>& a)
  {
    return Ref<T>(static_cast<T*>(a.get()));
  }

  template<typename T, typename U>
  __forceinline Ref<T> dynamicRefCast(const Ref<U>& a)
  {
    return Ref<T>(dynamic_cast<T*>(a.get()));
  }

} // namespace oidn
