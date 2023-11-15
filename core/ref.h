// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN

  class RefCount
  {
  public:
    oidn_inline RefCount(size_t count = 0) noexcept : count(count) {}

    oidn_inline size_t incRef() noexcept
    {
      return ++count;
    }

    oidn_inline size_t decRef()
    {
      const size_t newCount = decRefKeep();
      if (newCount == 0)
        destroy();
      return newCount;
    }

    oidn_inline size_t decRefKeep() noexcept
    {
      return --count;
    }

    oidn_inline void destroy()
    {
      delete this;
    }

  protected:
    // Disable copying
    RefCount(const RefCount&) = delete;
    RefCount& operator =(const RefCount&) = delete;

    virtual ~RefCount() noexcept = default;

  private:
    std::atomic<size_t> count;
  };

  template<typename T>
  class Ref
  {
    template<typename Y>
    using Convertible = typename std::enable_if<std::is_convertible<Y*, T*>::value>::type;

  public:
    oidn_inline Ref() noexcept : ptr(nullptr) {}
    oidn_inline Ref(std::nullptr_t) noexcept : ptr(nullptr) {}
    oidn_inline Ref(const Ref& other) noexcept : ptr(other.ptr) { if (ptr) ptr->incRef(); }
    oidn_inline Ref(Ref&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    oidn_inline Ref(T* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    template<typename Y, typename = Convertible<Y>>
    oidn_inline Ref(const Ref<Y>& other) noexcept : ptr(other.get()) { if (ptr) ptr->incRef(); }

    template<typename Y, typename = Convertible<Y>>
    oidn_inline explicit Ref(Y* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    oidn_inline ~Ref() { if (ptr) ptr->decRef(); }

    oidn_inline Ref& operator =(const Ref& other)
    {
      if (other.ptr)
        other.ptr->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other.ptr;
      return *this;
    }

    oidn_inline Ref& operator =(Ref&& other)
    {
      T* otherPtr = other.ptr;
      other.ptr = nullptr;
      if (ptr)
        ptr->decRef();
      ptr = otherPtr;
      return *this;
    }

    oidn_inline Ref& operator =(T* other)
    {
      if (other)
        other->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other;
      return *this;
    }

    oidn_inline Ref& operator =(std::nullptr_t)
    {
      if (ptr)
        ptr->decRef();
      ptr = nullptr;
      return *this;
    }

    oidn_inline operator bool() const noexcept { return ptr != nullptr; }

    oidn_inline T& operator  *() const noexcept { return *ptr; }
    oidn_inline T* operator ->() const noexcept { return  ptr; }

    oidn_inline T* get() const noexcept { return ptr; }

    oidn_inline void reset()
    {
      if (ptr)
        ptr->decRef();
      ptr = nullptr;
    }

    oidn_inline T* detach() noexcept
    {
      T* res = ptr;
      ptr = nullptr;
      return res;
    }

    friend oidn_inline bool operator < (const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   <  b.ptr;   }

    friend oidn_inline bool operator ==(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   == nullptr; }
    friend oidn_inline bool operator ==(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr == b.ptr;   }
    friend oidn_inline bool operator ==(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   == b.ptr;   }

    friend oidn_inline bool operator !=(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   != nullptr; }
    friend oidn_inline bool operator !=(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr != b.ptr;   }
    friend oidn_inline bool operator !=(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   != b.ptr;   }

  private:
    T* ptr;
  };

  template<typename T, typename... Args>
  oidn_inline Ref<T> makeRef(Args&&... args)
  {
    return Ref<T>(new T(std::forward<Args>(args)...));
  }

  template<typename T, typename U>
  oidn_inline Ref<T> staticRefCast(const Ref<U>& a)
  {
    return Ref<T>(static_cast<T*>(a.get()));
  }

  template<typename T, typename U>
  oidn_inline Ref<T> dynamicRefCast(const Ref<U>& a)
  {
    return Ref<T>(dynamic_cast<T*>(a.get()));
  }

OIDN_NAMESPACE_END
