// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  class RefCount
  {
  public:
    OIDN_INLINE RefCount(size_t count = 0) noexcept : count(count) {}
  
    OIDN_INLINE size_t incRef() noexcept
    {
      return ++count;
    }

    OIDN_INLINE size_t decRef()
    {
      const size_t newCount = decRefKeep();
      if (newCount == 0)
        destroy();
      return newCount;
    }

    OIDN_INLINE size_t decRefKeep() noexcept
    {
      return --count;
    }

    OIDN_INLINE void destroy()
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
    OIDN_INLINE Ref() noexcept : ptr(nullptr) {}
    OIDN_INLINE Ref(std::nullptr_t) noexcept : ptr(nullptr) {}
    OIDN_INLINE Ref(const Ref& other) noexcept : ptr(other.ptr) { if (ptr) ptr->incRef(); }
    OIDN_INLINE Ref(Ref&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    OIDN_INLINE Ref(T* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    template<typename Y, typename = Convertible<Y>>
    OIDN_INLINE Ref(const Ref<Y>& other) noexcept : ptr(other.get()) { if (ptr) ptr->incRef(); }

    template<typename Y, typename = Convertible<Y>>
    OIDN_INLINE explicit Ref(Y* ptr) noexcept : ptr(ptr) { if (ptr) ptr->incRef(); }

    OIDN_INLINE ~Ref() { if (ptr) ptr->decRef(); }

    OIDN_INLINE Ref& operator =(const Ref& other)
    {
      if (other.ptr)
        other.ptr->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other.ptr;
      return *this;
    }

    OIDN_INLINE Ref& operator =(Ref&& other)
    {
      if (ptr)
        ptr->decRef();
      ptr = other.ptr;
      other.ptr = nullptr;
      return *this;
    }

    OIDN_INLINE Ref& operator =(T* other)
    {
      if (other)
        other->incRef();
      if (ptr)
        ptr->decRef();
      ptr = other;
      return *this;
    }

    OIDN_INLINE Ref& operator =(std::nullptr_t)
    {
      if (ptr)
        ptr->decRef();
      ptr = nullptr;
      return *this;
    }

    OIDN_INLINE operator bool() const noexcept { return ptr != nullptr; }

    OIDN_INLINE T& operator  *() const noexcept { return *ptr; }
    OIDN_INLINE T* operator ->() const noexcept { return  ptr; }

    OIDN_INLINE T* get() const noexcept { return ptr; }

    OIDN_INLINE T* detach() noexcept
    {
      T* res = ptr;
      ptr = nullptr;
      return res;
    }

    friend OIDN_INLINE bool operator < (const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   <  b.ptr;   }

    friend OIDN_INLINE bool operator ==(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   == nullptr; }
    friend OIDN_INLINE bool operator ==(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr == b.ptr;   }
    friend OIDN_INLINE bool operator ==(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   == b.ptr;   }

    friend OIDN_INLINE bool operator !=(const Ref<T>& a, std::nullptr_t)  noexcept { return a.ptr   != nullptr; }
    friend OIDN_INLINE bool operator !=(std::nullptr_t,  const Ref<T>& b) noexcept { return nullptr != b.ptr;   }
    friend OIDN_INLINE bool operator !=(const Ref<T>& a, const Ref<T>& b) noexcept { return a.ptr   != b.ptr;   }

  private:
    T* ptr;
  };

  template<typename T, typename... Args>
  OIDN_INLINE Ref<T> makeRef(Args&&... args)
  {
    return Ref<T>(new T(std::forward<Args>(args)...));
  }

  template<typename T, typename U>
  OIDN_INLINE Ref<T> staticRefCast(const Ref<U>& a)
  {
    return Ref<T>(static_cast<T*>(a.get()));
  }

  template<typename T, typename U>
  OIDN_INLINE Ref<T> dynamicRefCast(const Ref<U>& a)
  {
    return Ref<T>(dynamic_cast<T*>(a.get()));
  }

} // namespace oidn
