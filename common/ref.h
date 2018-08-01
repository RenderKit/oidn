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

#pragma once

#include "platform.h"

namespace oidn {

  class RefCount
  {
  private:
    std::atomic<size_t> count;

  public:
    RefCount(int count = 0) : count(count) {}
  
    void inc_ref()
    {
      count.fetch_add(1);
    }

    void dec_ref()
    {
      if (count.fetch_add(-1) == 1)
        delete this;
    }

  protected:
    virtual ~RefCount() = default;
  };

  template<typename T>
  class Ref
  {
  private:
    T* ptr;

  public:
    __forceinline Ref() : ptr(nullptr) {}
    __forceinline Ref(std::nullptr_t) : ptr(nullptr) {}
    __forceinline Ref(const Ref& other) : ptr(other.ptr) { if (ptr) ptr->inc_ref(); }
    __forceinline Ref(Ref&& other) : ptr(other.ptr) { other.ptr = nullptr; }
    __forceinline Ref(T* ptr) : ptr(ptr) { if (ptr) ptr->inc_ref(); }

    template<typename Y>
    __forceinline Ref(const Ref<Y>& other) : ptr(other.get()) { if (ptr) ptr->inc_ref(); }

    template<typename Y>
    __forceinline explicit Ref(Y* ptr) : ptr(ptr) { if (ptr) ptr->inc_ref(); }

    __forceinline ~Ref() { if (ptr) ptr->dec_ref(); }

    __forceinline Ref& operator =(const Ref& other)
    {
      if (other.ptr)
        other.ptr->inc_ref();
      if (ptr)
        ptr->dec_ref();
      ptr = other.ptr;
      return *this;
    }

    __forceinline Ref& operator =(Ref&& other)
    {
      if (ptr)
        ptr->dec_ref();
      ptr = other.ptr;
      other.ptr = nullptr;
      return *this;
    }

    __forceinline Ref& operator =(T* other)
    {
      if (other)
        other->inc_ref();
      if (ptr)
        ptr->dec_ref();
      ptr = other;
      return *this;
    }

    __forceinline Ref& operator =(std::nullptr_t)
    {
      if (ptr)
        ptr->dec_ref();
      ptr = nullptr;
      return *this;
    }

    __forceinline operator bool() const { return ptr != nullptr; }

    __forceinline T& operator  *() const { return *ptr; }
    __forceinline T* operator ->() const { return  ptr; }

    __forceinline T* get() const { return ptr; }

    __forceinline T* detach()
    {
      T* res = ptr;
      ptr = nullptr;
      return res;
    }
  };

  template<typename T> __forceinline bool operator < (const Ref<T>& a, const Ref<T>& b) { return a.ptr   <  b.ptr;   }

  template<typename T> __forceinline bool operator ==(const Ref<T>& a, std::nullptr_t)  { return a.ptr   == nullptr; }
  template<typename T> __forceinline bool operator ==(std::nullptr_t,  const Ref<T>& b) { return nullptr == b.ptr;   }
  template<typename T> __forceinline bool operator ==(const Ref<T>& a, const Ref<T>& b) { return a.ptr   == b.ptr;   }

  template<typename T> __forceinline bool operator !=(const Ref<T>& a, std::nullptr_t)  { return a.ptr   != nullptr; }
  template<typename T> __forceinline bool operator !=(std::nullptr_t,  const Ref<T>& b) { return nullptr != b.ptr;   }
  template<typename T> __forceinline bool operator !=(const Ref<T>& a, const Ref<T>& b) { return a.ptr   != b.ptr;   }

  template<typename T, typename... Args>
  __forceinline Ref<T> make_ref(Args&&... args)
  {
    return Ref<T>(new T(std::forward<Args>(args)...));
  }

  template<typename T, typename Y>
  __forceinline Ref<Y> static_pointer_cast(const Ref<T>& a)
  {
    return Ref<Y>(static_cast<Y*>(a.get()));
  }

  template<typename T, typename Y>
  __forceinline Ref<Y> dynamic_pointer_cast(const Ref<T>& a)
  {
    return Ref<Y>(dynamic_cast<Y*>(a.get()));
  }

} // ::oidn
