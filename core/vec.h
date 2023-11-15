// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "math.h"

OIDN_NAMESPACE_BEGIN
namespace math {

  template<typename T>
  struct vec2
  {
    T x, y;

    oidn_host_device_inline vec2() {}
    oidn_host_device_inline vec2(T x) : x(x), y(x) {}
    oidn_host_device_inline vec2(T x, T y) : x(x), y(y) {}

    template<typename U>
    oidn_host_device_inline vec2(vec2<U> v) : x(v.x), y(v.y) {}
  };

  template<typename T>
  struct vec3
  {
    T x, y, z;

    oidn_host_device_inline vec3() {}
    oidn_host_device_inline vec3(T x) : x(x), y(x), z(x) {}
    oidn_host_device_inline vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    template<typename U>
    oidn_host_device_inline vec3(vec3<U> v) : x(v.x), y(v.y), z(v.z) {}
  };

  using vec2f = vec2<float>;
  using vec2i = vec2<int>;
  using vec3f = vec3<float>;
  using vec3i = vec3<int>;

  #define define_vec_binary_op(name, op)                         \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(vec2<T> a, vec2<T> b) { \
      return vec2<T>(a.x op b.x, a.y op b.y);                    \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(vec2<T> a, T b) {       \
      return vec2<T>(a.x op b, a.y op b);                        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec2<T> name(T a, vec2<T> b) {       \
      return vec2<T>(a op b.x, a op b.y);                        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(vec3<T> a, vec3<T> b) { \
      return vec3<T>(a.x op b.x, a.y op b.y, a.z op b.z);        \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(vec3<T> a, T b) {       \
      return vec3<T>(a.x op b, a.y op b, a.z op b);              \
    }                                                            \
    template<typename T>                                         \
    oidn_host_device_inline vec3<T> name(T a, vec3<T> b) {       \
      return vec3<T>(a op b.x, a op b.y, a op b.z);              \
    }

  define_vec_binary_op(operator+, +)
  define_vec_binary_op(operator-, -)
  define_vec_binary_op(operator*, *)
  define_vec_binary_op(operator/, /)

  #undef define_vec_binary_op

  #define define_vec_unary_func(f)                 \
    template<typename T>                           \
    oidn_host_device_inline vec2<T> f(vec2<T> v) { \
      return vec2<T>(f(v.x), f(v.y));              \
    }                                              \
    template<typename T>                           \
    oidn_host_device_inline vec3<T> f(vec3<T> v) { \
      return vec3<T>(f(v.x), f(v.y), f(v.z));      \
    }

  define_vec_unary_func(pow)
  define_vec_unary_func(powr)
  define_vec_unary_func(log)
  define_vec_unary_func(exp)
  define_vec_unary_func(nan_to_zero)

  #undef define_vec_unary_func

  #define define_vec_binary_func(f)                           \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(vec2<T> a, vec2<T> b) { \
      return vec2<T>(f(a.x, b.x), f(a.y, b.y));               \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(vec2<T> a, T b) {       \
      return vec2<T>(f(a.x, b), f(a.y, b));                   \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec2<T> f(T a, vec2<T> b) {       \
      return vec2<T>(f(a, b.x), f(a, b.y));                   \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(vec3<T> a, vec3<T> b) { \
      return vec3<T>(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z));  \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(vec3<T> a, T b) {       \
      return vec3<T>(f(a.x, b), f(a.y, b), f(a.z, b));        \
    }                                                         \
    template<typename T>                                      \
    oidn_host_device_inline vec3<T> f(T a, vec3<T> b) {       \
      return vec3<T>(f(a, b.x), f(a, b.y), f(a, b.z));        \
    }

  define_vec_binary_func(min)
  define_vec_binary_func(max)

  #undef define_vec_binary_func

  #define define_vec_reduce(f)                        \
    template<typename T>                              \
    oidn_host_device_inline T reduce_##f(vec2<T> v) { \
      return f(v.x, v.y);                             \
    }                                                 \
    template<typename T>                              \
    oidn_host_device_inline T reduce_##f(vec3<T> v) { \
      return f(f(v.x, v.y), v.z);                     \
    }

  define_vec_reduce(min)
  define_vec_reduce(max)

  #undef define_vec_reduce

  template<typename T>
  oidn_host_device_inline vec2<T> clamp(vec2<T> v, T minVal, T maxVal)
  {
    return vec2<T>(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal));
  }

  template<typename T>
  oidn_host_device_inline vec3<T> clamp(vec3<T> v, T minVal, T maxVal)
  {
    return vec3<T>(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal));
  }

} // namespace math

using math::vec2;
using math::vec2f;
using math::vec2i;
using math::vec3;
using math::vec3f;
using math::vec3i;

OIDN_NAMESPACE_END