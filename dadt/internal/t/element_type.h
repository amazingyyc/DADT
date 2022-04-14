#pragma once

#include <mpi.h>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

// define the half
#ifndef HAVE_NCCL
#pragma pack(1)
struct half {
  uint16_t value;
};
static_assert(2 == sizeof(half));
#pragma pack()
#endif

namespace dadt {

// define a unknow type
struct UnKnown {};

enum struct DType : uint8_t {
  kUnKnown = 0,
  kBool = 1,
  kUint8 = 2,
  kInt8 = 3,
  kUint16 = 4,
  kInt16 = 5,
  kUint32 = 6,
  kInt32 = 7,
  kUint64 = 8,
  kInt64 = 9,
  kFloat16 = 10,
  kFloat32 = 11,
  kFloat64 = 12,
};

struct ElementType {
  DType dtype;

  const char* Name() const {
    switch (dtype) {
      case DType::kUnKnown:
        return "UnKnown";
      case DType::kBool:
        return "Bool";
      case DType::kUint8:
        return "Uint8";
      case DType::kInt8:
        return "Int8";
      case DType::kUint16:
        return "Uint16";
      case DType::kInt16:
        return "Int16";
      case DType::kUint32:
        return "Uint32";
      case DType::kInt32:
        return "Int32";
      case DType::kUint64:
        return "Uint64";
      case DType::kInt64:
        return "Int64";
      case DType::kFloat16:
        return "Float16";
      case DType::kFloat32:
        return "Float32";
      case DType::kFloat64:
        return "Float64";
      default:
        return "UnKnown";
    }
  }

  size_t ByteWidth() const {
    switch (dtype) {
      case DType::kUnKnown:
        return sizeof(UnKnown);
      case DType::kBool:
        return sizeof(uint8_t);
      case DType::kUint8:
        return sizeof(uint8_t);
      case DType::kInt8:
        return sizeof(int8_t);
      case DType::kUint16:
        return sizeof(uint16_t);
      case DType::kInt16:
        return sizeof(int16_t);
      case DType::kUint32:
        return sizeof(uint32_t);
      case DType::kInt32:
        return sizeof(int32_t);
      case DType::kUint64:
        return sizeof(uint64_t);
      case DType::kInt64:
        return sizeof(int64_t);
      case DType::kFloat16:
        return sizeof(half);
      case DType::kFloat32:
        return sizeof(float);
      case DType::kFloat64:
        return sizeof(double);
      default:
        return 0;
    }
  }

  bool operator==(const ElementType& other) const {
    return dtype == other.dtype;
  }

  bool operator!=(const ElementType& other) const {
    return dtype != other.dtype;
  }

  template <typename T>
  bool Is() const {
    return false;
  }

  template <typename T>
  static ElementType From() {
    ElementType etype;
    etype.dtype = DType::kUnKnown;

    return etype;
  }
};
template <>
inline bool ElementType::Is<UnKnown>() const {
  return dtype == DType::kUnKnown;
}

template <>
inline bool ElementType::Is<bool>() const {
  return dtype == DType::kBool;
}

template <>
inline bool ElementType::Is<uint8_t>() const {
  return dtype == DType::kUint8;
}

template <>
inline bool ElementType::Is<int8_t>() const {
  return dtype == DType::kInt8;
}

template <>
inline bool ElementType::Is<uint16_t>() const {
  return dtype == DType::kUint16;
}

template <>
inline bool ElementType::Is<int16_t>() const {
  return dtype == DType::kInt16;
}

template <>
inline bool ElementType::Is<uint32_t>() const {
  return dtype == DType::kUint32;
}

template <>
inline bool ElementType::Is<int32_t>() const {
  return dtype == DType::kInt32;
}

template <>
inline bool ElementType::Is<uint64_t>() const {
  return dtype == DType::kUint64;
}

template <>
inline bool ElementType::Is<int64_t>() const {
  return dtype == DType::kInt64;
}

template <>
inline bool ElementType::Is<half>() const {
  return dtype == DType::kFloat16;
}

template <>
inline bool ElementType::Is<float>() const {
  return dtype == DType::kFloat32;
}

template <>
inline bool ElementType::Is<double>() const {
  return dtype == DType::kFloat64;
}

#undef DEF_FROM_FUNC
#define DEF_FROM_FUNC(Type, T) \
  template <> \
  inline ElementType ElementType::From<T>() { \
    ElementType etype; \
    etype.dtype = DType::Type; \
    return etype; \
  }

DEF_FROM_FUNC(kUnKnown, UnKnown);
DEF_FROM_FUNC(kBool, bool);
DEF_FROM_FUNC(kUint8, uint8_t);
DEF_FROM_FUNC(kInt8, int8_t);
DEF_FROM_FUNC(kUint16, uint16_t);
DEF_FROM_FUNC(kInt16, int16_t);
DEF_FROM_FUNC(kUint32, uint32_t);
DEF_FROM_FUNC(kInt32, int32_t);
DEF_FROM_FUNC(kUint64, uint64_t);
DEF_FROM_FUNC(kInt64, int64_t);
DEF_FROM_FUNC(kFloat16, half);
DEF_FROM_FUNC(kFloat32, float);
DEF_FROM_FUNC(kFloat64, double);

#undef DEF_FROM_FUNC

}  // namespace dadt
