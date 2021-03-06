#ifndef ELEMENT_TYEP_H
#define ELEMENT_TYEP_H

#include <iostream>
#include <mpi.h>

#ifdef HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include "definition.h"

// define the half
#ifndef HAVE_NCCL

typedef struct half_ {
  uint16_t value;
} half;

#endif

// define a unknow type
class UnKnownDType {
};

namespace dadt {

enum class DType : int32_t {
  UnKnown = 0,
  Bool    = 1,
  Uint8   = 2,
  Int8    = 3,
  Uint16  = 4,
  Int16   = 5,
  Uint32  = 6,
  Int32   = 7,
  Uint64  = 8,
  Int64   = 9,
  Float16 = 10,
  Float32 = 11,
  Float64 = 12,
};

class ElementType {
private:
  DType id_;

  size_t byte_width_;

  std::string name_;

  explicit ElementType(DType id, size_t byte_width, std::string name)
      : id_(id), byte_width_(byte_width), name_(name) {}

public:
  DType id() const {
    return id_;
  }

  size_t byte_width() const {
    return byte_width_;
  }

  std::string name() const {
    return name_;
  }

  bool operator==(const ElementType &other) const {
    return this->id_ == other.id_;
  }

  bool operator!=(const ElementType &other) const {
    return this->id_ != other.id_;
  }

  template <typename T> bool is() const {
    RUNTIME_ERROR("Unknow type");
  }

  template <typename T> static ElementType from() {
    RUNTIME_ERROR("Unknow type");
  }
};

template <> inline bool ElementType::is<UnKnownDType>() const {
  return this->id_ == DType::UnKnown;
}

template <> inline bool ElementType::is<bool>() const {
  return this->id_ == DType::Bool;
}

template <> inline bool ElementType::is<uint8_t>() const {
  return this->id_ == DType::Uint8;
}

template <> inline bool ElementType::is<int8_t>() const {
  return this->id_ == DType::Int8;
}

template <> inline bool ElementType::is<uint16_t>() const {
  return this->id_ == DType::Uint16;
}

template <> inline bool ElementType::is<int16_t>() const {
  return this->id_ == DType::Int16;
}

template <> inline bool ElementType::is<uint32_t>() const {
  return this->id_ == DType::Uint32;
}

template <> inline bool ElementType::is<int32_t>() const {
  return this->id_ == DType::Int32;
}

template <> inline bool ElementType::is<uint64_t>() const {
  return this->id_ == DType::Uint64;
}

template <> inline bool ElementType::is<int64_t>() const {
  return this->id_ == DType::Int64;
}

template <> inline bool ElementType::is<half>() const {
  return this->id_ == DType::Float16;
}

template <> inline bool ElementType::is<float>() const {
  return this->id_ == DType::Float32;
}

template <> inline bool ElementType::is<double>() const {
  return this->id_ == DType::Float64;
}

template <> inline ElementType ElementType::from<UnKnownDType>() {
  return ElementType(DType::UnKnown, 0, "unknown");
}

template <> inline ElementType ElementType::from<bool>() {
  return ElementType(DType::Bool, sizeof(bool), "bool");
}

template <> inline ElementType ElementType::from<uint8_t>() {
  return ElementType(DType::Uint8, sizeof(uint8_t), "uint8_t");
}

template <> inline ElementType ElementType::from<int8_t>() {
  return ElementType(DType::Int8, sizeof(int8_t), "int8_t");
}

template <> inline ElementType ElementType::from<uint16_t>() {
  return ElementType(DType::Uint16, sizeof(uint16_t), "uint16_t");
}

template <> inline ElementType ElementType::from<int16_t>() {
  return ElementType(DType::Int16, sizeof(int16_t), "int16_t");
}

template <> inline ElementType ElementType::from<uint32_t>() {
  return ElementType(DType::Uint32, sizeof(uint32_t), "uint32_t");
}

template <> inline ElementType ElementType::from<int32_t>() {
  return ElementType(DType::Int32, sizeof(int32_t), "int32_t");
}

template <> inline ElementType ElementType::from<uint64_t>() {
  return ElementType(DType::Uint64, sizeof(uint64_t), "uint64_t");
}

template <> inline ElementType ElementType::from<int64_t>() {
  return ElementType(DType::Int64, sizeof(int64_t), "int64_t");
}

template <> inline ElementType ElementType::from<half>() {
  return ElementType(DType::Float16, sizeof(half), "float16");
}

template <> inline ElementType ElementType::from<float>() {
  return ElementType(DType::Float32, sizeof(float), "float32");
}

template <> inline ElementType ElementType::from<double>() {
  return ElementType(DType::Float64, sizeof(double), "float64");
}

}

#endif