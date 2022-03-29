#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ireader.h"
#include "task.h"

namespace dadt {

class Deserialize {
private:
  IReader* reader_;

public:
  Deserialize(IReader* reader) : reader_(reader) {
  }

public:
  bool Read(void* target, size_t size) {
    return reader_->Read(target, size);
  }

  template <typename T>
  bool operator>>(T& v) {
    return false;
  }
};

#define BASIC_TYPE_DESERIALIZE(T) \
  template <> \
  inline bool Deserialize::operator>>(T& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    return Read(&v, sizeof(v)); \
  }

BASIC_TYPE_DESERIALIZE(bool);
BASIC_TYPE_DESERIALIZE(uint8_t);
BASIC_TYPE_DESERIALIZE(int8_t);
BASIC_TYPE_DESERIALIZE(uint16_t);
BASIC_TYPE_DESERIALIZE(int16_t);
BASIC_TYPE_DESERIALIZE(uint32_t);
BASIC_TYPE_DESERIALIZE(int32_t);
BASIC_TYPE_DESERIALIZE(uint64_t);
BASIC_TYPE_DESERIALIZE(int64_t);
BASIC_TYPE_DESERIALIZE(float);
BASIC_TYPE_DESERIALIZE(double);

#undef BASIC_TYPE_DESERIALIZE

#define VEC_BASIC_TYPE_DESERIALIZE(T) \
  template <> \
  inline bool Deserialize::operator>>(std::vector<T>& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    uint64_t size; \
    if (((*this) >> size) == false) { \
      return false; \
    } \
    v.resize(size); \
    return Read(&(v[0]), size * sizeof(T)); \
  }

VEC_BASIC_TYPE_DESERIALIZE(uint8_t);
VEC_BASIC_TYPE_DESERIALIZE(int8_t);
VEC_BASIC_TYPE_DESERIALIZE(uint16_t);
VEC_BASIC_TYPE_DESERIALIZE(int16_t);
VEC_BASIC_TYPE_DESERIALIZE(uint32_t);
VEC_BASIC_TYPE_DESERIALIZE(int32_t);
VEC_BASIC_TYPE_DESERIALIZE(uint64_t);
VEC_BASIC_TYPE_DESERIALIZE(int64_t);
VEC_BASIC_TYPE_DESERIALIZE(float);
VEC_BASIC_TYPE_DESERIALIZE(double);

#undef VEC_BASIC_TYPE_DESERIALIZE

template <>
inline bool Deserialize::operator>>(std::string& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);

  return Read((void*)v.data(), size);
}

template <>
inline bool Deserialize::operator>>(TaskKey& v) {
  return (*this) >> v.type && (*this) >> v.id;
}

template <>
inline bool Deserialize::operator>>(std::vector<TaskKey>& v) {
  uint64_t size;
  if (((*this) >> size) == false) {
    return false;
  }

  v.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    if (((*this) >> v[i]) == false) {
      return false;
    }
  }

  return true;
}

}  // namespace dadt
