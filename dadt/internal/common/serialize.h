#pragma once

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "iwriter.h"
#include "task.h"

namespace dadt {

class Serialize {
private:
  // We donot remove or clear the buffer, just append data.
  IWriter* buf_;

public:
  Serialize(IWriter* buf) : buf_(buf) {
  }

  ~Serialize() = default;

  bool Write(const void* ptr, size_t size) {
    return buf_->Write((const char*)ptr, size);
  }

  template <typename T>
  bool operator<<(const T& v) {
    return false;
  }
};

#define BASIC_TYPE_SERIALIZE(T) \
  template <> \
  inline bool Serialize::operator<<(const T& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    return Write(&v, sizeof(v)); \
  }

BASIC_TYPE_SERIALIZE(bool);
BASIC_TYPE_SERIALIZE(uint8_t);
BASIC_TYPE_SERIALIZE(int8_t);
BASIC_TYPE_SERIALIZE(uint16_t);
BASIC_TYPE_SERIALIZE(int16_t);
BASIC_TYPE_SERIALIZE(uint32_t);
BASIC_TYPE_SERIALIZE(int32_t);
BASIC_TYPE_SERIALIZE(uint64_t);
BASIC_TYPE_SERIALIZE(int64_t);
BASIC_TYPE_SERIALIZE(float);
BASIC_TYPE_SERIALIZE(double);

#undef BASIC_TYPE_SERIALIZE

#define VEC_BASIC_TYPE_SERIALIZE(T) \
  template <> \
  inline bool Serialize::operator<<(const std::vector<T>& v) { \
    static_assert(std::is_pod<T>::value, #T " must be a POD type."); \
    uint64_t size = v.size(); \
    if (((*this) << size) == false) { \
      return false; \
    } \
    return Write(&(v[0]), size * sizeof(T)); \
  }

VEC_BASIC_TYPE_SERIALIZE(uint8_t);
VEC_BASIC_TYPE_SERIALIZE(int8_t);
VEC_BASIC_TYPE_SERIALIZE(uint16_t);
VEC_BASIC_TYPE_SERIALIZE(int16_t);
VEC_BASIC_TYPE_SERIALIZE(uint32_t);
VEC_BASIC_TYPE_SERIALIZE(int32_t);
VEC_BASIC_TYPE_SERIALIZE(uint64_t);
VEC_BASIC_TYPE_SERIALIZE(int64_t);
VEC_BASIC_TYPE_SERIALIZE(float);
VEC_BASIC_TYPE_SERIALIZE(double);

#undef VEC_BASIC_TYPE_SERIALIZE

template <>
inline bool Serialize::operator<<(const std::string& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  return Write(v.data(), v.size());
}

template <>
inline bool Serialize::operator<<(const TaskKey& v) {
  return (*this) << v.type && (*this) << v.id;
}

template <>
inline bool Serialize::operator<<(const std::vector<TaskKey>& v) {
  uint64_t size = v.size();
  if (((*this) << size) == false) {
    return false;
  }

  for (auto& i : v) {
    if (((*this) << i) == false) {
      return false;
    }
  }

  return true;
}

}  // namespace dadt
