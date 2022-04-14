#pragma once

#include "common/iwriter.h"

namespace dadt {

class MemWriter : public IWriter {
private:
  char* ptr_;

  size_t capacity_;
  size_t offset_;

public:
  MemWriter();

  explicit MemWriter(size_t expect);
  explicit MemWriter(MemWriter&&);

  const MemWriter& operator=(MemWriter&&);

  MemWriter(const MemWriter&) = delete;
  MemWriter& operator=(const MemWriter&) = delete;

  ~MemWriter();

private:
  void Growth(size_t new_size);

public:
  char* ptr() const;

  size_t capacity() const;

  size_t offset() const;

  bool Write(const char* bytes, size_t size) override;
};

}  // namespace dadt
