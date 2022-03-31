#pragma once

namespace dadt {

class StreamGuard {
public:
  StreamGuard() = default;

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;

  const StreamGuard& operator=(const StreamGuard&) = delete;
  const StreamGuard& operator=(StreamGuard&&) = delete;

  virtual ~StreamGuard() = default;
};

}  // namespace dadt
