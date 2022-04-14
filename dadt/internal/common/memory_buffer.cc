// #include "common/memory_buffer.h"

// namespace dadt {

// MemoryBuffer::MemoryBuffer(Device* device)
//     : device_(device), ptr_(nullptr), size_(0) {
// }

// MemoryBuffer::~MemoryBuffer() {
//   if (nullptr != ptr_) {
//     device_->Free(ptr_);
//   }

//   ptr_ = nullptr;
//   size_ = 0;
// }

// void* MemoryBuffer::ptr(size_t offset) {
//   return ((uint8_t*)ptr_) + offset;
// }

// size_t MemoryBuffer::size() {
//   return size_;
// }

// void MemoryBuffer::Reserve(size_t new_size) {
//   if (new_size > size_) {
//     if (nullptr != ptr_) {
//       device_->Free(ptr_);
//     }

//     ptr_ = device_->Malloc(new_size);
//     size_ = new_size;
//   }
// }

// void MemoryBuffer::Zero() {
//   if (nullptr != ptr_) {
//     device_->Zero(ptr_, size_);
//   }
// }

// }  // namespace dadt
