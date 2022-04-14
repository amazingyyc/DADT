#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

#include "common/context.h"

namespace dadt {
namespace pytorch {

void Initialize(const Config& config);

void Shutdown();

bool Initialized();

int Size();

int LocalSize();

int Rank();

int LocalRank();

void Barrier();

void LocalBarrier();

torch::Tensor BroadCast(uint32_t id, torch::Tensor input);

torch::Tensor AllReduce(uint32_t id, torch::Tensor input);

torch::Tensor AllReduceAsync(uint32_t id, torch::Tensor input);

torch::Tensor CooAllReduce(uint32_t id, torch::Tensor input);

torch::Tensor CooAllReduceAsync(uint32_t id, torch::Tensor input);

}  // namespace pytorch
}  // namespace dadt
