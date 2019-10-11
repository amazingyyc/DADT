#include "group.h"

namespace dadt {

Group::Group() {
}

Group::Group(std::unordered_set<TaskKey, TaskKeyHash, TaskKeyEqual> aggregate): aggregate_(std::move(aggregate)) {
}

void Group::insert_to_aggregate(TaskKey key) {
  aggregate_.insert(key);
}

}