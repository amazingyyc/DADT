#include "timeline.h"

namespace dadt {

TimeLine::TimeLine(const std::string &timeline_path)
  : timeline_path_(timeline_path), initialized_(false), stoped_(false) {
  writer_thread_ = std::thread(&TimeLine::writer_do_work, this);

  while (false == initialized_.load()) {
    // just wait
  }
}

TimeLine::~TimeLine() {
  stoped_ = true;

  writer_thread_.join();
}

// get current microseconds
int64_t TimeLine::get_current_microseconds() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}

void TimeLine::write(const TimeEvent &event) {
  writer_ << "{";
  writer_ << "\"name\": \"" << event.name << "\"";
  writer_ << ",";
  writer_ << "\"cat\": \"" << event.cat << "\"";
  writer_ << ",";
  writer_ << "\"ph\": \"" << event.ph << "\"";
  writer_ << ",";
  writer_ << "\"pid\": \"" << event.pid << "\"";
  writer_ << ",";
  writer_ << "\"ts\":" << event.ts;
  writer_ << "},";
  writer_ << "\n";

  writer_.flush();
}

void TimeLine::writer_do_work() {
  // init file
  writer_.open(timeline_path_, std::ios::out);

  ARGUMENT_CHECK(writer_.is_open(), "open file:" << timeline_path_ << " failed!");

  writer_ << "[\n";
  writer_.flush();

  initialized_ = true;

  while (true) {
    TimeEvent event;

    if (time_event_queue_.try_dequeue(event)) {
      write(event);
    } else if (stoped_) {
      break;
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  writer_.close();
}

// event begin
// sormat: {"name": "Asub1", "cat": "PERF", "ph": "B", "pid": "event", "ts": 829},
void TimeLine::begin(const std::string &op_name, const std::string& event_name) {
  TimeEvent event = {
    event_name,
    std::string("PERF"),
    std::string("B"),
    op_name,
    get_current_microseconds()
  };

  time_event_queue_.enqueue(std::move(event));
}

// event end
void TimeLine::end(const std::string &op_name, const std::string& event_name) {
  TimeEvent event = {
    event_name,
    std::string("PERF"),
    std::string("E"),
    op_name,
    get_current_microseconds()
  };

  time_event_queue_.enqueue(std::move(event));
}

// a lot of event begin
void TimeLine::begin(const std::vector<std::string> &op_names, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("B");

  auto ts = get_current_microseconds();

  for (auto &op_name : op_names) {
    TimeEvent event = {
      event_name,
      cat,
      ph,
      op_name,
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

// a lot of event end
void TimeLine::end(const std::vector<std::string> &op_names, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("E");

  auto ts = get_current_microseconds();

  for (auto &op_name : op_names) {
    TimeEvent event = {
      event_name,
      cat,
      ph,
      op_name,
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

// a lot of task begin
void TimeLine::begin(const std::vector<Task> &tasks, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("B");

  auto ts = get_current_microseconds();

  for (auto &task : tasks) {
    TimeEvent event = {
      event_name,
      cat,
      ph,
      task.name,
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

//a lot of task end
void TimeLine::end(const std::vector<Task> &tasks, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("E");

  auto ts = get_current_microseconds();

  for (auto &task : tasks) {
    TimeEvent event = {
      event_name,
      cat,
      ph,
      task.name,
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

void TimeLine::begin(const std::vector<TaskKey> &task_keys, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("B");

  auto ts = get_current_microseconds();

  for (auto &key : task_keys) {
    TimeEvent event = {
      event_name,
      cat,
      ph,
      std::get<1>(key),
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

void TimeLine::end(const std::vector<TaskKey> &task_keys, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("E");

  auto ts = get_current_microseconds();

    for (auto &key : task_keys) {
      TimeEvent event = {
      event_name,
      cat,
      ph,
      std::get<1>(key),
      ts
    };

    time_event_queue_.enqueue(std::move(event));
  }
}

}