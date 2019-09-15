#include "timeline.h"

namespace dadt {

TimeLine::TimeLine(const std::string &timeline_path): timeline_path_(timeline_path) {
  // open file
  writer_.open(timeline_path_, std::ios::out);

  ARGUMENT_CHECK(writer_.is_open(), "open file:" << timeline_path_ << " failed!");

  writer_ << "[\n";
  writer_.flush();
}

TimeLine::~TimeLine() {
  writer_.close();
}

// get current microseconds
int64_t TimeLine::get_current_microseconds() {
  auto time_now = std::chrono::system_clock::now();
  auto duration_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(time_now.time_since_epoch());
  return duration_in_ms.count();
}

void TimeLine::write(const std::string &event_name, 
                     const std::string &cat, 
                     const std::string &ph, 
                     const std::string &op_name, 
                     int64_t ts) {
  writer_ << "{";
  writer_ << "\"name\": \"" << event_name << "\"";
  writer_ << ",";
  writer_ << "\"cat\": \"" << cat << "\"";
  writer_ << ",";
  writer_ << "\"ph\": \"" << ph << "\"";
  writer_ << ",";
  writer_ << "\"pid\": \"" << op_name << "\"";
  writer_ << ",";
  writer_ << "\"ts\":" << ts;
  writer_ << "},";
  writer_ << "\n";

  writer_.flush();
}

// event begin
// sormat: {"name": "Asub1", "cat": "PERF", "ph": "B", "pid": "event", "ts": 829},
void TimeLine::begin(const std::string &op_name, const std::string& event_name) {
  std::unique_lock<std::mutex> lock(mutex_);

  write(event_name, std::string("PERF"), std::string("B"), op_name, get_current_microseconds());
}

// event end
void TimeLine::end(const std::string &op_name, const std::string& event_name) {
  std::unique_lock<std::mutex> lock(mutex_);

  write(event_name, std::string("PERF"), std::string("E"), op_name, get_current_microseconds());
}

// a lot of event begin
void TimeLine::begin(const std::vector<std::string> &op_names, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("B");

  auto ts = get_current_microseconds();

  std::unique_lock<std::mutex> lock(mutex_);

  for (auto &op_name : op_names) {
    write(event_name, cat, ph, op_name, ts);
  }
}

// a lot of event end
void TimeLine::end(const std::vector<std::string> &op_names, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("E");

  auto ts = get_current_microseconds();

  std::unique_lock<std::mutex> lock(mutex_);

  for (auto &op_name : op_names) {
    write(event_name, cat, ph, op_name, ts);
  }
}

// a lot of task begin
void TimeLine::begin(const std::vector<Task> &tasks, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("B");

  auto ts = get_current_microseconds();

  std::unique_lock<std::mutex> lock(mutex_);

  for (auto &task : tasks) {
    write(event_name, cat, ph, task.name, ts);
  }
}

//a lot of task end
void TimeLine::end(const std::vector<Task> &tasks, const std::string& event_name) {
  std::string cat("PERF");
  std::string ph("E");

  auto ts = get_current_microseconds();

  std::unique_lock<std::mutex> lock(mutex_);

  for (auto &task : tasks) {
    write(event_name, cat, ph, task.name, ts);
  }
}

}