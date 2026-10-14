#pragma once

#include <string>

namespace pbl {

class Logger {
public:
  static void init();
  static void log(const std::string &msg);
  static void log(const char *msg);
};

} // namespace pbl
