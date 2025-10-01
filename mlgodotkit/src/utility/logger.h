#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <functional>
#include <iostream>

namespace Logger {

    // Runtime-configurable logging handler
    inline std::function<void(const std::string&)> print_handler = [](const std::string& msg) {
        std::cout << msg << std::endl;  // default: stdout
    };

    inline int global_verbosity = 1;

    inline void set_handler(std::function<void(const std::string&)> handler) {
        print_handler = handler;
    }

    inline void set_verbosity(int level) {
        global_verbosity = level;
    }

    inline void debug(int level, const std::string& msg) {
        if (level <= global_verbosity && print_handler) {
            print_handler(msg);
        }
    }
}

#endif
