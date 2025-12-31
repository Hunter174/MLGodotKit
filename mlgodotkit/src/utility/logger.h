#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace Logger {

	inline int global_verbosity = 1; // 0 is error, 1 is warning, 2 is info

	// --- Verbosity control ---
	inline void set_verbosity(int level) {
		global_verbosity = level;
	}

	// --- Logging functions ---
	inline void debug(int level, const std::string &msg) {
		if (level <= global_verbosity) {
			godot::UtilityFunctions::print_rich(
				"[color=#888888][DEBUG] " + godot::String(msg.c_str()) + "[/color]");
		}
	}

	inline void info(const std::string &msg) {
		godot::UtilityFunctions::print(msg.c_str());
	}

	inline void warn(const std::string &msg) {
		godot::UtilityFunctions::push_warning(godot::String("[WARN] ") + msg.c_str());
	}

	inline void error(const std::string &msg) {
		godot::UtilityFunctions::push_error(godot::String("[ERROR] ") + msg.c_str());
	}

	// --- Fatal error (halts current function safely) ---
	inline godot::Error error_raise(const std::string &msg) {
		ERR_PRINT("[FATAL] " + godot::String(msg.c_str()));
		return godot::Error::FAILED;
	}

	// --- Assertion helper ---
	inline godot::Error assert_raise(bool condition, const std::string &msg) {
		if (!condition) {
			ERR_PRINT("[ASSERT FAILED] " + godot::String(msg.c_str()));
			return godot::Error::FAILED;
		}
		return godot::Error::OK;
	}

} // namespace Logger

#endif