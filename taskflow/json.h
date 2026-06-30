#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace taskflow::json {

std::string escape(std::string_view input);
void write_string(std::ostream& out, std::string_view value);
void write_uint64(std::ostream& out, std::uint64_t value);
void write_double(std::ostream& out, double value);

}  // namespace taskflow::json
