#include "taskflow/json.h"

#include <iomanip>
#include <sstream>

namespace taskflow::json {

std::string escape(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const char c : input) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream oss;
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c));
                    out += oss.str();
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

void write_string(std::ostream& out, std::string_view value) {
    out << '"' << escape(value) << '"';
}

void write_uint64(std::ostream& out, std::uint64_t value) {
    out << value;
}

void write_double(std::ostream& out, double value) {
    out << std::setprecision(17) << value;
}

}  // namespace taskflow::json
