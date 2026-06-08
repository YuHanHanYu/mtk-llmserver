#include "json_minimal.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace llmserver {
namespace {

const Json kNull;

class Parser {
public:
    Parser(const std::string& text, std::string* error) : text_(text), error_(error) {}

    Json parse() {
        skipWs();
        Json value = parseValue();
        skipWs();
        if (!failed_ && pos_ != text_.size()) {
            fail("unexpected trailing data");
        }
        return failed_ ? Json{} : value;
    }

private:
    Json parseValue() {
        skipWs();
        if (pos_ >= text_.size()) return fail("unexpected end of input");
        char c = text_[pos_];
        if (c == 'n') return parseLiteral("null", Json{});
        if (c == 't') return parseLiteral("true", jsonBool(true));
        if (c == 'f') return parseLiteral("false", jsonBool(false));
        if (c == '"') return parseString();
        if (c == '[') return parseArray();
        if (c == '{') return parseObject();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parseNumber();
        return fail("unexpected character");
    }

    Json parseLiteral(const char* literal, const Json& value) {
        std::string lit(literal);
        if (text_.compare(pos_, lit.size(), lit) != 0) return fail("invalid literal");
        pos_ += lit.size();
        return value;
    }

    Json parseString() {
        Json value;
        value.type = Json::String;
        pos_++;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') return value;
            if (c != '\\') {
                value.s.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return fail("invalid string escape");
            char e = text_[pos_++];
            switch (e) {
            case '"': value.s.push_back('"'); break;
            case '\\': value.s.push_back('\\'); break;
            case '/': value.s.push_back('/'); break;
            case 'b': value.s.push_back('\b'); break;
            case 'f': value.s.push_back('\f'); break;
            case 'n': value.s.push_back('\n'); break;
            case 'r': value.s.push_back('\r'); break;
            case 't': value.s.push_back('\t'); break;
            case 'u':
                if (pos_ + 4 > text_.size()) return fail("invalid unicode escape");
                value.s += "\\u" + text_.substr(pos_, 4);
                pos_ += 4;
                break;
            default:
                return fail("invalid string escape");
            }
        }
        return fail("unterminated string");
    }

    Json parseNumber() {
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        double number = std::strtod(begin, &end);
        if (end == begin) return fail("invalid number");
        pos_ += end - begin;
        return jsonNumber(number);
    }

    Json parseArray() {
        Json value;
        value.type = Json::Array;
        pos_++;
        skipWs();
        if (consume(']')) return value;
        while (true) {
            value.a.push_back(parseValue());
            if (failed_) return Json{};
            skipWs();
            if (consume(']')) return value;
            if (!consume(',')) return fail("expected ',' or ']'");
        }
    }

    Json parseObject() {
        Json value;
        value.type = Json::Object;
        pos_++;
        skipWs();
        if (consume('}')) return value;
        while (true) {
            skipWs();
            if (pos_ >= text_.size() || text_[pos_] != '"') return fail("expected object key");
            Json key = parseString();
            skipWs();
            if (!consume(':')) return fail("expected ':'");
            value.o[key.s] = parseValue();
            if (failed_) return Json{};
            skipWs();
            if (consume('}')) return value;
            if (!consume(',')) return fail("expected ',' or '}'");
        }
    }

    bool consume(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            pos_++;
            return true;
        }
        return false;
    }

    void skipWs() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) pos_++;
    }

    Json fail(const std::string& message) {
        failed_ = true;
        if (error_ && error_->empty()) {
            std::ostringstream oss;
            oss << message << " at byte " << pos_;
            *error_ = oss.str();
        }
        return Json{};
    }

    const std::string& text_;
    std::string* error_;
    size_t pos_ = 0;
    bool failed_ = false;
};

void dumpJsonValue(const Json& value, std::ostringstream& out) {
    switch (value.type) {
    case Json::Null:
        out << "null";
        break;
    case Json::Bool:
        out << (value.b ? "true" : "false");
        break;
    case Json::Number:
        if (std::floor(value.n) == value.n) out << static_cast<long long>(value.n);
        else out << value.n;
        break;
    case Json::String:
        out << '"' << jsonEscape(value.s) << '"';
        break;
    case Json::Array:
        out << '[';
        for (size_t i = 0; i < value.a.size(); ++i) {
            if (i) out << ',';
            dumpJsonValue(value.a[i], out);
        }
        out << ']';
        break;
    case Json::Object:
        out << '{';
        for (auto it = value.o.begin(); it != value.o.end(); ++it) {
            if (it != value.o.begin()) out << ',';
            out << '"' << jsonEscape(it->first) << "\":";
            dumpJsonValue(it->second, out);
        }
        out << '}';
        break;
    }
}

} // namespace

const Json& Json::get(const std::string& key) const {
    if (type != Object) return kNull;
    auto it = o.find(key);
    return it == o.end() ? kNull : it->second;
}

std::string Json::stringValue(const std::string& fallback) const {
    return type == String ? s : fallback;
}

int Json::intValue(int fallback) const {
    return type == Number ? static_cast<int>(n) : fallback;
}

bool Json::boolValue(bool fallback) const {
    return type == Bool ? b : fallback;
}

Json parseJson(const std::string& text, std::string* error) {
    if (error) error->clear();
    return Parser(text, error).parse();
}

std::string dumpJson(const Json& value) {
    std::ostringstream out;
    dumpJsonValue(value, out);
    return out.str();
}

std::string jsonEscape(const std::string& text) {
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) out << " ";
            else out << c;
        }
    }
    return out.str();
}

Json jsonObject(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json v;
    v.type = Json::Object;
    for (const auto& item : values) v.o[item.first] = item.second;
    return v;
}

Json jsonArray(std::initializer_list<Json> values) {
    Json v;
    v.type = Json::Array;
    v.a = values;
    return v;
}

Json jsonString(const std::string& value) {
    Json v;
    v.type = Json::String;
    v.s = value;
    return v;
}

Json jsonNumber(double value) {
    Json v;
    v.type = Json::Number;
    v.n = value;
    return v;
}

Json jsonBool(bool value) {
    Json v;
    v.type = Json::Bool;
    v.b = value;
    return v;
}

} // namespace llmserver
