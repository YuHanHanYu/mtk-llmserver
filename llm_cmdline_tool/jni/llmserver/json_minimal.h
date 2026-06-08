#pragma once

#include <map>
#include <string>
#include <vector>

namespace llmserver {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;

    bool isNull() const { return type == Null; }
    bool isBool() const { return type == Bool; }
    bool isNumber() const { return type == Number; }
    bool isString() const { return type == String; }
    bool isArray() const { return type == Array; }
    bool isObject() const { return type == Object; }

    const Json& get(const std::string& key) const;
    std::string stringValue(const std::string& fallback = "") const;
    int intValue(int fallback = 0) const;
    bool boolValue(bool fallback = false) const;
};

Json parseJson(const std::string& text, std::string* error = nullptr);
std::string dumpJson(const Json& value);
std::string jsonEscape(const std::string& text);
Json jsonObject(std::initializer_list<std::pair<const std::string, Json>> values);
Json jsonArray(std::initializer_list<Json> values);
Json jsonString(const std::string& value);
Json jsonNumber(double value);
Json jsonBool(bool value);

} // namespace llmserver
