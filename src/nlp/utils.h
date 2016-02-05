#pragma once

#include <string>

std::wstring ToWideStr(const std::string& str);
std::string ToNarrowStr(const std::wstring& str);

