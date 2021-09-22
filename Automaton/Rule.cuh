#pragma once

#include <cassert>
#include <cinttypes>

#include <Utils.hpp>

#include "Environment.hpp"


namespace ca {
using Cell = int64_t;

enum class RuleType {
    KILL = 0,
    REVIVE = 1,
    LEAVE = 127
};

class Rule {
public:
    __host__ Rule() = default;
    __host__ Rule(RuleType* rules, Environment env, int len, bool host);
    __host__ Rule(const Rule& old);
    __host__ Rule& operator = (const Rule& old);
    __host__ Rule(Rule&& old);
    __host__ Rule& operator = (Rule&& old);
    __host__ ~Rule();

    ATTRIBS Cell apply(const Cell* locality, Cell current) const;

    ATTRIBS int len() const { return _len; }
    ATTRIBS const RuleType* rules() const { return _rules; }
    ATTRIBS Environment env() const { return _env; }
    ATTRIBS bool host() const { return _host; }

    ATTRIBS void dump() const;

private:
    Environment _env;
    RuleType* _rules = nullptr;
    int _len = 0;
    bool _host = false;
};

Rule initialize_native(const char* raw, Environment env);
Rule initialize_global(const char* raw, Environment env);

}
