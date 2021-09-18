#ifdef _MSC_VER
    #include <nmmintrin.h>
    #define bitcount _mm_popcnt_u64
#endif

#include <stdio.h>
#include "Rule.cuh"


namespace ca {

__host__ Rule::Rule(RuleType* rules, Environment env, int len, bool host)
    : _rules(rules)
    , _env(env)
    , _len(len)
    , _host(host)
{}


__host__ Rule::Rule(const Rule& old) {
    _len = old._len;
    _host = old._host;
    _rules = utils::allocate<RuleType>(_len, _host);
    utils::copy(_rules, old._rules, _len, _host);
}


__host__ Rule& Rule::operator = (const Rule& old) {
    if (this == &old) return *this;

    _len = old._len;
    _host = old._host;
    RuleType* rules = utils::allocate<RuleType>(_len, _host);
    utils::copy(rules, old._rules, _len, _host);
    utils::free(_rules, _host);
    _rules = rules;
    return *this;
}


__host__ Rule::Rule(Rule&& old) {
    _rules = old._rules;
    utils::free(old._rules, old._host);
    _len = old._len;
    _host = old._host;
}


__host__ Rule& Rule::operator = (Rule&& old) {
    if (this == &old) return *this;

    utils::free(_rules, old._host);
    _rules = old._rules;
    _len = old._len;
    _host = old._host;
    return *this;
}


__host__ Rule::~Rule() { utils::free(_rules, _host); }


ATTRIBS Cell Rule::apply(const Cell* locality, Cell current) const {
    assert(locality != nullptr);

    int8_t summ = 0;
    for (int i = 0; i < len(); ++i) {
        summ += ((locality[i] & 0x1) == 0x1) ? 1 : 0;
    } assert(summ <= len());

    Cell last_state = current & 0x1;
    current = current << 1;
    RuleType rule = _rules[summ - 1];
    return rule == RuleType::LEAVE ? (current | last_state) :
                                     (current | (Cell)rule);
}



RuleType convert(char symbol) {
    if (symbol == '0') { return RuleType::KILL; }
    if (symbol == '1') { return RuleType::REVIVE; }
    return RuleType::LEAVE;
}

RuleType* convert(const char* raw, int len) {
    RuleType* rules = new RuleType[len];
    for (size_t i = 0; i < len; ++i) {
        rules[i] = convert(raw[i]);
    }
    return rules;
}

RuleType* convert_to_device(RuleType* host_vector, int len) {
    RuleType* out = utils::allocate_dev<RuleType>(len);
    utils::host_to_device(out, host_vector, len);
    return out;
}

Rule initialize_native(const char* raw, Environment env) {
    int len = std::strlen(raw);
    assert(len == bitcount((int)env));
    return Rule(convert(raw, len), env, len, true);
}

Rule initialize_global(const char* raw, Environment env) {
    int len = std::strlen(raw);
    assert(len == bitcount((int)env));
    return Rule(convert_to_device(convert(raw, len), len), env, len, false);
}

}
