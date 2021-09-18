#pragma once


namespace ca {

enum  class Environment : int {
    NORD_WEST  = 0b1,
    NORD       = 0b10,
    NORD_EAST  = 0b100,
    EAST       = 0b1000,
    SOUTH_EAST = 0b10000,
    SOUTH      = 0b100000,
    SOUTH_WEST = 0b1000000,
    WEST       = 0b10000000,
    CENTER     = 0b100000000
};

inline Environment operator | (Environment lhs, Environment rhs) {
    return Environment((int)lhs | (int)rhs);
}

inline Environment operator & (Environment lhs, Environment rhs) {
    return Environment((int)lhs & (int)rhs);
}

inline Environment moore() {
    return Environment::NORD_WEST  |
           Environment::NORD       |
           Environment::NORD_EAST  |
           Environment::EAST       |
           Environment::SOUTH_EAST |
           Environment::SOUTH      |
           Environment::SOUTH_WEST |
           Environment::WEST;
}

inline Environment von_neuman() {
    return Environment::NORD  |
           Environment::EAST  |
           Environment::SOUTH |
           Environment::WEST;
}

inline Environment full() {
    return moore() | Environment::CENTER;
}

}
