#pragma once
#include <tuple>

#include "constants.hpp"

struct MuZeroConfig
{
    unsigned int seed = 0;
    const std::tuple<unsigned int, unsigned int, unsigned int> observation_shape = std::make_tuple(PLANE_NUM, NUM_ROW, NUM_COL);
};
