#include <thrust/device_vector.h>
#include <gtest/gtest.h>

#include "Rule.cuh"


static const std::vector<ca::Cell>& native_locality() {
    static std::vector<ca::Cell> locality = {
        0x0, 0x0, 0x0,
        0x0,      0x1,
        0x0, 0x1, 0x1
    };
    return locality;
}


TEST(Rule, native) {
    ca::Rule rule = ca::initialize_native("00U100000", ca::moore());
    ca::Cell current = 0x0;
    current = rule.apply(native_locality().data(), current);
    EXPECT_EQ(current, 0x1);
    EXPECT_EQ(rule.env(), ca::moore());
}

__global__ void __apply__(ca::Rule* ctx, ca::Cell* locality, ca::Cell current) {
    current = ctx->apply(locality, current);
    assert(current == 1);
}


TEST(Rule, global) {
    auto rule = ca::initialize_global("00U100000", ca::moore());
    ca::Cell current = 0x0;
    ca::Cell result;
    ca::Rule* dev_rule = nullptr;
    dev_rule = utils::copy_allocate_managed(dev_rule, rule);

    thrust::device_vector<ca::Cell> locality = native_locality();
    __apply__<<<1, 1>>>(dev_rule, thrust::raw_pointer_cast(locality.data()),
                        current);
    cudaDeviceSynchronize();

    cudaFree(dev_rule);
}

