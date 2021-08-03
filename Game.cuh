#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace game {

using Color = glm::vec3;

class Life {
public:
    Life(int width, int height)
        : _width(width)
        , _height(height)
        , _len(width * height)
    {
        HANDLE_ERROR(cudaMalloc(&_next_device_generation, _len * sizeof (bool)));
        HANDLE_ERROR(cudaMalloc(&_prev_device_generation, _len * sizeof (bool)));
    }

    ~Life() {
        cudaFree(_prev_device_generation);
        cudaFree(_next_device_generation);
    }

    __host__ void initialize() {
        static constexpr bool SPACE[] = {true, false, false};
        static constexpr size_t SIZE = sizeof(SPACE) / sizeof(bool);

        bool* initial = new bool [_len];
        for (int i = 0; i < _len; ++i) {
            bool index = 0 + rand() % SIZE;
            initial[i] = SPACE[index];
        }
        HANDLE_ERROR(cudaMemcpy(_prev_device_generation, initial,
                     _len * sizeof (bool), cudaMemcpyHostToDevice));

        delete [] initial;
    }

    __host__ __device__ int eval_index(int w_pos, int h_pos) const {
        if (w_pos == -1) {
            w_pos = width() - 1;
        }
        if (w_pos == width()) {
            w_pos = 0;
        }
        if (h_pos == -1) {
            h_pos = height() - 1;
        }
        if (h_pos == height()) {
            h_pos = 0;
        }

        int index = w_pos + (width() * h_pos);
        assert(index < len());
        return index;
    }

    __device__ bool cell_status(int w_pos, int h_pos) const {
        const int current_idx = eval_index(w_pos, h_pos);
        assert(current_idx < len());

        bool neighbours[8];
        neighbours[0] = _prev_device_generation[eval_index(w_pos - 1, h_pos - 1)];
        neighbours[1] = _prev_device_generation[eval_index(w_pos,     h_pos - 1)];
        neighbours[2] = _prev_device_generation[eval_index(w_pos + 1, h_pos - 1)];
        neighbours[3] = _prev_device_generation[eval_index(w_pos - 1, h_pos    )];
        neighbours[4] = _prev_device_generation[eval_index(w_pos + 1, h_pos    )];
        neighbours[5] = _prev_device_generation[eval_index(w_pos - 1, h_pos + 1)];
        neighbours[6] = _prev_device_generation[eval_index(w_pos,     h_pos + 1)];
        neighbours[7] = _prev_device_generation[eval_index(w_pos + 1, h_pos + 1)];

        int count = 0;
        for (int i = 0; i < 8; ++i) { count += neighbours[i] ? 1 : 0; }

        return _prev_device_generation[current_idx] ?
            count == 2 || count == 3 :
            count == 3;
    }

    __device__ bool eval_generation(int w_pos, int h_pos) {
        const int current_idx = eval_index(w_pos, h_pos);
        assert(current_idx < len());
    
        const bool status = cell_status(w_pos, h_pos);
        _next_device_generation[current_idx] = status;
        return status;
    }

    __device__ bool* prev() { return _prev_device_generation; }
    __device__ bool* next() { return _next_device_generation; }

    __host__ __device__ int len() const { return _len; }
    __host__ __device__ int width() const { return _width; }
    __host__ __device__ int height() const { return _height; }

private:
    int _width, _height, _len;
    bool* _prev_device_generation;
    bool* _next_device_generation;
};

}
