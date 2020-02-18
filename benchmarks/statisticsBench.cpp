// Copyright (c) 2019 Shapelets.io
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <benchmark/benchmark.h>
#include <khiva/statistics.h>
#include "khivaBenchmark.h"

template <af::Backend BE, int D>
void seq_PearsonCorr(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);
    // int k = state.range(2);
    af::array result = af::constant(0, m, m);

    auto t1 = af::randu(n, m, f32);
    auto t2 = af::randu(n, m, f32);

    af::sync();
    while (state.KeepRunning()) {
        af::array a = af::constant(0, n);
        af::array b = af::constant(0, n);
        for (int i = 0; i < t1.dims(1); i++) {
            for (int j = 0; j < t1.dims(1); j++) {
                a = t1(af::span, i);
                b = t2(af::span, j);
                result(i, j) = af::corrcoef<float>(a, b);
            }
        }
        result.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void PearsonCorr(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);
    // int k = state.range(2);

    auto t1 = af::randu(n, m, f32);
    auto t2 = af::randu(n, m, f32);

    af::sync();
    while (state.KeepRunning()) {
        af::array result = khiva::statistics::pearsonCorrelation(t1, t2);
        result.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void Covariance(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto cov = khiva::statistics::covariance(t);
        cov.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void Kurtosis(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto kurtosis = khiva::statistics::kurtosis(t);
        kurtosis.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void LjungBox(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);
    auto lag = state.range(2);

    auto t = af::randu(n, m, f64);
    af::sync();
    while (state.KeepRunning()) {
        auto ljungBox = khiva::statistics::ljungBox(t, lag);
        ljungBox.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void Moment(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);
    auto k = state.range(2);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto moment = khiva::statistics::moment(t, k);
        moment.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void Quantile(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);

    float qs[] = {0.3f, 0.4f, 0.5f, 0.99f};
    auto q = af::array(4, qs);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto quantile = khiva::statistics::quantile(t, q);
        quantile.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void QuantilesCut(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);
    auto qs = static_cast<float>(state.range(2));

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto quantilesCut = khiva::statistics::quantilesCut(t, qs);
        quantilesCut.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void SampleStdev(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto sampleStdev = khiva::statistics::sampleStdev(t);
        sampleStdev.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

template <af::Backend BE, int D>
void Skewness(benchmark::State &state) {
    af::setBackend(BE);
    af::setDevice(D);

    auto n = state.range(0);
    auto m = state.range(1);

    auto t = af::randu(n, m, f64);

    af::sync();
    while (state.KeepRunning()) {
        auto kurtosis = khiva::statistics::skewness(t);
        kurtosis.eval();
        af::sync();
    }
    addMemoryCounters(state);
}

void cudaBenchmarks() {
    BENCHMARK_TEMPLATE(seq_PearsonCorr, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(PearsonCorr, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(Covariance, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 32 << 10}, {1, 16}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Kurtosis, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(LjungBox, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {32, 256}, {32, 256}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Moment, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {1, 4}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Quantile, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(QuantilesCut, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {2, 8}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(SampleStdev, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Skewness, af::Backend::AF_BACKEND_CUDA, CUDA_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
}

void openclBenchmarks() {
    BENCHMARK_TEMPLATE(seq_PearsonCorr, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(PearsonCorr, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(Covariance, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 32 << 10}, {1, 16}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Kurtosis, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(LjungBox, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {32, 256}, {32, 256}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Moment, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {1, 4}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Quantile, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(QuantilesCut, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {2, 8}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(SampleStdev, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Skewness, af::Backend::AF_BACKEND_OPENCL, OPENCL_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
}

void cpuBenchmarks() {
    BENCHMARK_TEMPLATE(seq_PearsonCorr, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(PearsonCorr, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 256 << 10}, {16, 128}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
    BENCHMARK_TEMPLATE(Covariance, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 32 << 10}, {1, 16}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Kurtosis, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(LjungBox, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {32, 256}, {32, 256}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Moment, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {1, 4}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Quantile, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(QuantilesCut, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}, {2, 8}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(SampleStdev, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);

    BENCHMARK_TEMPLATE(Skewness, af::Backend::AF_BACKEND_CPU, CPU_BENCHMARKING_DEVICE)
        ->RangeMultiplier(2)
        ->Ranges({{1 << 10, 512 << 10}, {16, 32}})
        ->Unit(benchmark::TimeUnit::kMicrosecond);
}

KHIVA_BENCHMARK_MAIN(cudaBenchmarks, openclBenchmarks, cpuBenchmarks)
