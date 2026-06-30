/*
 * benchmark_trsv_cpu.cpp — CPU TRSV benchmark (零依赖, 纯 C++ 手写)
 *
 * 编译: cl /O2 /arch:AVX2 benchmark_trsv_cpu.cpp /Fe:benchmark_trsv_cpu.exe
 * VTune: Launch Application, 参数: <N> --no-pause
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>

static void trsv_upper(int N, const float* T, int lda, float* x) {
    for (int i = N - 1; i >= 0; --i) {
        float sum = 0.0f;
        for (int j = i + 1; j < N; ++j)
            sum += T[i * lda + j] * x[j];
        x[i] = (x[i] - sum) / T[i * lda + i];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N> [--no-pause]\n";
        return -1;
    }
    int n = std::stoi(argv[1]);
    bool noPause = (argc > 2 && std::string(argv[2]) == "--no-pause");

    std::vector<float> T(n * n, 0.0f), x(n, 1.0f);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            T[i * n + j] = (i == j) ? 2.0f : 0.1f;

    std::vector<float> xw(x);
    trsv_upper(n, T.data(), n, xw.data());  // warmup

    if (!noPause) {
        std::cout << "Ready. Press Enter to run TRSV...\n";
        std::cin.get();
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    trsv_upper(n, T.data(), n, x.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = std::chrono::duration<double>(t1 - t0).count() * 1e6;
    std::cout << "SUCCESS: TRSV N=" << n << "  time=" << us << " us\n";
    return 0;
}
