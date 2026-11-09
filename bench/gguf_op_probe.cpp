// Actual GGUF weight, installed ggml Metal MUL_MAT. Never writes model bytes.
// GPU execution requires --run-metal and the shared serialized GPU supervisor.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <gguf.h>
#include <CommonCrypto/CommonDigest.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static std::string quoted(const std::string & s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else o << c;
    }
    return o.str() + '"';
}
static std::string sha256(const std::vector<uint8_t> & bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), digest);
    std::ostringstream o;
    for (auto b : digest) o << std::hex << std::setw(2) << std::setfill('0') << int(b);
    return o.str();
}
static int bounded_int(const std::string & s, int maximum) {
    size_t pos = 0; long value = std::stol(s, &pos);
    require(pos == s.size() && value >= 1 && value <= maximum, "integer outside bounds");
    return int(value);
}
struct File {
    int fd;
    explicit File(const std::string & path) : fd(open(path.c_str(), O_RDONLY)) { require(fd >= 0, "cannot open model read-only"); }
    ~File() { close(fd); }
};
static bool same_file(const struct stat & a, const struct stat & b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
        a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
        a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
}
static std::vector<uint8_t> read_bytes(int fd, uint64_t offset, size_t size) {
    std::vector<uint8_t> data(size);
    size_t done = 0;
    while (done < size) {
        ssize_t n = pread(fd, data.data() + done, size - done, off_t(offset + done));
        require(n > 0, "short tensor read"); done += size_t(n);
    }
    return data;
}
int main(int argc, char ** argv) try {
    std::string model, tensor, metal;
    int m = 1, iters = 10, evict_mib = 0;
    bool run = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--run-metal") { run = true; continue; }
        if (key == "--help") {
            std::cout << "--model FILE --tensor NAME [--m 1..512] [--iters 1..100] [--evict-mib 0|128] [--metal-lib FILE --run-metal]\n"
                         "Without --run-metal: metadata only. With it: use shared GPU supervisor.\n";
            return 0;
        }
        require(i + 1 < argc, "missing option value"); std::string value = argv[++i];
        if (key == "--model") model = value;
        else if (key == "--tensor") tensor = value;
        else if (key == "--metal-lib") metal = value;
        else if (key == "--m") m = bounded_int(value, 512);
        else if (key == "--iters") iters = bounded_int(value, 100);
        else if (key == "--evict-mib") {
            require(value == "0" || value == "128", "--evict-mib must be 0 or 128");
            evict_mib = value == "128" ? 128 : 0;
        }
        else throw std::runtime_error("unknown option: " + key);
    }
    require(!model.empty() && !tensor.empty(), "--model and --tensor required");
    require(!run || !metal.empty(), "GPU run requires explicit --metal-lib provenance");
    File file(model); struct stat before{}, after{};
    require(fstat(file.fd, &before) == 0 && S_ISREG(before.st_mode), "model must be a regular file");
    // Parse through a duplicate of the same read-only descriptor; allocate no model tensors.
    FILE * stream = fdopen(dup(file.fd), "rb"); require(stream, "fdopen failed");
    gguf_init_params gp{true, nullptr};
    auto raw = gguf_init_from_file_ptr(stream, gp); fclose(stream);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> header(raw, gguf_free);
    require(bool(header), "invalid GGUF header");
    int64_t id = gguf_find_tensor(header.get(), tensor.c_str()); require(id >= 0, "tensor not found");
    const auto ne = gguf_get_tensor_ne(header.get(), id);
    const int64_t k = ne[0], n = ne[1];
    require(ne[2] == 1 && ne[3] == 1 && k > 0 && n > 0 && k <= 32768 && n <= 32768,
            "only bounded 2D projection tensors admitted (K,N<=32768)");
    auto type = gguf_get_tensor_type(header.get(), id);
    require(ggml_is_quantized(type), "baseline requires an actual quantized tensor");
    const size_t size = gguf_get_tensor_size(header.get(), id);
    require(size > 0 && size <= 256ULL * 1024 * 1024, "selected tensor exceeds 256 MiB cap");
    const uint64_t data_offset = gguf_get_data_offset(header.get());
    const uint64_t relative_offset = gguf_get_tensor_offset(header.get(), id);
    require(data_offset <= uint64_t(before.st_size) && relative_offset <= uint64_t(before.st_size) - data_offset,
            "tensor offset outside file");
    const uint64_t offset = data_offset + relative_offset;
    require(size <= uint64_t(before.st_size) - offset, "tensor extends beyond file");
    require(ggml_row_size(type, k) * size_t(n) == size, "unexpected packed matrix size");
    Dl_info library{}; dladdr(reinterpret_cast<void *>(&ggml_init), &library);
    std::cout << "{\"event\":\"metadata\",\"model\":" << quoted(model) << ",\"tensor\":" << quoted(tensor)
              << ",\"K\":" << k << ",\"N\":" << n << ",\"M\":" << m << ",\"type\":" << quoted(ggml_type_name(type))
              << ",\"tensor_bytes\":" << size << ",\"file_bytes\":" << before.st_size
              << ",\"data_offset\":" << offset << ",\"ggml_library\":" << quoted(library.dli_fname ? library.dli_fname : "unknown")
              << ",\"metal_library\":" << quoted(metal) << ",\"evict_mib\":" << evict_mib << "}" << std::endl;
    if (!run) return 0;
    auto packed = read_bytes(file.fd, offset, size);
    const std::string weight_hash = sha256(packed);
    auto traits = ggml_get_type_traits(type); require(traits && traits->to_float, "no CPU dequantization reference");
    // Values are exactly fp16-representable. Flat order is [M,K], shared with EXL3 probes.
    std::vector<float> input(size_t(m) * k);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((uint64_t(i) * 17 + 13) % 257) - 128) / 128.0f;
    // Independent CPU scalar dot products over ggml's CPU block decoder: evenly spaced
    // output channels and input rows. No Metal arithmetic or quantized CPU matmul oracle.
    const int check_n = int(std::min<int64_t>(n, 64)), check_m = std::min(m, 8);
    std::vector<double> reference(size_t(check_n) * check_m);
    std::vector<float> decoded(k);
    for (int j = 0; j < check_n; ++j) {
        int64_t channel = check_n == 1 ? 0 : int64_t(j) * (n - 1) / (check_n - 1);
        traits->to_float(packed.data() + channel * ggml_row_size(type, k), decoded.data(), k);
        for (int r = 0; r < check_m; ++r) {
            int row = check_m == 1 ? 0 : r * (m - 1) / (check_m - 1);
            double sum = 0;
            for (int64_t q = 0; q < k; ++q) sum += double(decoded[q]) * input[size_t(row) * k + q];
            require(std::isfinite(sum), "nonfinite CPU reference"); reference[r * check_n + j] = sum;
        }
    }
    auto reg = ggml_backend_load(metal.c_str()); require(reg, "cannot load explicit Metal backend library");
    const std::string registration_name = ggml_backend_reg_name(reg);
    require(registration_name == "Metal" || registration_name == "MTL", "explicit library is not the Metal backend");
    require(ggml_backend_reg_dev_count(reg) > 0, "Metal backend has no device");
    auto dev = ggml_backend_reg_dev_get(reg, 0);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    require(bool(backend), "Metal backend initialization failed");
    ggml_init_params cp{8 * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(16, false) + 4096, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(cp), ggml_free); require(bool(ctx), "context allocation failed");
    auto w = ggml_new_tensor_2d(ctx.get(), type, k, n);
    auto x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    auto y = ggml_mul_mat(ctx.get(), w, x); ggml_set_output(y);
    require(ggml_backend_supports_op(backend.get(), y), "Metal does not support selected MUL_MAT");
    auto graph = ggml_new_graph_custom(ctx.get(), 16, false); ggml_build_forward_expand(graph, y);
    // Distinct resident input/output: 128 MiB read + 128 MiB write per sweep.
    // This treats cache residency; it does not establish a perfectly cold cache.
    const size_t eviction_bytes = size_t(evict_mib) * 1024 * 1024;
    ggml_tensor * eviction_input = nullptr;
    ggml_tensor * eviction_output = nullptr;
    ggml_cgraph * eviction_graph = nullptr;
    if (eviction_bytes) {
        eviction_input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, eviction_bytes / sizeof(float));
        eviction_output = ggml_scale(ctx.get(), eviction_input, 1.000001f);
        ggml_set_output(eviction_output);
        require(ggml_backend_supports_op(backend.get(), eviction_output), "Metal does not support eviction SCALE");
        eviction_graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(eviction_graph, eviction_output);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "Metal buffer allocation failed");
    ggml_backend_tensor_set(w, packed.data(), 0, size);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (eviction_input) {
        // Initialize once without retaining a second host-sized 128 MiB array.
        std::vector<float> chunk(256 * 1024, 0.25f);
        const size_t chunk_bytes = chunk.size() * sizeof(float);
        for (size_t offset = 0; offset < eviction_bytes; offset += chunk_bytes)
            ggml_backend_tensor_set(eviction_input, chunk.data(), offset, std::min(chunk_bytes, eviction_bytes - offset));
    }
    std::vector<float> output(size_t(m) * n), poison(output.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<double> times;
    double worst_rms = 0, worst_max = 0;
    for (int trial = -3; trial < iters; ++trial) {
        ggml_backend_tensor_set(y, poison.data(), 0, poison.size() * sizeof(float));
        ggml_backend_synchronize(backend.get());
        if (eviction_graph) {
            auto eviction_status = ggml_backend_graph_compute(backend.get(), eviction_graph);
            ggml_backend_synchronize(backend.get());
            require(eviction_status == GGML_STATUS_SUCCESS, "Metal eviction graph failed");
        }
        auto start = std::chrono::steady_clock::now();
        auto status = ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        require(status == GGML_STATUS_SUCCESS, "Metal graph execution failed");
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        for (float value : output) require(std::isfinite(value), "nonfinite or unwritten output");
        double err2 = 0, ref2 = 0, errmax = 0, refmax = 0;
        for (int r = 0; r < check_m; ++r) for (int j = 0; j < check_n; ++j) {
            int row = check_m == 1 ? 0 : r * (m - 1) / (check_m - 1);
            int64_t channel = check_n == 1 ? 0 : int64_t(j) * (n - 1) / (check_n - 1);
            double want = reference[r * check_n + j], err = output[size_t(row) * n + channel] - want;
            err2 += err * err; ref2 += want * want; errmax = std::max(errmax, std::abs(err)); refmax = std::max(refmax, std::abs(want));
        }
        double rms = std::sqrt(err2 / std::max(ref2, 1e-30)), maxerr = errmax / std::max(refmax, 1e-30);
        worst_rms = std::max(worst_rms, rms); worst_max = std::max(worst_max, maxerr);
        require(rms < 0.003 && maxerr < 0.004, "CPU reference tolerance failed");
        if (trial >= 0) times.push_back(ms);
    }
    if (eviction_output) {
        float first = 0, last = 0;
        ggml_backend_tensor_get(eviction_output, &first, 0, sizeof(float));
        ggml_backend_tensor_get(eviction_output, &last, eviction_bytes - sizeof(float), sizeof(float));
        const float expected = 0.25f * 1.000001f;
        require(first == expected && last == expected, "eviction graph endpoint check failed");
    }
    require(fstat(file.fd, &after) == 0 && same_file(before, after), "model identity/stat changed during probe");
    require(sha256(read_bytes(file.fd, offset, size)) == weight_hash, "selected source tensor changed during probe");
    auto sorted = times; std::sort(sorted.begin(), sorted.end());
    double median = sorted.size() % 2 ? sorted[sorted.size()/2] : (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]) / 2;
    std::cout << std::setprecision(10) << "{\"event\":\"result\",\"status\":\"passed\",\"backend\":" << quoted(ggml_backend_name(backend.get()))
              << ",\"registration_name\":" << quoted(registration_name)
              << ",\"tensor_sha256\":" << quoted(weight_hash) << ",\"median_ms\":" << median
              << ",\"worst_relative_rms\":" << worst_rms << ",\"worst_relative_max\":" << worst_max
              << ",\"reference_channels\":" << check_n << ",\"reference_rows\":" << check_m
              << ",\"eviction_input_bytes\":" << eviction_bytes << ",\"eviction_output_bytes\":" << eviction_bytes
              << ",\"cache_treatment\":" << quoted(eviction_bytes ? "distinct-buffer SCALE sweep before every call; not a proven cold cache" : "no eviction sweep")
              << ",\"buffer_bytes\":" << ggml_backend_buffer_get_size(buffer.get()) << ",\"times_ms\":[";
    for (size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
    std::cout << "],\"timing_scope\":\"resident graph dispatch plus synchronize; reusable output; excludes loading, allocation, transfers, poisoning, eviction sweep and validation\"}" << std::endl;
    return 0;
} catch (const std::exception & e) {
    std::cerr << "{\"event\":\"error\",\"message\":" << quoted(e.what()) << "}" << std::endl; return 1;
}
