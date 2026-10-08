// CUDA kernels for batched lattice spin-model Monte Carlo.
//
// Layout: R independent replicas (usually one per temperature) of an L^D
// periodic hypercubic lattice, stored back to back. Site i of replica r lives
// at global index g = r*N + i (times NC for vector spins). Neighbors are
// computed arithmetically from the C-order site index, so the same kernels
// work in any dimension.
//
// Random numbers: Philox4x32-10 counter-based generator (Salmon et al. 2011).
// Every draw is a pure function of (counter = thread/site id, launch step,
// tag; key = seed), so no RNG state is stored and results are reproducible.

typedef unsigned int u32;
typedef unsigned long long u64;

#define MAX_Z 8  // supports D <= 4

// ---------------------------------------------------------------------------
// Philox4x32-10
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 philox(u32 c0, u32 c1, u32 c2, u32 c3, u32 k0, u32 k1) {
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        u32 hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        u32 hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return make_uint4(c0, c1, c2, c3);
}

__device__ __forceinline__ uint4 rng(u64 counter, u64 step, u32 tag, u32 k0, u32 k1) {
    return philox((u32)counter, (u32)(counter >> 32) ^ tag, (u32)step, (u32)(step >> 32), k0, k1);
}

// Uniform in [0, 1) and (0, 1] with 24-bit resolution.
__device__ __forceinline__ float u01(u32 x) { return (x >> 8) * (1.0f / 16777216.0f); }
__device__ __forceinline__ float u01_open(u32 x) { return ((x >> 8) + 1) * (1.0f / 16777216.0f); }

__device__ __forceinline__ void box_muller(u32 a, u32 b, float &g0, float &g1) {
    float r = sqrtf(-2.0f * logf(u01_open(a)));
    float s, c;
    sincospif(2.0f * u01(b), &s, &c);
    g0 = r * c;
    g1 = r * s;
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------
// Fill nb[2a] (+a neighbor) and nb[2a+1] (-a neighbor) of site i.
__device__ __forceinline__ void neighbors(int i, int L, int D, int *nb) {
    int stride = 1;
    for (int a = D - 1; a >= 0; --a) {
        int x = (i / stride) % L;
        nb[2 * a] = (x == L - 1) ? i - (L - 1) * stride : i + stride;
        nb[2 * a + 1] = (x == 0) ? i + (L - 1) * stride : i - stride;
        stride *= L;
    }
}

// Forward (+a) neighbor of site i along axis a.
__device__ __forceinline__ int forward(int i, int a, int L, int D) {
    int stride = 1;
    for (int b = D - 1; b > a; --b) stride *= L;
    int x = (i / stride) % L;
    return (x == L - 1) ? i - (L - 1) * stride : i + stride;
}

// k-th site (0 <= k < N/2) of checkerboard sublattice `color` (L even).
// Sites 2k and 2k+1 differ only in the last coordinate, so exactly one of
// them has the requested parity.
__device__ __forceinline__ int checkerboard_site(int k, int color, int L, int D) {
    int i = 2 * k, rem = i, parity = 0;
    for (int a = 0; a < D; ++a) {
        parity += rem % L;
        rem /= L;
    }
    return i + ((parity & 1) != color);
}

// ---------------------------------------------------------------------------
// Checkerboard local updates (one sublattice per launch)
//
// Ising/Potts use heat-bath (Gibbs) updates. Checkerboard *Metropolis* with
// discrete spins is not ergodic: moves with dE = 0 are accepted with
// probability 1, so e.g. in 1D every domain wall moves deterministically and
// the difference between left- and right-moving walls is conserved, which
// biases averages. Heat-bath samples each spin from its exact conditional
// distribution and has no deterministic moves.
// ---------------------------------------------------------------------------
#define MAX_Q 32

extern "C" __global__ void ising_heatbath(signed char *s, int R, int N, int L, int D, int color,
                                          const float *beta, float J, float h,
                                          u32 k0, u32 k1, u64 step) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int Nh = N / 2;
    if (t >= (long long)R * Nh) return;
    int r = t / Nh;
    int i = checkerboard_site(t - (long long)r * Nh, color, L, D);
    signed char *sr = s + (long long)r * N;
    int nb[MAX_Z];
    neighbors(i, L, D, nb);
    int sum = 0;
    for (int k = 0; k < 2 * D; ++k) sum += sr[nb[k]];
    // P(s_i = +1 | neighbors) = 1 / (1 + exp(-2 beta (J sum + h)))
    float p_up = 1.0f / (1.0f + __expf(-2.0f * beta[r] * (J * sum + h)));
    sr[i] = (u01(rng(t, step, 0, k0, k1).x) < p_up) ? 1 : -1;
}

extern "C" __global__ void potts_heatbath(signed char *s, int R, int N, int L, int D, int color, int q,
                                          const float *beta, float J, float h,
                                          u32 k0, u32 k1, u64 step) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int Nh = N / 2;
    if (t >= (long long)R * Nh) return;
    int r = t / Nh;
    int i = checkerboard_site(t - (long long)r * Nh, color, L, D);
    signed char *sr = s + (long long)r * N;
    int nb[MAX_Z];
    neighbors(i, L, D, nb);
    // Local energy of each candidate state is -(J n_s + h delta_s0); only the
    // (at most 2D) states present among the neighbors, plus state 0, differ
    // from the baseline weight exp(0) = 1.
    float w[MAX_Q];
    for (int c = 0; c < q; ++c) w[c] = (c == 0) ? beta[r] * h : 0.0f;
    for (int k = 0; k < 2 * D; ++k) w[sr[nb[k]]] += beta[r] * J;
    float wmax = w[0];
    for (int c = 1; c < q; ++c) wmax = fmaxf(wmax, w[c]);
    float total = 0.0f;
    for (int c = 0; c < q; ++c) {
        w[c] = __expf(w[c] - wmax);
        total += w[c];
    }
    float u = u01(rng(t, step, 0, k0, k1).x) * total;
    int c = 0;
    while (c < q - 1 && u >= w[c]) {
        u -= w[c];
        ++c;
    }
    sr[i] = c;
}

// O(n) model: proposal S' = normalize(S + delta * gaussian).
template <int NC>
__global__ void on_metropolis(float *s, int R, int N, int L, int D, int color,
                              const float *beta, const float *delta, float J, float h,
                              u32 k0, u32 k1, u64 step, unsigned char *accepted) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int Nh = N / 2;
    if (t >= (long long)R * Nh) return;
    int r = t / Nh;
    int i = checkerboard_site(t - (long long)r * Nh, color, L, D);
    float *sr = s + (long long)r * N * NC;

    uint4 a = rng(t, step, 0, k0, k1);
    uint4 b = rng(t, step, 1, k0, k1);
    float g[4];
    box_muller(a.x, a.y, g[0], g[1]);
    box_muller(a.z, a.w, g[2], g[3]);

    float old[NC], nw[NC], field[NC];
    float norm2 = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        old[c] = sr[i * NC + c];
        nw[c] = old[c] + delta[r] * g[c];
        norm2 += nw[c] * nw[c];
        field[c] = 0.0f;
    }
    float inv = rsqrtf(norm2);
    int nb[MAX_Z];
    neighbors(i, L, D, nb);
    for (int k = 0; k < 2 * D; ++k) {
#pragma unroll
        for (int c = 0; c < NC; ++c) field[c] += sr[nb[k] * NC + c];
    }
    float dE = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        nw[c] *= inv;
        dE -= J * (nw[c] - old[c]) * field[c];
    }
    dE -= h * (nw[0] - old[0]);
    bool acc = dE <= 0.0f || u01(b.x) < __expf(-beta[r] * dE);
    if (acc) {
#pragma unroll
        for (int c = 0; c < NC; ++c) sr[i * NC + c] = nw[c];
    }
    if (accepted) accepted[t] = acc;
}

// ---------------------------------------------------------------------------
// Swendsen-Wang: bond activation -> union-find labelling -> cluster update
// ---------------------------------------------------------------------------
// Bonds are stored as a bitmask per site: bit a = bond to the +a neighbor.

extern "C" __global__ void ising_sw_bonds(const signed char *s, unsigned char *bonds, int R, int N, int L, int D,
                                          const float *beta, float J, u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= (long long)R * N) return;
    int r = g / N, i = g - (long long)r * N;
    const signed char *sr = s + (long long)r * N;
    float p = 1.0f - __expf(-2.0f * beta[r] * J);
    uint4 rn = rng(g, step, 0, k0, k1);
    u32 u[4] = {rn.x, rn.y, rn.z, rn.w};
    unsigned char mask = 0;
    for (int a = 0; a < D; ++a) {
        if (sr[forward(i, a, L, D)] == sr[i] && u01(u[a]) < p) mask |= (1 << a);
    }
    bonds[g] = mask;
}

extern "C" __global__ void potts_sw_bonds(const signed char *s, unsigned char *bonds, int R, int N, int L, int D,
                                          const float *beta, float J, u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= (long long)R * N) return;
    int r = g / N, i = g - (long long)r * N;
    const signed char *sr = s + (long long)r * N;
    float p = 1.0f - __expf(-beta[r] * J);
    uint4 rn = rng(g, step, 0, k0, k1);
    u32 u[4] = {rn.x, rn.y, rn.z, rn.w};
    unsigned char mask = 0;
    for (int a = 0; a < D; ++a) {
        if (sr[forward(i, a, L, D)] == sr[i] && u01(u[a]) < p) mask |= (1 << a);
    }
    bonds[g] = mask;
}

// Embedded-Ising bonds for O(n): project on the replica's random unit vector rvec.
template <int NC>
__global__ void on_sw_bonds(const float *s, const float *rvec, unsigned char *bonds, int R, int N, int L, int D,
                            const float *beta, float J, u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= (long long)R * N) return;
    int r = g / N, i = g - (long long)r * N;
    const float *sr = s + (long long)r * N * NC;
    const float *rv = rvec + r * NC;
    float pi = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) pi += sr[i * NC + c] * rv[c];
    uint4 rn = rng(g, step, 0, k0, k1);
    u32 u[4] = {rn.x, rn.y, rn.z, rn.w};
    unsigned char mask = 0;
    for (int a = 0; a < D; ++a) {
        int j = forward(i, a, L, D);
        float pj = 0.0f;
#pragma unroll
        for (int c = 0; c < NC; ++c) pj += sr[j * NC + c] * rv[c];
        float x = 2.0f * beta[r] * J * pi * pj;
        if (x > 0.0f && u01(u[a]) < 1.0f - __expf(-x)) mask |= (1 << a);
    }
    bonds[g] = mask;
}

// Union-find with atomicMin hooking (Playne & Hawick, IEEE TPDS 2018).
// Labels only ever decrease and roots satisfy label[x] == x; volatile reads
// make sure concurrent hooks by other threads are seen.
__device__ __forceinline__ int find_root(volatile int *label, int x) {
    int y = label[x];
    while (y != x) {
        x = y;
        y = label[x];
    }
    return x;
}

__device__ void unite(int *label, int a, int b) {
    bool done = false;
    while (!done) {
        a = find_root(label, a);
        b = find_root(label, b);
        if (a < b) {
            int old = atomicMin(&label[b], a);
            done = (old == b);
            b = old;
        } else if (b < a) {
            int old = atomicMin(&label[a], b);
            done = (old == a);
            a = old;
        } else {
            done = true;
        }
    }
}

extern "C" __global__ void sw_init(int *label, long long total) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g < total) label[g] = (int)g;
}

extern "C" __global__ void sw_union(const unsigned char *bonds, int *label, int R, int N, int L, int D) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= (long long)R * N) return;
    unsigned char mask = bonds[g];
    if (!mask) return;
    int r = g / N, i = g - (long long)r * N;
    for (int a = 0; a < D; ++a) {
        if (mask & (1 << a)) unite(label, (int)g, r * N + forward(i, a, L, D));
    }
}

extern "C" __global__ void sw_flatten(int *label, long long total) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g < total) label[g] = find_root(label, (int)g);
}

// Each cluster's random choice is drawn from the RNG keyed by its root
// index, so every site of the cluster computes the same decision.
extern "C" __global__ void ising_sw_apply(signed char *s, const int *label, long long total,
                                          u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= total) return;
    if (rng(label[g], step, 2, k0, k1).x & 1u) s[g] = -s[g];
}

extern "C" __global__ void potts_sw_apply(signed char *s, const int *label, long long total, int q,
                                          u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= total) return;
    s[g] = min((int)(u01(rng(label[g], step, 2, k0, k1).x) * q), q - 1);
}

template <int NC>
__global__ void on_sw_apply(float *s, const float *rvec, const int *label, int R, int N,
                            u32 k0, u32 k1, u64 step) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= (long long)R * N) return;
    if (!(rng(label[g], step, 2, k0, k1).x & 1u)) return;
    int r = g / N;
    const float *rv = rvec + r * NC;
    float *sg = s + g * NC;
    float p = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) p += sg[c] * rv[c];
#pragma unroll
    for (int c = 0; c < NC; ++c) sg[c] -= 2.0f * p * rv[c];
}

// Restore |S| = 1 (float32 drift after many reflections).
template <int NC>
__global__ void on_normalize(float *s, long long total) {
    long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (g >= total) return;
    float *sg = s + g * NC;
    float n2 = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) n2 += sg[c] * sg[c];
    float inv = rsqrtf(n2);
#pragma unroll
    for (int c = 0; c < NC; ++c) sg[c] *= inv;
}

// ---------------------------------------------------------------------------
// Fused measurement: per-replica sums in one pass.
//   out[r*K + 0] = bond sum      (Ising: s_i s_j, Potts: delta, O(n): S_i.S_j)
//   out[r*K + 1] = field sum     (Ising: s_i,     Potts: delta(s,0), O(n): S^x)
//   out[r*K + 2..4] = order-parameter vector components (summed over sites)
//   out[r*K + 5..8] = O(n), n=2: sum_i (S_i x S_{i+a})_z for each axis a
// Launch with grid (blocks, R); per-warp shuffle reduction, then double atomics.
// ---------------------------------------------------------------------------
#define MEAS_K 10

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ void flush(float *acc, double *out, int r) {
#pragma unroll
    for (int k = 0; k < MEAS_K; ++k) {
        float v = warp_sum(acc[k]);
        if ((threadIdx.x & 31) == 0 && v != 0.0f) atomicAdd(&out[r * MEAS_K + k], (double)v);
    }
}

extern "C" __global__ void ising_measure(const signed char *s, double *out, int N, int L, int D) {
    int r = blockIdx.y;
    const signed char *sr = s + (long long)r * N;
    float acc[MEAS_K] = {0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int si = sr[i];
        int b = 0;
        for (int a = 0; a < D; ++a) b += sr[forward(i, a, L, D)];
        acc[0] += si * b;
        acc[1] += si;
        acc[2] += si;
    }
    flush(acc, out, r);
}

extern "C" __global__ void potts_measure(const signed char *s, double *out, int N, int L, int D, int q) {
    int r = blockIdx.y;
    const signed char *sr = s + (long long)r * N;
    float acc[MEAS_K] = {0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int si = sr[i];
        for (int a = 0; a < D; ++a) acc[0] += (sr[forward(i, a, L, D)] == si);
        acc[1] += (si == 0);
        float sn, cs;
        sincospif(2.0f * si / q, &sn, &cs);
        acc[2] += cs;
        acc[3] += sn;
    }
    flush(acc, out, r);
}

template <int NC>
__global__ void on_measure(const float *s, double *out, int N, int L, int D) {
    int r = blockIdx.y;
    const float *sr = s + (long long)r * N * NC;
    float acc[MEAS_K] = {0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float si[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            si[c] = sr[i * NC + c];
            acc[2 + c] += si[c];
        }
        acc[1] += si[0];
        for (int a = 0; a < D; ++a) {
            const float *sj = sr + forward(i, a, L, D) * NC;
            float dot = 0.0f;
#pragma unroll
            for (int c = 0; c < NC; ++c) dot += si[c] * sj[c];
            acc[0] += dot;
            if (NC == 2 && a < 4) acc[5 + a] += si[0] * sj[1] - si[1] * sj[0];
        }
    }
    flush(acc, out, r);
}
