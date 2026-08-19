/* snt_trellis.c -- portable fp32 reference of the 995,058-parameter
 * Trellis-RIFT English text-to-wave model. Plain C99 + libm.
 *
 * Semantics mirror tools/trellis_numpy_reference.py operator for operator;
 * docs/trellis-web-port-operators.md is the prose spec. Section numbers in the
 * comments below refer to that document.
 *
 * Deliberate, documented deviations from a literal reading of the spec:
 *
 *  1. The duration blocks' `* mask` multiplies (§3.1) are omitted. Inference
 *     always builds mask = ones([N]) (see duration_forward in the NumPy
 *     reference), and `x * 1.0f` is a bit-exact identity for every float, so
 *     the omission cannot change a single output bit. Reinstate the multiply
 *     if this port ever grows batched/padded inference.
 *  2. The DC blocker (§3.7) is evaluated as a double-precision overlap-add FFT
 *     convolution with a fixed 8192-point transform instead of one transform of
 *     next_pow2(S + 4095). Both compute the same 4096-tap linear convolution;
 *     the fixed size bounds the working set at ~200 KB instead of 268 MB for a
 *     worst-case 177 s utterance. The difference is double-precision rounding,
 *     far below the float32 output's last bit.
 *  3. The noise PRNG (§4) reproduces the MT19937 uint32 stream, the 24-bit
 *     uniform mapping and the normal_fill block layout exactly, and evaluates
 *     log/sqrt/cos/sin with plain float32 libm as §4.3 prescribes. Note that
 *     the committed numpy_decoder_noise.npy is NOT a bit-exact target for any
 *     libm: it was produced by NumPy's x86-64 float32 SIMD log/sin/cos, which
 *     already disagrees with NumPy's own arm64 build at 1 ULP on ~12 % of the
 *     values. 1 ULP here is 2.4e-7 absolute, i.e. the same magnitude the spec
 *     already measured between PyTorch and NumPy.
 */
#include "snt_trellis.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define T_PI 3.14159265358979323846
#define T_SQRT2 1.41421356237309504880

/* ---- little-endian readers -------------------------------------------- */

static int32_t rd_i32(const unsigned char *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* ---- tensor layout ------------------------------------------------------
 * Index order is exactly the sequence in bundle/trellis_rift_995058_meta.json
 * (built by build_front_spec / build_decoder_spec in
 * tools/export_trellis_bundle.py). Every index below is derived from the
 * meta.bin header rather than hard-coded, and init() checks each tensor's
 * float count against the shape the header implies. */

/* one TrellisResidualBlock occupies 13 slots: scale, then two
 * FactorizedTemporalConv1d (reduce w/b, temporal w/b, expand w/b) */
#define T_BLOCK_SLOTS 13
/* one decoder ConvNeXt-style block occupies 11 slots */
#define T_DEC_BLOCK_SLOTS 11

static int t_dur_block(const snt_trellis_model *m, int i) {
    (void)m;
    return 3 + T_BLOCK_SLOTS * i;
}
static int t_dur_out(const snt_trellis_model *m) {
    return 3 + T_BLOCK_SLOTS * m->duration_depth;
}
static int t_aco_emb(const snt_trellis_model *m) { return t_dur_out(m) + 2; }
static int t_tok_block(const snt_trellis_model *m, int i) {
    return t_aco_emb(m) + 3 + T_BLOCK_SLOTS * i;
}
static int t_frm_proj(const snt_trellis_model *m) {
    return t_aco_emb(m) + 3 + T_BLOCK_SLOTS * m->token_depth;
}
static int t_frm_block(const snt_trellis_model *m, int i) {
    return t_frm_proj(m) + 2 + T_BLOCK_SLOTS * i;
}
static int t_head(const snt_trellis_model *m) {
    return t_frm_proj(m) + 2 + T_BLOCK_SLOTS * m->frame_depth;
}
static int t_dec(const snt_trellis_model *m) { return t_head(m) + 12; }
static int t_dec_block(const snt_trellis_model *m, int i) {
    return t_dec(m) + 6 + T_DEC_BLOCK_SLOTS * i;
}
static int t_dec_final(const snt_trellis_model *m) {
    return t_dec(m) + 6 + T_DEC_BLOCK_SLOTS * m->decoder_num_layers;
}

/* Append the expected float count of every tensor, in bundle order.
 * Returns the tensor count, or -1 on overflow. */
static int t_fill_expect(const snt_trellis_model *m, long *expect, int cap) {
    int n = 0, i, j;
    const int V = m->vocab_size;
    const int DW = m->duration_width, DR = m->duration_rank,
              DK = m->duration_kernel_size;
    const int AW = m->acoustic_width, AR = m->acoustic_rank,
              AK = m->acoustic_kernel_size;
    const int W = m->decoder_width, R = m->decoder_pointwise_rank;
    const int HID = m->decoder_width * m->decoder_expand;
    const int BINS = m->n_fft / 2 + 1;

#define PUSH(v)                        \
    do {                               \
        if (n >= cap) return -1;       \
        expect[n++] = (long)(v);       \
    } while (0)
#define PUSH_BLOCK(C, RK, K)                       \
    do {                                           \
        PUSH(1);                    /* scale */    \
        for (j = 0; j < 2; j++) {                  \
            PUSH((long)(RK) * (C)); /* reduce.w */ \
            PUSH(RK);                              \
            PUSH((long)(RK) * (K)); /* temporal */ \
            PUSH(RK);                              \
            PUSH((long)(C) * (RK)); /* expand.w */ \
            PUSH(C);                               \
        }                                          \
    } while (0)

    PUSH((long)V * DW);                 /* duration.embedding.weight   */
    PUSH((long)DW * (DW + 3));          /* duration.input_proj.weight  */
    PUSH(DW);                           /* duration.input_proj.bias    */
    for (i = 0; i < m->duration_depth; i++) PUSH_BLOCK(DW, DR, DK);
    PUSH(DW);                           /* duration.output.weight [1,DW,1] */
    PUSH(1);                            /* duration.output.bias        */

    PUSH((long)V * AW);                 /* acoustic.embedding.weight   */
    PUSH((long)AW * (AW + 2));          /* acoustic.token_input_proj.w */
    PUSH(AW);
    for (i = 0; i < m->token_depth; i++) PUSH_BLOCK(AW, AR, AK);
    PUSH((long)AW * (AW + 3));          /* acoustic.frame_input_proj.w */
    PUSH(AW);
    for (i = 0; i < m->frame_depth; i++) PUSH_BLOCK(AW, AR, AK);

    PUSH((long)AW * m->head_kernel_size); /* head.shared_cell.depthwise.w */
    PUSH(AW);
    PUSH(AW);                             /* norm.weight */
    PUSH(AW);                             /* norm.bias   */
    PUSH((long)AW * m->head_expansion * AW); /* up.weight */
    PUSH((long)AW * m->head_expansion);      /* up.bias   */
    PUSH((long)AW * AW * m->head_expansion); /* down.weight */
    PUSH(AW);                             /* down.bias */
    PUSH(1);                              /* residual_scale */
    PUSH((long)m->head_steps * AW);       /* loop_bias */
    PUSH((long)m->n_mels * AW * m->output_kernel_size); /* head.output.w */
    PUSH(m->n_mels);

    PUSH((long)W * m->n_mels * SNT_TRELLIS_DECODER_KERNEL); /* embed.weight */
    PUSH(W);
    PUSH((long)W * m->decoder_noise_channels * SNT_TRELLIS_DECODER_KERNEL);
    PUSH(W);
    PUSH(W);                              /* norm.weight */
    PUSH(W);                              /* norm.bias   */
    for (i = 0; i < m->decoder_num_layers; i++) {
        PUSH((long)W * SNT_TRELLIS_DECODER_KERNEL); /* dwconv.weight */
        PUSH(W);
        PUSH(W);                          /* norm.weight */
        PUSH(W);                          /* norm.bias   */
        PUSH((long)R * W);                /* pwconv1_in.weight  */
        PUSH((long)HID * R);              /* pwconv1_out.weight */
        PUSH(HID);                        /* pwconv1_out.bias   */
        PUSH((long)R * HID);              /* pwconv2_in.weight  */
        PUSH((long)W * R);                /* pwconv2_out.weight */
        PUSH(W);                          /* pwconv2_out.bias   */
        PUSH(W);                          /* gamma              */
    }
    PUSH(W);                              /* final_norm.weight */
    PUSH(W);                              /* final_norm.bias   */
    PUSH((long)2 * BINS * W);             /* head.weight */
    PUSH((long)2 * BINS);                 /* head.bias   */
#undef PUSH_BLOCK
#undef PUSH
    return n;
}

int snt_trellis_bundle_split(const void *blob, size_t blob_bytes,
                             size_t *meta_bytes, size_t *weight_floats) {
    const unsigned char *p = (const unsigned char *)blob;
    int32_t n_tensors, last_off, last_size;
    size_t header, total;
    if (!p || !meta_bytes || !weight_floats) return -1;
    if (blob_bytes < (size_t)SNT_TRELLIS_HEADER_FIELDS * 4) return -2;
    if (rd_i32(p) != (int32_t)SNT_TRELLIS_MAGIC) return -3;
    if (rd_i32(p + 4) != SNT_TRELLIS_VERSION) return -3;
    n_tensors = rd_i32(p + 29 * 4);
    if (n_tensors <= 0 || n_tensors > SNT_TRELLIS_MAX_TENSORS) return -4;
    header = (size_t)SNT_TRELLIS_HEADER_FIELDS * 4 + (size_t)n_tensors * 8;
    if (blob_bytes < header) return -2;
    last_off = rd_i32(p + header - 8);
    last_size = rd_i32(p + header - 4);
    if (last_off < 0 || last_size <= 0) return -5;
    total = (size_t)last_off + (size_t)last_size;
    if (blob_bytes < header + total * 4) return -5;
    *meta_bytes = header;
    *weight_floats = total;
    return 0;
}

int snt_trellis_init(snt_trellis_model *m, const void *meta, size_t meta_bytes,
                     const float *weights, size_t weight_floats) {
    const unsigned char *p = (const unsigned char *)meta;
    long expect[SNT_TRELLIS_MAX_TENSORS];
    int i, want;
    if (!m || !p || !weights) return -1;
    if (meta_bytes < (size_t)SNT_TRELLIS_HEADER_FIELDS * 4) return -2;
    memset(m, 0, sizeof *m);
    m->magic = rd_i32(p + 0 * 4);
    m->version = rd_i32(p + 1 * 4);
    if (m->magic != (int32_t)SNT_TRELLIS_MAGIC) return -3;
    if (m->version != SNT_TRELLIS_VERSION) return -4;
    m->vocab_size = rd_i32(p + 2 * 4);
    m->duration_width = rd_i32(p + 3 * 4);
    m->duration_depth = rd_i32(p + 4 * 4);
    m->duration_kernel_size = rd_i32(p + 5 * 4);
    m->duration_rank = rd_i32(p + 6 * 4);
    m->acoustic_width = rd_i32(p + 7 * 4);
    m->token_depth = rd_i32(p + 8 * 4);
    m->frame_depth = rd_i32(p + 9 * 4);
    m->acoustic_kernel_size = rd_i32(p + 10 * 4);
    m->acoustic_rank = rd_i32(p + 11 * 4);
    m->n_mels = rd_i32(p + 12 * 4);
    m->head_steps = rd_i32(p + 13 * 4);
    m->head_kernel_size = rd_i32(p + 14 * 4);
    m->head_expansion = rd_i32(p + 15 * 4);
    m->output_kernel_size = rd_i32(p + 16 * 4);
    m->max_tokens = rd_i32(p + 17 * 4);
    m->min_duration = rd_i32(p + 18 * 4);
    m->max_duration = rd_i32(p + 19 * 4);
    m->decoder_width = rd_i32(p + 20 * 4);
    m->decoder_num_layers = rd_i32(p + 21 * 4);
    m->decoder_expand = rd_i32(p + 22 * 4);
    m->decoder_pointwise_rank = rd_i32(p + 23 * 4);
    m->decoder_noise_channels = rd_i32(p + 24 * 4);
    m->n_fft = rd_i32(p + 25 * 4);
    m->hop_length = rd_i32(p + 26 * 4);
    m->win_length = rd_i32(p + 27 * 4);
    m->sample_rate = rd_i32(p + 28 * 4);
    m->n_tensors = rd_i32(p + 29 * 4);
    m->front_floats = rd_i32(p + 30 * 4);
    m->decoder_floats = rd_i32(p + 31 * 4);

    if (m->vocab_size <= 0 || m->duration_width <= 0 || m->duration_depth <= 0 ||
        m->duration_rank <= 0 || m->acoustic_width <= 0 || m->token_depth <= 0 ||
        m->frame_depth <= 0 || m->acoustic_rank <= 0 || m->n_mels <= 0 ||
        m->head_steps <= 0 || m->head_expansion <= 0 || m->max_tokens <= 0 ||
        m->decoder_width <= 0 || m->decoder_num_layers <= 0 ||
        m->decoder_expand <= 0 || m->decoder_pointwise_rank <= 0 ||
        m->decoder_noise_channels <= 0 || m->sample_rate <= 0)
        return -5;
    if (m->duration_kernel_size <= 0 || m->duration_kernel_size % 2 == 0 ||
        m->acoustic_kernel_size <= 0 || m->acoustic_kernel_size % 2 == 0 ||
        m->head_kernel_size <= 0 || m->head_kernel_size % 2 == 0 ||
        m->output_kernel_size <= 0 || m->output_kernel_size % 2 == 0)
        return -5;
    if (m->min_duration < 1 || m->max_duration < m->min_duration) return -5;
    if (m->n_fft <= 0 || (m->n_fft & (m->n_fft - 1)) != 0) return -5;
    if (m->hop_length <= 0 || m->win_length != m->n_fft) return -5;

    want = t_fill_expect(m, expect, SNT_TRELLIS_MAX_TENSORS);
    if (want < 0) return -6;
    if (m->n_tensors != want) return -6;
    if (meta_bytes < (size_t)SNT_TRELLIS_HEADER_FIELDS * 4 +
                         (size_t)m->n_tensors * 8)
        return -2;
    for (i = 0; i < m->n_tensors; i++) {
        long off = rd_i32(p + SNT_TRELLIS_HEADER_FIELDS * 4 + 8 * i);
        long size = rd_i32(p + SNT_TRELLIS_HEADER_FIELDS * 4 + 8 * i + 4);
        if (off < 0 || size <= 0) return -7;
        if (size != expect[i]) return -8;
        if ((size_t)off + (size_t)size > weight_floats) return -9;
        m->w[i] = weights + off;
        m->wn[i] = (int32_t)size;
    }
    return 0;
}

/* ---- SHA-256 (§4.1 seed derivation) ------------------------------------ */

static const uint32_t T_SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static uint32_t t_rotr(uint32_t v, int n) {
    return (v >> n) | (v << (32 - n));
}

static void t_sha256_block(uint32_t *h, const unsigned char *block) {
    uint32_t w[64], a, b, c, d, e, f, g, hh, t1, t2;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[4 * i] << 24) | ((uint32_t)block[4 * i + 1] << 16) |
               ((uint32_t)block[4 * i + 2] << 8) | (uint32_t)block[4 * i + 3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = t_rotr(w[i - 15], 7) ^ t_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = t_rotr(w[i - 2], 17) ^ t_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0]; b = h[1]; c = h[2]; d = h[3];
    e = h[4]; f = h[5]; g = h[6]; hh = h[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = t_rotr(e, 6) ^ t_rotr(e, 11) ^ t_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t S0 = t_rotr(a, 2) ^ t_rotr(a, 13) ^ t_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        t1 = hh + S1 + ch + T_SHA_K[i] + w[i];
        t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

/* Streaming SHA-256 over `data`, writing 32 bytes to `digest`. */
static void t_sha256(const unsigned char *data, size_t len, unsigned char *digest) {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    unsigned char tail[128];
    size_t i, full = len / 64, rem = len % 64;
    size_t tail_len;
    uint64_t bits = (uint64_t)len * 8u;
    for (i = 0; i < full; i++) t_sha256_block(h, data + i * 64);
    memset(tail, 0, sizeof tail);
    memcpy(tail, data + full * 64, rem);
    tail[rem] = 0x80;
    tail_len = (rem < 56) ? 64u : 128u;
    for (i = 0; i < 8; i++)
        tail[tail_len - 1 - i] = (unsigned char)((bits >> (8 * i)) & 0xffu);
    t_sha256_block(h, tail);
    if (tail_len == 128) t_sha256_block(h, tail + 64);
    for (i = 0; i < 8; i++) {
        digest[4 * i] = (unsigned char)(h[i] >> 24);
        digest[4 * i + 1] = (unsigned char)(h[i] >> 16);
        digest[4 * i + 2] = (unsigned char)(h[i] >> 8);
        digest[4 * i + 3] = (unsigned char)h[i];
    }
}

/* ---- torch MT19937 + normal_fill (§4.1) -------------------------------- */

#define T_MT_N 624
#define T_MT_M 397
#define T_MT_MATRIX_A 0x9908b0dfu
#define T_MT_UMASK 0x80000000u
#define T_MT_LMASK 0x7fffffffu

typedef struct {
    uint32_t state[T_MT_N];
    int left;
    int next;
} t_mt19937;

static uint32_t t_mt_twist(uint32_t u, uint32_t v) {
    uint32_t mixed = (u & T_MT_UMASK) | (v & T_MT_LMASK);
    return (mixed >> 1) ^ ((v & 1u) ? T_MT_MATRIX_A : 0u);
}

/* ATen mt19937_engine::init_with_uint32 -- only the low 32 seed bits are used. */
static void t_mt_seed(t_mt19937 *g, uint64_t seed) {
    int i;
    g->state[0] = (uint32_t)(seed & 0xffffffffu);
    for (i = 1; i < T_MT_N; i++)
        g->state[i] = (uint32_t)(1812433253u *
                                     (g->state[i - 1] ^ (g->state[i - 1] >> 30)) +
                                 (uint32_t)i);
    g->left = 1;
    g->next = 0;
}

/* ATen mt19937_engine::next_state, in place and in the original scalar order. */
static void t_mt_next_state(t_mt19937 *g) {
    uint32_t *s = g->state;
    int i;
    g->left = T_MT_N;
    g->next = 0;
    for (i = 0; i < T_MT_N - T_MT_M; i++)
        s[i] = s[i + T_MT_M] ^ t_mt_twist(s[i], s[i + 1]);
    for (; i < T_MT_N - 1; i++)
        s[i] = s[i + T_MT_M - T_MT_N] ^ t_mt_twist(s[i], s[i + 1]);
    s[T_MT_N - 1] = s[T_MT_M - 1] ^ t_mt_twist(s[T_MT_N - 1], s[0]);
}

static uint32_t t_mt_random(t_mt19937 *g) {
    uint32_t y;
    if (--g->left <= 0) t_mt_next_state(g);
    y = g->state[g->next++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

/* at::uniform_real_distribution<float>: (raw & (2^24 - 1)) * 2^-24 */
static float t_mt_uniform(t_mt19937 *g) {
    return (float)(t_mt_random(g) & 0xffffffu) * (1.0f / 16777216.0f);
}

/* normal_fill_16 with mean 0, std 1, entirely in float32 (§4.1 step 3). */
static void t_normal_fill_16(float *d) {
    int j;
    for (j = 0; j < 8; j++) {
        float u1 = 1.0f - d[j];
        float u2 = d[j + 8];
        float radius = sqrtf(-2.0f * logf(u1));
        float theta = (float)(2.0 * T_PI) * u2;
        d[j] = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}

/* torch.randn(size, generator=Generator('cpu').manual_seed(seed)), float32. */
static int t_torch_randn(uint64_t seed, long size, float *out) {
    t_mt19937 g;
    long i;
    int j;
    if (size < 16) return -1; /* torch dispatches < 16 to a scalar path */
    t_mt_seed(&g, seed);
    for (i = 0; i < size; i++) out[i] = t_mt_uniform(&g);
    for (i = 0; i + 16 <= size; i += 16) t_normal_fill_16(out + i);
    if (size % 16 != 0) {
        float tail[16];
        for (j = 0; j < 16; j++) tail[j] = t_mt_uniform(&g);
        t_normal_fill_16(tail);
        for (j = 0; j < 16; j++) out[size - 16 + j] = tail[j];
    }
    return 0;
}

/* train_e6_e9_mel_refiner.seeded_noise: sha256(key + "_chunk00")[:8] big-endian,
 * masked to 63 bits. Writes [channels, frames] row-major. */
static int t_seeded_noise(const char *seed_key, int channels, int frames,
                          float *out) {
    unsigned char digest[32];
    unsigned char *buf;
    size_t klen = strlen(seed_key);
    static const char suffix[] = "_chunk00";
    uint64_t seed = 0;
    int i, rc;
    buf = (unsigned char *)malloc(klen + sizeof suffix);
    if (!buf) return -2;
    memcpy(buf, seed_key, klen);
    memcpy(buf + klen, suffix, sizeof suffix); /* includes the NUL, not hashed */
    t_sha256(buf, klen + sizeof suffix - 1, digest);
    free(buf);
    for (i = 0; i < 8; i++) seed = (seed << 8) | (uint64_t)digest[i];
    seed &= 0x7fffffffffffffffull;
    rc = t_torch_randn(seed, (long)channels * frames, out);
    return rc;
}

/* ---- radix-2 complex FFT (double) -------------------------------------- */

typedef struct {
    int n;
    int *rev;
    double *twr, *twi; /* forward twiddles e^{-2 pi i j / n}, j < n/2 */
} t_fft;

static void t_fft_free(t_fft *p) {
    if (!p) return;
    free(p->rev);
    free(p->twr);
    free(p->twi);
    p->rev = NULL;
    p->twr = NULL;
    p->twi = NULL;
    p->n = 0;
}

static int t_fft_init(t_fft *p, int n) {
    int i, j, bits = 0;
    memset(p, 0, sizeof *p);
    if (n < 2 || (n & (n - 1)) != 0) return -1;
    while ((1 << bits) < n) bits++;
    p->n = n;
    p->rev = (int *)malloc((size_t)n * sizeof(int));
    p->twr = (double *)malloc((size_t)(n / 2) * sizeof(double));
    p->twi = (double *)malloc((size_t)(n / 2) * sizeof(double));
    if (!p->rev || !p->twr || !p->twi) { t_fft_free(p); return -2; }
    for (i = 0; i < n; i++) {
        int r = 0, v = i;
        for (j = 0; j < bits; j++) { r = (r << 1) | (v & 1); v >>= 1; }
        p->rev[i] = r;
    }
    for (j = 0; j < n / 2; j++) {
        double angle = -2.0 * T_PI * (double)j / (double)n;
        p->twr[j] = cos(angle);
        p->twi[j] = sin(angle);
    }
    return 0;
}

/* In-place, unnormalized. inverse != 0 conjugates the twiddles. */
static void t_fft_run(const t_fft *p, double *re, double *im, int inverse) {
    const int n = p->n;
    int i, j, len;
    for (i = 0; i < n; i++) {
        j = p->rev[i];
        if (i < j) {
            double tr = re[i], ti = im[i];
            re[i] = re[j]; im[i] = im[j];
            re[j] = tr; im[j] = ti;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (i = 0; i < n; i += len) {
            for (j = 0; j < half; j++) {
                double wr = p->twr[j * step];
                double wi = inverse ? -p->twi[j * step] : p->twi[j * step];
                double xr = re[i + j + half], xi = im[i + j + half];
                double ar = xr * wr - xi * wi;
                double ai = xr * wi + xi * wr;
                re[i + j + half] = re[i + j] - ar;
                im[i + j + half] = im[i + j] - ai;
                re[i + j] += ar;
                im[i + j] += ai;
            }
        }
    }
}

/* ---- primitive operators ----------------------------------------------- */

/* nn.GELU() / F.gelu default: exact erf form, evaluated in double exactly as
 * tools/trellis_numpy_reference.py::gelu does. */
static float t_gelu(float v) {
    double x = (double)v;
    return (float)(0.5 * x * (1.0 + erf(x / T_SQRT2)));
}

static float t_silu(float v) {
    return v / (1.0f + expf(-v));
}

/* Conv1d over [in_ch, in_len] with zero padding; out has
 * out_len = in_len + 2*pad - K + 1 columns. w is [out_ch, in_ch, K]. */
static void t_conv1d(float *out, const float *x, const float *w, const float *b,
                     int in_ch, int out_ch, int in_len, int K, int pad) {
    const int out_len = in_len + 2 * pad - K + 1;
    int oc, ic, k, t;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * out_len;
        const float bias = b[oc];
        for (t = 0; t < out_len; t++) {
            double acc = 0.0;
            const int base = t - pad;
            for (ic = 0; ic < in_ch; ic++) {
                const float *xrow = x + (long)ic * in_len;
                const float *wrow = w + ((long)oc * in_ch + ic) * K;
                for (k = 0; k < K; k++) {
                    const int idx = base + k;
                    if (idx >= 0 && idx < in_len)
                        acc += (double)wrow[k] * (double)xrow[idx];
                }
            }
            orow[t] = (float)acc + bias;
        }
    }
}

/* Conv1d with kernel_size 1; w is [out_ch, in_ch, 1] i.e. a dense [out, in]. */
static void t_pointwise(float *out, const float *x, const float *w,
                        const float *b, int in_ch, int out_ch, int len) {
    int oc, ic, t;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * len;
        const float *wrow = w + (long)oc * in_ch;
        const float bias = b[oc];
        for (t = 0; t < len; t++) {
            double sum = 0.0;
            for (ic = 0; ic < in_ch; ic++)
                sum += (double)wrow[ic] * (double)x[(long)ic * len + t];
            orow[t] = (float)sum + bias;
        }
    }
}

/* Conv1d with groups == channels; w is [C, 1, K], zero padding. */
static void t_depthwise(float *out, const float *x, const float *w,
                        const float *b, int ch, int len, int K, int pad) {
    int c, k, t;
    for (c = 0; c < ch; c++) {
        const float *xrow = x + (long)c * len;
        const float *wrow = w + (long)c * K;
        float *orow = out + (long)c * len;
        const float bias = b[c];
        for (t = 0; t < len; t++) {
            double acc = 0.0;
            const int base = t - pad;
            for (k = 0; k < K; k++) {
                const int idx = base + k;
                if (idx >= 0 && idx < len)
                    acc += (double)wrow[k] * (double)xrow[idx];
            }
            orow[t] = (float)acc + bias;
        }
    }
}

/* nn.LayerNorm over the last axis of [rows, C]. In-place safe (out may == x). */
static void t_layer_norm(float *out, const float *x, const float *gamma,
                         const float *beta, int rows, int C, float eps) {
    int r, c;
    for (r = 0; r < rows; r++) {
        const float *row = x + (long)r * C;
        float *orow = out + (long)r * C;
        double sum = 0.0, var = 0.0;
        float mean, scale;
        for (c = 0; c < C; c++) sum += (double)row[c];
        mean = (float)(sum / (double)C);
        for (c = 0; c < C; c++) {
            float d = row[c] - mean;
            var += (double)d * (double)d;
        }
        scale = sqrtf((float)(var / (double)C) + eps);
        for (c = 0; c < C; c++)
            orow[c] = (row[c] - mean) / scale * gamma[c] + beta[c];
    }
}

/* nn.GroupNorm(1, C) on [C, T]: a single mean and biased variance over C*T. */
static void t_group_norm(float *out, const float *x, const float *gamma,
                         const float *beta, int C, int T, float eps) {
    const long n = (long)C * T;
    double sum = 0.0, var = 0.0;
    float mean, scale;
    long i;
    int c, t;
    for (i = 0; i < n; i++) sum += (double)x[i];
    mean = (float)(sum / (double)n);
    for (i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += (double)d * (double)d;
    }
    scale = sqrtf((float)(var / (double)n) + eps);
    for (c = 0; c < C; c++) {
        const float g = gamma[c], bb = beta[c];
        for (t = 0; t < T; t++) {
            long k = (long)c * T + t;
            out[k] = (x[k] - mean) / scale * g + bb;
        }
    }
}

/* nn.Linear on [rows, in]; w is [out, in] row-major, bias may be NULL. */
static void t_linear(float *out, const float *x, const float *w, const float *b,
                     int rows, int in_dim, int out_dim) {
    int r, o, i;
    for (r = 0; r < rows; r++) {
        const float *row = x + (long)r * in_dim;
        float *orow = out + (long)r * out_dim;
        for (o = 0; o < out_dim; o++) {
            const float *wrow = w + (long)o * in_dim;
            double sum = 0.0;
            for (i = 0; i < in_dim; i++) sum += (double)wrow[i] * (double)row[i];
            orow[o] = b ? (float)sum + b[o] : (float)sum;
        }
    }
}

static void t_transpose_ct_tc(float *out, const float *x, int C, int T) {
    int c, t;
    for (t = 0; t < T; t++)
        for (c = 0; c < C; c++) out[(long)t * C + c] = x[(long)c * T + t];
}

static void t_transpose_tc_ct(float *out, const float *x, int C, int T) {
    int c, t;
    for (c = 0; c < C; c++)
        for (t = 0; t < T; t++) out[(long)c * T + t] = x[(long)t * C + c];
}

/* torch.linspace(0, 1, count) cast to float32; count == 1 yields [0.0].
 * numpy/torch both build the ramp in float64 and pin the last entry to 1.0. */
static void t_linspace01(float *out, int count) {
    int i;
    if (count <= 0) return;
    if (count == 1) { out[0] = 0.0f; return; }
    {
        const double step = 1.0 / (double)(count - 1);
        for (i = 0; i < count; i++) out[i] = (float)((double)i * step);
        out[count - 1] = 1.0f;
    }
}

/* ---- front end (§3.1 - §3.3) ------------------------------------------- */

/* FactorizedTemporalConv1d: expand(temporal(reduce(x))). `base` is the tensor
 * index of reduce.weight. `out` may alias `x`. */
static void t_factorized(float *out, const float *x, const snt_trellis_model *m,
                         int base, int C, int R, int K, int len, float *red,
                         float *tmp) {
    t_pointwise(red, x, m->w[base], m->w[base + 1], C, R, len);
    t_depthwise(tmp, red, m->w[base + 2], m->w[base + 3], R, len, K, K / 2);
    t_pointwise(out, tmp, m->w[base + 4], m->w[base + 5], R, C, len);
}

/* TrellisResidualBlock: x += scale * second(silu(first(x))).
 * `base` is the tensor index of the block's 0-d scale. The inference mask is
 * all ones (see the header comment), so no mask multiply is applied. */
static void t_residual_block(float *x, const snt_trellis_model *m, int base,
                             int C, int R, int K, int len, float *h, float *red,
                             float *tmp) {
    const float scale = m->w[base][0];
    const long n = (long)C * len;
    long i;
    t_factorized(h, x, m, base + 1, C, R, K, len, red, tmp);
    for (i = 0; i < n; i++) h[i] = t_silu(h[i]);
    t_factorized(h, h, m, base + 7, C, R, K, len, red, tmp);
    for (i = 0; i < n; i++) x[i] += scale * h[i];
}

/* ---- pcm --------------------------------------------------------------- */

long snt_trellis_pcm16(const float *wav, long n, int16_t *out) {
    long i;
    if (!wav || !out || n <= 0) return 0;
    for (i = 0; i < n; i++) {
        float v = wav[i];
        if (v < -1.0f) v = -1.0f;
        if (v > 1.0f) v = 1.0f;
        out[i] = (int16_t)rintf(v * 32767.0f);
    }
    return n;
}

int snt_trellis_default_seed_key(const int32_t *ids, int n_ids, char *buf,
                                 size_t cap) {
    size_t pos = 0;
    int i;
    if (!ids || n_ids <= 0 || !buf || cap < 3) return -1;
    buf[pos++] = '[';
    for (i = 0; i < n_ids; i++) {
        char digits[16];
        int len = 0, neg = 0;
        long v = (long)ids[i];
        if (i > 0) {
            if (pos + 1 >= cap) return -1;
            buf[pos++] = ',';
        }
        if (v < 0) { neg = 1; v = -v; }
        do { digits[len++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
        if (neg) digits[len++] = '-';
        if (pos + (size_t)len + 2 > cap) return -1;
        while (len > 0) buf[pos++] = digits[--len];
    }
    if (pos + 2 > cap) return -1;
    buf[pos++] = ']';
    buf[pos] = '\0';
    return (int)pos;
}

/* ---- allocation helper -------------------------------------------------- */

static float *t_alloc(size_t count, int *ok) {
    float *p;
    if (!*ok) return NULL;
    if (count == 0) count = 1;
    p = (float *)malloc(count * sizeof(float));
    if (!p) *ok = 0;
    return p;
}

void snt_trellis_stages_free(snt_trellis_stages *s) {
    if (!s) return;
    free(s->log_durations);
    free(s->durations);
    free(s->acoustic_hidden);
    free(s->coordinates);
    free(s->noise);
    free(s->spec_real);
    free(s->spec_imag);
    free(s->waveform);
    memset(s, 0, sizeof *s);
}

/* ---- iSTFT (§3.6) ------------------------------------------------------- */

/* torch.istft(center=True, normalized=False, onesided=True). All arithmetic in
 * float64, matching the NumPy reference. Returns 0, -7 on a NOLA violation. */
static int t_istft(const float *spec_re, const float *spec_im, int bins, int T,
                   int n_fft, int hop, const float *window, float *wav,
                   double *sig, double *env, const t_fft *plan, double *fre,
                   double *fim) {
    const long expected = (long)n_fft + (long)hop * (T - 1);
    const int half = n_fft / 2;
    const long out_len = (long)hop * (T - 1);
    long i;
    int t, k;
    for (i = 0; i < expected; i++) { sig[i] = 0.0; env[i] = 0.0; }
    for (t = 0; t < T; t++) {
        const long start = (long)t * hop;
        for (k = 0; k < n_fft; k++) { fre[k] = 0.0; fim[k] = 0.0; }
        for (k = 0; k < bins; k++) {
            fre[k] = (double)spec_re[(long)k * T + t];
            fim[k] = (double)spec_im[(long)k * T + t];
        }
        for (k = 1; k < bins - 1; k++) {
            fre[n_fft - k] = fre[k];
            fim[n_fft - k] = -fim[k];
        }
        t_fft_run(plan, fre, fim, 1);
        for (k = 0; k < n_fft; k++) {
            double w = (double)window[k];
            sig[start + k] += fre[k] / (double)n_fft * w;
            env[start + k] += w * w;
        }
    }
    for (i = 0; i < out_len; i++) {
        double e = env[half + i];
        if (fabs(e) <= 1e-11) return -7;
        wav[i] = (float)(sig[half + i] / e);
    }
    return 0;
}

/* ---- DC blocker (§3.7) -------------------------------------------------- */

/* Truncated-IIR 4096-tap FIR, applied as a double-precision overlap-add FFT
 * convolution (see deviation 2 in the file header). */
static int t_dc_block(float *wav, long n, double pole, int taps) {
    const int fftn = 8192;
    const int blk = fftn - taps; /* 4096 samples advanced per transform */
    t_fft plan;
    double *hr = NULL, *hi = NULL, *xr = NULL, *xi = NULL, *acc = NULL;
    long b, nblocks, i;
    int k, rc = 0;
    if (taps >= fftn) return -9;
    if (t_fft_init(&plan, fftn) != 0) return -9;
    hr = (double *)malloc((size_t)fftn * sizeof(double));
    hi = (double *)malloc((size_t)fftn * sizeof(double));
    xr = (double *)malloc((size_t)fftn * sizeof(double));
    xi = (double *)malloc((size_t)fftn * sizeof(double));
    acc = (double *)calloc((size_t)n + (size_t)fftn, sizeof(double));
    if (!hr || !hi || !xr || !xi || !acc) { rc = -4; goto done; }

    /* h is built in float64 then cast to float32 before use (§3.7). */
    for (k = 0; k < fftn; k++) { hr[k] = 0.0; hi[k] = 0.0; }
    hr[0] = (double)(float)1.0;
    for (k = 1; k < taps; k++)
        hr[k] = (double)(float)((pole - 1.0) * pow(pole, (double)(k - 1)));
    t_fft_run(&plan, hr, hi, 0);

    nblocks = (n + blk - 1) / blk;
    for (b = 0; b < nblocks; b++) {
        const long start = b * blk;
        const long len = (n - start < blk) ? (n - start) : blk;
        for (k = 0; k < fftn; k++) { xr[k] = 0.0; xi[k] = 0.0; }
        for (i = 0; i < len; i++) xr[i] = (double)wav[start + i];
        t_fft_run(&plan, xr, xi, 0);
        for (k = 0; k < fftn; k++) {
            double ar = xr[k], ai = xi[k];
            xr[k] = ar * hr[k] - ai * hi[k];
            xi[k] = ar * hi[k] + ai * hr[k];
        }
        t_fft_run(&plan, xr, xi, 1);
        for (k = 0; k < fftn; k++) acc[start + k] += xr[k] / (double)fftn;
    }
    for (i = 0; i < n; i++) wav[i] = (float)acc[i];

done:
    free(hr); free(hi); free(xr); free(xi); free(acc);
    t_fft_free(&plan);
    return rc;
}

/* ---- pipeline ----------------------------------------------------------- */

int snt_trellis_run(const snt_trellis_model *m, const int32_t *ids, int n_ids,
                    const char *seed_key, int keep, snt_trellis_stages *out) {
    const int V = m ? m->vocab_size : 0;
    int DW, DR, DK, AW, AR, AK, W, RANK, HID, BINS, NC, NM;
    int i, j, t, c, step, rc = 0, ok = 1;
    long T = 0, S = 0;
    float *feat = NULL, *dx = NULL, *dh = NULL, *dred = NULL, *dtmp = NULL;
    float *ax = NULL, *ah = NULL, *ared = NULL, *atmp = NULL, *acat = NULL;
    float *hx = NULL, *hr = NULL, *hup = NULL, *hpad = NULL;
    float *dec_ct = NULL, *dec_ct2 = NULL, *dec_tc = NULL, *dec_rk = NULL;
    float *dec_hid = NULL, *dec_out = NULL;
    float *window = NULL;
    double *sig = NULL, *env = NULL, *fre = NULL, *fim = NULL;
    t_fft plan;
    int plan_ready = 0;
    snt_trellis_stages st;

    memset(&plan, 0, sizeof plan);
    memset(&st, 0, sizeof st);
    if (out) memset(out, 0, sizeof *out);
    if (!m || !ids || !seed_key || !out) return -1;
    if (n_ids <= 0 || n_ids > m->max_tokens) return -2;
    for (i = 0; i < n_ids; i++)
        if (ids[i] < 0 || ids[i] >= V) return -3;

    DW = m->duration_width; DR = m->duration_rank; DK = m->duration_kernel_size;
    AW = m->acoustic_width; AR = m->acoustic_rank; AK = m->acoustic_kernel_size;
    W = m->decoder_width; RANK = m->decoder_pointwise_rank;
    HID = m->decoder_width * m->decoder_expand;
    BINS = m->n_fft / 2 + 1;
    NC = m->decoder_noise_channels;
    NM = m->n_mels;

    st.n_tokens = n_ids;
    st.log_durations = t_alloc((size_t)n_ids, &ok);
    st.durations = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
    if (!st.durations) ok = 0;
    feat = t_alloc((size_t)(DW + 3) * n_ids, &ok);
    dx = t_alloc((size_t)DW * n_ids, &ok);
    dh = t_alloc((size_t)DW * n_ids, &ok);
    dred = t_alloc((size_t)DR * n_ids, &ok);
    dtmp = t_alloc((size_t)DR * n_ids, &ok);
    if (!ok) { rc = -4; goto done; }

    /* ---- §3.1 duration trellis ---- */
    {
        const float *emb = m->w[0];
        const float length_hint =
            (float)(log1p((double)n_ids) / log1p((double)m->max_tokens));
        float *pos = feat + (long)DW * n_ids;
        t_linspace01(pos, n_ids);
        for (i = 0; i < n_ids; i++) {
            feat[(long)(DW + 1) * n_ids + i] = length_hint;
            feat[(long)(DW + 2) * n_ids + i] = 1.0f; /* mask, always ones */
        }
        for (c = 0; c < DW; c++)
            for (i = 0; i < n_ids; i++)
                feat[(long)c * n_ids + i] = emb[(long)ids[i] * DW + c];
        t_pointwise(dx, feat, m->w[1], m->w[2], DW + 3, DW, n_ids);
        for (i = 0; i < m->duration_depth; i++)
            t_residual_block(dx, m, t_dur_block(m, i), DW, DR, DK, n_ids, dh,
                             dred, dtmp);
        {
            const int oi = t_dur_out(m);
            t_pointwise(st.log_durations, dx, m->w[oi], m->w[oi + 1], DW, 1,
                        n_ids);
        }
        for (i = 0; i < n_ids; i++) {
            float pred = expf(st.log_durations[i]);
            float r;
            if (pred < 1.0f) pred = 1.0f;
            r = rintf(pred); /* round-half-to-even, matching torch/numpy */
            if (r < (float)m->min_duration) r = (float)m->min_duration;
            if (r > (float)m->max_duration) r = (float)m->max_duration;
            st.durations[i] = (int32_t)r;
            T += (long)st.durations[i];
        }
    }
    if (T < 2) { rc = -5; goto done; }
    if ((long)NC * T < 16) { rc = -6; goto done; }
    st.frames = (int)T;
    S = (long)m->hop_length * (T - 1);
    st.samples = S;

    /* ---- §3.2 acoustic trellis ---- */
    acat = t_alloc((size_t)(AW + 3) * (size_t)T, &ok);
    ax = t_alloc((size_t)AW * (size_t)T, &ok);
    ah = t_alloc((size_t)AW * (size_t)T, &ok);
    ared = t_alloc((size_t)AR * (size_t)T, &ok);
    atmp = t_alloc((size_t)AR * (size_t)T, &ok);
    st.acoustic_hidden = t_alloc((size_t)AW * (size_t)T, &ok);
    if (!ok) { rc = -4; goto done; }
    {
        const float *emb = m->w[t_aco_emb(m)];
        const int pi = t_aco_emb(m) + 1;
        int max_dur = 1;
        double denom;
        float *tokpos = acat + (long)AW * n_ids;
        for (i = 0; i < n_ids; i++)
            if (st.durations[i] > max_dur) max_dur = st.durations[i];
        denom = log1p((double)max_dur);
        t_linspace01(tokpos, n_ids);
        for (i = 0; i < n_ids; i++)
            acat[(long)(AW + 1) * n_ids + i] =
                (float)(log1p((double)st.durations[i]) / denom);
        for (c = 0; c < AW; c++)
            for (i = 0; i < n_ids; i++)
                acat[(long)c * n_ids + i] = emb[(long)ids[i] * AW + c];
        t_pointwise(ax, acat, m->w[pi], m->w[pi + 1], AW + 2, AW, n_ids);
        for (i = 0; i < m->token_depth; i++)
            t_residual_block(ax, m, t_tok_block(m, i), AW, AR, AK, n_ids, ah,
                             ared, atmp);
        /* repeat_interleave into the [AW, T] slot of the frame concat buffer */
        for (c = 0; c < AW; c++) {
            long w = 0;
            for (i = 0; i < n_ids; i++) {
                const float v = ax[(long)c * n_ids + i];
                for (j = 0; j < st.durations[i]; j++)
                    acat[(long)c * T + w++] = v;
            }
        }
        {
            float *frame_pos = acat + (long)AW * T;
            float *tok_pos = acat + (long)(AW + 1) * T;
            float *dur_pos = acat + (long)(AW + 2) * T;
            const double tok_den = (double)(n_ids - 1 >= 1 ? n_ids - 1 : 1);
            long w = 0;
            t_linspace01(frame_pos, (int)T);
            for (i = 0; i < n_ids; i++) {
                const float tp = (float)((double)i / tok_den);
                t_linspace01(dur_pos + w, st.durations[i]);
                for (j = 0; j < st.durations[i]; j++) tok_pos[w + j] = tp;
                w += st.durations[i];
            }
        }
        {
            const int fi = t_frm_proj(m);
            t_pointwise(ax, acat, m->w[fi], m->w[fi + 1], AW + 3, AW, (int)T);
        }
        for (i = 0; i < m->frame_depth; i++)
            t_residual_block(ax, m, t_frm_block(m, i), AW, AR, AK, (int)T, ah,
                             ared, atmp);
        memcpy(st.acoustic_hidden, ax, (size_t)AW * (size_t)T * sizeof(float));
    }

    /* ---- §3.3 recurrent transplant head ---- */
    hx = ax; /* the frame state is the head's input; reuse the buffer */
    hr = ah;
    hup = t_alloc((size_t)AW * (size_t)m->head_expansion * (size_t)T, &ok);
    hpad = t_alloc((size_t)AW * (size_t)(T + m->output_kernel_size - 1), &ok);
    st.coordinates = t_alloc((size_t)NM * (size_t)T, &ok);
    if (!ok) { rc = -4; goto done; }
    {
        const int hi = t_head(m);
        const float residual_scale = m->w[hi + 8][0];
        const float *loop_bias = m->w[hi + 9];
        const int radius = m->output_kernel_size / 2;
        const long n = (long)AW * T;
        for (step = 0; step < m->head_steps; step++) {
            long q;
            t_depthwise(hr, hx, m->w[hi], m->w[hi + 1], AW, (int)T,
                        m->head_kernel_size, m->head_kernel_size / 2);
            t_group_norm(hr, hr, m->w[hi + 2], m->w[hi + 3], AW, (int)T,
                         SNT_TRELLIS_GN_EPS);
            for (c = 0; c < AW; c++) {
                const float bias = loop_bias[(long)step * AW + c];
                for (t = 0; t < (int)T; t++) hr[(long)c * T + t] += bias;
            }
            t_pointwise(hup, hr, m->w[hi + 4], m->w[hi + 5], AW,
                        AW * m->head_expansion, (int)T);
            for (q = 0; q < (long)AW * m->head_expansion * T; q++)
                hup[q] = t_gelu(hup[q]);
            t_pointwise(hr, hup, m->w[hi + 6], m->w[hi + 7],
                        AW * m->head_expansion, AW, (int)T);
            for (q = 0; q < n; q++) hx[q] += residual_scale * hr[q];
        }
        /* replicate pad then a valid k=5 conv */
        for (c = 0; c < AW; c++) {
            float *dst = hpad + (long)c * (T + 2 * radius);
            const float *src = hx + (long)c * T;
            for (i = 0; i < radius; i++) dst[i] = src[0];
            memcpy(dst + radius, src, (size_t)T * sizeof(float));
            for (i = 0; i < radius; i++) dst[radius + T + i] = src[T - 1];
        }
        t_conv1d(st.coordinates, hpad, m->w[hi + 10], m->w[hi + 11], AW, NM,
                 (int)(T + 2 * radius), m->output_kernel_size, 0);
    }

    /* ---- §3.4 decoder noise ---- */
    st.noise = t_alloc((size_t)NC * (size_t)T, &ok);
    if (!ok) { rc = -4; goto done; }
    {
        int nrc = t_seeded_noise(seed_key, NC, (int)T, st.noise);
        if (nrc == -1) { rc = -6; goto done; }
        if (nrc != 0) { rc = -4; goto done; }
    }

    /* ---- §3.5 low-rank TinyVocos decoder ---- */
    dec_ct = t_alloc((size_t)W * (size_t)T, &ok);
    dec_ct2 = t_alloc((size_t)W * (size_t)T, &ok);
    dec_tc = t_alloc((size_t)W * (size_t)T, &ok);
    dec_rk = t_alloc((size_t)RANK * (size_t)T, &ok);
    dec_hid = t_alloc((size_t)HID * (size_t)T, &ok);
    dec_out = t_alloc((size_t)2 * BINS * (size_t)T, &ok);
    st.spec_real = t_alloc((size_t)BINS * (size_t)T, &ok);
    st.spec_imag = t_alloc((size_t)BINS * (size_t)T, &ok);
    if (!ok) { rc = -4; goto done; }
    {
        const int di = t_dec(m);
        const int K = SNT_TRELLIS_DECODER_KERNEL;
        long q;
        t_conv1d(dec_ct, st.coordinates, m->w[di], m->w[di + 1], NM, W, (int)T,
                 K, K / 2);
        t_conv1d(dec_ct2, st.noise, m->w[di + 2], m->w[di + 3], NC, W, (int)T, K,
                 K / 2);
        for (q = 0; q < (long)W * T; q++) dec_ct[q] += dec_ct2[q];
        t_transpose_ct_tc(dec_tc, dec_ct, W, (int)T);
        t_layer_norm(dec_tc, dec_tc, m->w[di + 4], m->w[di + 5], (int)T, W,
                     SNT_TRELLIS_LN_EPS);
        t_transpose_tc_ct(dec_ct, dec_tc, W, (int)T);

        for (i = 0; i < m->decoder_num_layers; i++) {
            const int bi = t_dec_block(m, i);
            const float *gamma = m->w[bi + 10];
            t_depthwise(dec_ct2, dec_ct, m->w[bi], m->w[bi + 1], W, (int)T, K,
                        K / 2);
            t_transpose_ct_tc(dec_tc, dec_ct2, W, (int)T);
            t_layer_norm(dec_tc, dec_tc, m->w[bi + 2], m->w[bi + 3], (int)T, W,
                         SNT_TRELLIS_LN_EPS);
            t_linear(dec_rk, dec_tc, m->w[bi + 4], NULL, (int)T, W, RANK);
            t_linear(dec_hid, dec_rk, m->w[bi + 5], m->w[bi + 6], (int)T, RANK,
                     HID);
            for (q = 0; q < (long)HID * T; q++) dec_hid[q] = t_gelu(dec_hid[q]);
            t_linear(dec_rk, dec_hid, m->w[bi + 7], NULL, (int)T, HID, RANK);
            t_linear(dec_tc, dec_rk, m->w[bi + 8], m->w[bi + 9], (int)T, RANK, W);
            for (t = 0; t < (int)T; t++)
                for (c = 0; c < W; c++)
                    dec_ct[(long)c * T + t] += dec_tc[(long)t * W + c] * gamma[c];
        }

        {
            const int fi = t_dec_final(m);
            t_transpose_ct_tc(dec_tc, dec_ct, W, (int)T);
            t_layer_norm(dec_tc, dec_tc, m->w[fi], m->w[fi + 1], (int)T, W,
                         SNT_TRELLIS_LN_EPS);
            t_linear(dec_out, dec_tc, m->w[fi + 2], m->w[fi + 3], (int)T, W,
                     2 * BINS);
        }
        for (t = 0; t < (int)T; t++) {
            const float *row = dec_out + (long)t * 2 * BINS;
            for (c = 0; c < BINS; c++) {
                float mag = expf(row[c]);
                float phase = row[BINS + c];
                if (c == 0 || c == BINS - 1) mag = 0.0f; /* zero DC + Nyquist */
                if (mag > SNT_TRELLIS_MAG_CLIP) mag = SNT_TRELLIS_MAG_CLIP;
                st.spec_real[(long)c * T + t] = mag * cosf(phase);
                st.spec_imag[(long)c * T + t] = mag * sinf(phase);
            }
        }
    }

    /* ---- §3.6 iSTFT + §3.7 DC blocker ---- */
    st.waveform = t_alloc((size_t)S, &ok);
    window = t_alloc((size_t)m->n_fft, &ok);
    sig = (double *)malloc(((size_t)m->n_fft + (size_t)m->hop_length * (T - 1)) *
                           sizeof(double));
    env = (double *)malloc(((size_t)m->n_fft + (size_t)m->hop_length * (T - 1)) *
                           sizeof(double));
    fre = (double *)malloc((size_t)m->n_fft * sizeof(double));
    fim = (double *)malloc((size_t)m->n_fft * sizeof(double));
    if (!ok || !sig || !env || !fre || !fim) { rc = -4; goto done; }
    for (i = 0; i < m->n_fft; i++)
        window[i] = (float)(0.5 - 0.5 * cos(2.0 * T_PI * (double)i /
                                            (double)m->n_fft));
    if (t_fft_init(&plan, m->n_fft) != 0) { rc = -9; goto done; }
    plan_ready = 1;
    rc = t_istft(st.spec_real, st.spec_imag, BINS, (int)T, m->n_fft,
                 m->hop_length, window, st.waveform, sig, env, &plan, fre, fim);
    if (rc != 0) goto done;
    rc = t_dc_block(st.waveform, S, SNT_TRELLIS_DC_POLE, SNT_TRELLIS_DC_TAPS);
    if (rc != 0) goto done;
    {
        long q;
        for (q = 0; q < S; q++) {
            const float v = st.waveform[q];
            if (!(v == v) || v > 1e30f || v < -1e30f) { rc = -8; goto done; }
        }
    }

done:
    free(feat); free(dx); free(dh); free(dred); free(dtmp);
    free(acat); free(ah); free(ared); free(atmp);
    free(hup); free(hpad);
    free(dec_ct); free(dec_ct2); free(dec_tc); free(dec_rk); free(dec_hid);
    free(dec_out);
    free(window);
    free(sig); free(env); free(fre); free(fim);
    free(ax); /* == hx */
    if (plan_ready) t_fft_free(&plan);
    if (rc != 0) {
        snt_trellis_stages_free(&st);
        return rc;
    }
    if (!(keep & SNT_TRELLIS_KEEP_FRONT)) {
        free(st.log_durations); st.log_durations = NULL;
        free(st.durations); st.durations = NULL;
        free(st.acoustic_hidden); st.acoustic_hidden = NULL;
        free(st.coordinates); st.coordinates = NULL;
    }
    if (!(keep & SNT_TRELLIS_KEEP_NOISE)) { free(st.noise); st.noise = NULL; }
    if (!(keep & SNT_TRELLIS_KEEP_SPEC)) {
        free(st.spec_real); st.spec_real = NULL;
        free(st.spec_imag); st.spec_imag = NULL;
    }
    *out = st;
    return 0;
}
