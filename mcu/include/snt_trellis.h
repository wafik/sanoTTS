/* snt_trellis.h -- portable fp32 reference of the 995,058-parameter
 * Trellis-RIFT English text-to-wave model (phoneme ids -> 24 kHz waveform).
 *
 * Pipeline (see docs/trellis-web-port-operators.md, which this file follows
 * section by section; tools/trellis_numpy_reference.py is the executable
 * ground truth):
 *
 *   ids[N]
 *     -> duration trellis        (embedding + 1x1 proj + 3 factorized blocks)
 *     -> round(exp(.)) clamp     -> integer durations[N], T = sum(durations)
 *     -> acoustic trellis        (token path, repeat_interleave, frame path)
 *     -> recurrent transplant head (one shared cell reused 4x + k5 conv)
 *     -> decoder coordinates     [100, T]
 *     -> deterministic noise     [4, T]   (sha256 -> MT19937 -> Box-Muller)
 *     -> low-rank TinyVocos      -> complex STFT [513, T]
 *     -> iSTFT (hann periodic, center, n_fft 1024 / hop 256)
 *     -> 4096-tap DC blocker     -> float32 PCM at 24 kHz
 *
 * No shapes are hard-coded: every dimension comes from the meta.bin header
 * shipped in artifacts/trellis-web-port-20260819/bundle/, and init()
 * cross-checks all 217 tensor sizes against the shapes those dims imply, so a
 * stale or reordered blob fails loudly instead of decoding garbage.
 *
 * Dependencies: libm plus the C standard library (malloc/free/memcpy). C99,
 * no VLAs, no global mutable state.
 *
 * NUMERICAL CONTRACT. Reductions (layer/group norm means, conv and linear dot
 * products) accumulate in double and round to float32 on store; the iSTFT, the
 * FFTs and the DC blocker run in double exactly as the NumPy reference does;
 * GELU is the exact erf form evaluated in double. Everything the reference
 * keeps in float32 (residual chains, magnitudes, phases, the PRNG) is float32
 * here too. The transcendentals in the noise PRNG use plain float32 libm,
 * which is the portable contract documented in section 4.3 of the operator
 * spec -- see the note in snt_trellis.c about why bit-exactness with
 * numpy_decoder_noise.npy is not attainable across hosts.
 */
#ifndef SNT_TRELLIS_H
#define SNT_TRELLIS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNT_TRELLIS_MAGIC 0x534E5452L /* 'SNTR' */
#define SNT_TRELLIS_VERSION 1
#define SNT_TRELLIS_HEADER_FIELDS 32
#define SNT_TRELLIS_MAX_TENSORS 256

/* Hard-coded structural constants that the meta.bin header does not carry;
 * every one of them is validated indirectly because init() checks the
 * corresponding tensor's float count. */
#define SNT_TRELLIS_DECODER_KERNEL 7  /* decoder embed / noise_adapter / dwconv */
#define SNT_TRELLIS_LN_EPS 1e-6f      /* nn.LayerNorm(dim, eps=1e-6) */
#define SNT_TRELLIS_GN_EPS 1e-5f      /* nn.GroupNorm(1, width) default */
#define SNT_TRELLIS_MAG_CLIP 100.0f   /* Vocos ISTFTHead safeguard */
#define SNT_TRELLIS_DC_POLE 0.9973    /* DC_BLOCK_R */
#define SNT_TRELLIS_DC_TAPS 4096      /* DC_BLOCK_TAPS */

typedef struct {
    /* meta.bin header, in field order (see header_int32_fields in the JSON) */
    int32_t magic, version;
    int32_t vocab_size, duration_width, duration_depth, duration_kernel_size,
        duration_rank;
    int32_t acoustic_width, token_depth, frame_depth, acoustic_kernel_size,
        acoustic_rank;
    int32_t n_mels, head_steps, head_kernel_size, head_expansion,
        output_kernel_size;
    int32_t max_tokens, min_duration, max_duration;
    int32_t decoder_width, decoder_num_layers, decoder_expand,
        decoder_pointwise_rank, decoder_noise_channels;
    int32_t n_fft, hop_length, win_length, sample_rate;
    int32_t n_tensors, front_floats, decoder_floats;
    /* per-tensor weight pointers and float counts, in bundle order */
    const float *w[SNT_TRELLIS_MAX_TENSORS];
    int32_t wn[SNT_TRELLIS_MAX_TENSORS];
} snt_trellis_model;

/* Every intermediate the gating procedure in section 7 of the operator spec
 * compares. All buffers are malloc'd by snt_trellis_run and released by
 * snt_trellis_stages_free; the ones the caller did not ask to keep are NULL. */
typedef struct {
    int n_tokens;      /* N */
    int frames;        /* T */
    long samples;      /* S = hop * (T - 1) */
    float *log_durations;    /* [N]        */
    int32_t *durations;      /* [N]        */
    float *acoustic_hidden;  /* [96, T]    */
    float *coordinates;      /* [100, T]   */
    float *noise;            /* [4, T]     */
    float *spec_real;        /* [513, T]   */
    float *spec_imag;        /* [513, T]   */
    float *waveform;         /* [S]        */
} snt_trellis_stages;

/* Keep-flags for snt_trellis_run's `keep` argument. The waveform is always
 * kept; the rest exist so a browser build does not pay for the 68 MB STFT
 * buffers of a worst-case utterance. */
#define SNT_TRELLIS_KEEP_NONE 0
#define SNT_TRELLIS_KEEP_FRONT 1  /* log_durations, durations, hidden, coords */
#define SNT_TRELLIS_KEEP_NOISE 2
#define SNT_TRELLIS_KEEP_SPEC 4
#define SNT_TRELLIS_KEEP_ALL 7

/* Split a combined `*_bundle.bin` (meta.bin ++ weights) into its two halves.
 * On success *meta_bytes is the header length and *weight_floats the number of
 * float32 values that follow it. Returns 0, or a negative error code:
 *  -1 bad arguments   -2 blob too short   -3 bad magic/version
 *  -4 implausible n_tensors   -5 declared float count does not fit the blob */
int snt_trellis_bundle_split(const void *blob, size_t blob_bytes,
                             size_t *meta_bytes, size_t *weight_floats);

/* Parse meta.bin and bind weight pointers. `weights` must outlive the model.
 * Returns 0, or a negative error code:
 *  -1 bad arguments        -2 meta blob too short for the header/table
 *  -3 bad magic            -4 bad version
 *  -5 a config dimension is out of range
 *  -6 n_tensors is not what the config implies (or exceeds the table)
 *  -7 malformed offset/size entry
 *  -8 a tensor's float count disagrees with the shape the config implies
 *  -9 a tensor runs past the end of the weight blob */
int snt_trellis_init(snt_trellis_model *m, const void *meta, size_t meta_bytes,
                     const float *weights, size_t weight_floats);

/* Build the default noise seed key for an id sequence: the exact string
 * json.dumps(ids, separators=(",", ":")) produces, e.g. "[1,37,2]".
 * Returns the number of bytes written excluding the NUL, or -1 if `cap` is too
 * small (the required capacity is at most 12 * n_ids + 3). */
int snt_trellis_default_seed_key(const int32_t *ids, int n_ids, char *buf,
                                 size_t cap);

/* Run the whole pipeline. `seed_key` is the UTF-8 string hashed to seed the
 * decoder noise (see snt_trellis_default_seed_key). `keep` is a bitwise OR of
 * SNT_TRELLIS_KEEP_*. On success returns 0 with `out` populated (call
 * snt_trellis_stages_free when done); on failure returns a negative code and
 * `out` is zeroed:
 *   -1 bad arguments                    -2 n_ids outside [1, max_tokens]
 *   -3 an id is outside [0, vocab)      -4 allocation failure
 *   -5 fewer than 2 frames (the iSTFT needs at least two)
 *   -6 fewer than 16 noise values (torch dispatches those to a scalar path
 *      this port deliberately does not implement)
 *   -7 the iSTFT window envelope violated NOLA
 *   -8 the output is not finite
 *   -9 an internal FFT plan could not be built */
int snt_trellis_run(const snt_trellis_model *m, const int32_t *ids, int n_ids,
                    const char *seed_key, int keep, snt_trellis_stages *out);

/* Release every buffer in `s` and zero it. Safe on an all-zero struct. */
void snt_trellis_stages_free(snt_trellis_stages *s);

/* clamp to [-1, 1], round-half-to-even, scale by 32767 -- byte-identical to
 * the NumPy reference's write_wav payload. Returns the sample count. */
long snt_trellis_pcm16(const float *wav, long n, int16_t *out);

#ifdef __cplusplus
}
#endif
#endif
