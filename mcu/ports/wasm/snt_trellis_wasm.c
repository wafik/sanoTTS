/* snt_trellis_wasm.c -- WebAssembly (Emscripten) shim around snt_trellis.c,
 * the 995,058-parameter Trellis-RIFT English text-to-wave model.
 *
 * Blob contract: `bundle` is the single-file form
 * artifacts/trellis-web-port-20260819/bundle/trellis_rift_995058_bundle.bin,
 * i.e. meta.bin (32 int32 header fields + 217 (offset,count) pairs) followed
 * immediately by the 995,058 little-endian float32 weights. Both extents are
 * derivable from the header, so JS only has to hand over the raw bytes and
 * their length -- no weights are baked into the module.
 *
 * The synthesis result is an opaque handle (a malloc'd struct pointer) rather
 * than module-global state, so several utterances can be in flight and the
 * shim stays reentrant. JS calls snt_trellis_wasm_run, reads what it needs
 * through the accessors, then calls snt_trellis_wasm_release exactly once.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "snt_trellis.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define SNT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SNT_EXPORT
#endif

/* stage selectors for snt_trellis_wasm_stage */
#define SNT_TRELLIS_STAGE_LOG_DURATIONS 0
#define SNT_TRELLIS_STAGE_DURATIONS 1
#define SNT_TRELLIS_STAGE_ACOUSTIC_HIDDEN 2
#define SNT_TRELLIS_STAGE_COORDINATES 3
#define SNT_TRELLIS_STAGE_NOISE 4
#define SNT_TRELLIS_STAGE_SPEC_REAL 5
#define SNT_TRELLIS_STAGE_SPEC_IMAG 6
#define SNT_TRELLIS_STAGE_WAVEFORM 7

typedef struct {
    snt_trellis_stages stages;
    int32_t sample_rate;
} snt_trellis_handle;

/* Bind a bundle blob. Returns 0 on success, else the negative code from
 * snt_trellis_bundle_split (-1..-5) or snt_trellis_init (-1..-9) offset by
 * -100 so the two sources stay distinguishable. */
static int bind_bundle(const uint8_t *bundle, size_t bundle_bytes,
                       snt_trellis_model *model) {
    size_t meta_bytes = 0, weight_floats = 0;
    int rc = snt_trellis_bundle_split(bundle, bundle_bytes, &meta_bytes,
                                      &weight_floats);
    if (rc != 0) return rc;
    rc = snt_trellis_init(model, bundle, meta_bytes,
                          (const float *)(const void *)(bundle + meta_bytes),
                          weight_floats);
    if (rc != 0) return rc - 100;
    return 0;
}

/* Sample rate declared by a bundle, or a negative bind error code. Lets a page
 * configure its AudioContext before synthesizing anything. */
SNT_EXPORT
int snt_trellis_wasm_sample_rate(const uint8_t *bundle, int bundle_bytes) {
    snt_trellis_model model;
    int rc;
    if (!bundle || bundle_bytes <= 0) return -1;
    rc = bind_bundle(bundle, (size_t)bundle_bytes, &model);
    if (rc != 0) return rc;
    return (int)model.sample_rate;
}

/* Write json.dumps(ids, separators=(",", ":")) into buf. Returns the byte
 * length excluding the NUL, or -1 if cap is too small. Exposed so a page can
 * see the exact seed key an utterance will use. */
SNT_EXPORT
int snt_trellis_wasm_seed_key(const int32_t *ids, int n_ids, char *buf,
                              int cap) {
    if (!ids || n_ids <= 0 || !buf || cap <= 0) return -1;
    return snt_trellis_default_seed_key(ids, n_ids, buf, (size_t)cap);
}

/* Run the full pipeline. `seed_key` may be NULL, in which case the default
 * id-derived key is used (§3.4 of the operator spec). `keep` is a bitwise OR
 * of SNT_TRELLIS_KEEP_* (the waveform is always kept). On success returns a
 * non-zero handle and stores 0 through `err`; on failure returns 0 and stores
 * the negative reason: the bind codes above, or the snt_trellis_run codes
 * (-1..-9), or -200 for an out-of-memory in this shim. */
SNT_EXPORT
void *snt_trellis_wasm_run(const uint8_t *bundle, int bundle_bytes,
                           const int32_t *ids, int n_ids, const char *seed_key,
                           int keep, int *err) {
    snt_trellis_model model;
    snt_trellis_handle *handle = NULL;
    char *owned_key = NULL;
    const char *key = seed_key;
    int rc;

    if (err) *err = 0;
    if (!bundle || bundle_bytes <= 0 || !ids || n_ids <= 0) {
        if (err) *err = -1;
        return NULL;
    }
    rc = bind_bundle(bundle, (size_t)bundle_bytes, &model);
    if (rc != 0) {
        if (err) *err = rc;
        return NULL;
    }
    if (!key) {
        /* worst case "-2147483648," per id, plus brackets and the NUL */
        size_t cap = (size_t)n_ids * 12u + 3u;
        owned_key = (char *)malloc(cap);
        if (!owned_key) {
            if (err) *err = -200;
            return NULL;
        }
        if (snt_trellis_default_seed_key(ids, n_ids, owned_key, cap) < 0) {
            free(owned_key);
            if (err) *err = -200;
            return NULL;
        }
        key = owned_key;
    }
    handle = (snt_trellis_handle *)calloc(1, sizeof *handle);
    if (!handle) {
        free(owned_key);
        if (err) *err = -200;
        return NULL;
    }
    rc = snt_trellis_run(&model, ids, n_ids, key, keep, &handle->stages);
    free(owned_key);
    if (rc != 0) {
        free(handle);
        if (err) *err = rc;
        return NULL;
    }
    handle->sample_rate = model.sample_rate;
    return handle;
}

SNT_EXPORT
int snt_trellis_wasm_tokens(const void *h) {
    return h ? ((const snt_trellis_handle *)h)->stages.n_tokens : -1;
}

SNT_EXPORT
int snt_trellis_wasm_frames(const void *h) {
    return h ? ((const snt_trellis_handle *)h)->stages.frames : -1;
}

SNT_EXPORT
int snt_trellis_wasm_samples(const void *h) {
    return h ? (int)((const snt_trellis_handle *)h)->stages.samples : -1;
}

SNT_EXPORT
int snt_trellis_wasm_handle_sample_rate(const void *h) {
    return h ? (int)((const snt_trellis_handle *)h)->sample_rate : -1;
}

/* Pointer to one retained stage tensor (see SNT_TRELLIS_STAGE_*), or 0 if the
 * handle is null, the selector is unknown, or that stage was not retained by
 * the `keep` mask the run was started with. */
SNT_EXPORT
void *snt_trellis_wasm_stage(const void *h, int which) {
    const snt_trellis_stages *s;
    if (!h) return NULL;
    s = &((const snt_trellis_handle *)h)->stages;
    switch (which) {
        case SNT_TRELLIS_STAGE_LOG_DURATIONS: return s->log_durations;
        case SNT_TRELLIS_STAGE_DURATIONS: return s->durations;
        case SNT_TRELLIS_STAGE_ACOUSTIC_HIDDEN: return s->acoustic_hidden;
        case SNT_TRELLIS_STAGE_COORDINATES: return s->coordinates;
        case SNT_TRELLIS_STAGE_NOISE: return s->noise;
        case SNT_TRELLIS_STAGE_SPEC_REAL: return s->spec_real;
        case SNT_TRELLIS_STAGE_SPEC_IMAG: return s->spec_imag;
        case SNT_TRELLIS_STAGE_WAVEFORM: return s->waveform;
        default: return NULL;
    }
}

/* Convert the retained waveform to int16 PCM. Returns the sample count, or a
 * negative code: -1 bad arguments, -2 out capacity too small. */
SNT_EXPORT
int snt_trellis_wasm_pcm16(const void *h, int16_t *out, int out_cap) {
    const snt_trellis_stages *s;
    if (!h || !out || out_cap <= 0) return -1;
    s = &((const snt_trellis_handle *)h)->stages;
    if (!s->waveform || s->samples <= 0) return -1;
    if (s->samples > (long)out_cap) return -2;
    return (int)snt_trellis_pcm16(s->waveform, s->samples, out);
}

SNT_EXPORT
void snt_trellis_wasm_release(void *h) {
    if (!h) return;
    snt_trellis_stages_free(&((snt_trellis_handle *)h)->stages);
    free(h);
}
