#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "unet.hpp"
#include "chroma.hpp"

struct DiffusionModel {
    virtual void compute(int n_threads,
                         struct ggml_tensor* x,
                         struct ggml_tensor* timesteps,
                         struct ggml_tensor* context,
                         struct ggml_tensor* c_concat,
                         struct ggml_tensor* y,
                         struct ggml_tensor* guidance,
                         int num_video_frames                      = -1,
                         std::vector<struct ggml_tensor*> controls = {},
                         float control_strength                    = 0.f,
                         struct ggml_tensor** output               = NULL,
                         struct ggml_context* output_ctx           = NULL,
                         std::vector<int> skip_layers              = std::vector<int>())             = 0;
    virtual void alloc_params_buffer()                                                  = 0;
    virtual void free_params_buffer()                                                   = 0;
    virtual void free_compute_buffer()                                                  = 0;
    virtual void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) = 0;
    virtual size_t get_params_buffer_size()                                             = 0;
    virtual int64_t get_adm_in_channels()                                               = 0;
};

struct UNetModel : public DiffusionModel {
    UNetModelRunner unet;

    UNetModel(ggml_backend_t backend,
              std::map<std::string, enum ggml_type>& tensor_types,
              SDVersion version = VERSION_SD1,
              bool flash_attn   = false)
        : unet(backend, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    void alloc_params_buffer() {
        unet.alloc_params_buffer();
    }

    void free_params_buffer() {
        unet.free_params_buffer();
    }

    void free_compute_buffer() {
        unet.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return unet.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return unet.unet.adm_in_channels;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        (void)skip_layers;  // SLG doesn't work with UNet models
        return unet.compute(n_threads, x, timesteps, context, c_concat, y, num_video_frames, controls, control_strength, output, output_ctx);
    }
};

struct MMDiTModel : public DiffusionModel {
    MMDiTRunner mmdit;

    MMDiTModel(ggml_backend_t backend,
               std::map<std::string, enum ggml_type>& tensor_types)
        : mmdit(backend, tensor_types, "model.diffusion_model") {
    }

    void alloc_params_buffer() {
        mmdit.alloc_params_buffer();
    }

    void free_params_buffer() {
        mmdit.free_params_buffer();
    }

    void free_compute_buffer() {
        mmdit.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        mmdit.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return mmdit.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768 + 1280;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return mmdit.compute(n_threads, x, timesteps, context, y, output, output_ctx, skip_layers);
    }
};

struct FluxModel : public DiffusionModel {
    Flux::FluxRunner flux;

    FluxModel(ggml_backend_t backend,
              std::map<std::string, enum ggml_type>& tensor_types,
              SDVersion version = VERSION_FLUX,
              bool flash_attn   = false)
        : flux(backend, tensor_types, "model.diffusion_model", version, flash_attn) {
    }

    void alloc_params_buffer() {
        flux.alloc_params_buffer();
    }

    void free_params_buffer() {
        flux.free_params_buffer();
    }

    void free_compute_buffer() {
        flux.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        flux.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return flux.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        return 768;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 struct ggml_tensor* guidance,
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        return flux.compute(n_threads, x, timesteps, context, c_concat, y, guidance, output, output_ctx, skip_layers);
    }
};

struct ChromaModel : public DiffusionModel {
    Chroma::ChromaRunner chroma;

    ChromaModel(ggml_backend_t backend,
                std::map<std::string, enum ggml_type>& tensor_types,
                SDVersion version = VERSION_CHROMA,
                bool flash_attn   = false)
        : chroma(backend, tensor_types,flash_attn) {
    }

    void alloc_params_buffer() {
        chroma.alloc_params_buffer();
    }

    void free_params_buffer() {
        chroma.free_params_buffer();
    }

    void free_compute_buffer() {
        chroma.free_compute_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        chroma.get_param_tensors(tensors, "model.diffusion_model");
    }

    size_t get_params_buffer_size() {
        return chroma.get_params_buffer_size();
    }

    int64_t get_adm_in_channels() {
        // Chroma does not use ADM, so return 0 or a suitable default
        return 0;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context, // T5 embeddings
                 struct ggml_tensor* c_concat, // Not used by Chroma
                 struct ggml_tensor* y, // Not used by Chroma
                 struct ggml_tensor* guidance, // Not used by Chroma
                 int num_video_frames                      = -1, // Not used by Chroma
                 std::vector<struct ggml_tensor*> controls = {}, // Not used by Chroma
                 float control_strength                    = 0.f, // Not used by Chroma
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        // For Chroma, context is T5 embeddings, c_concat and y are not used.
        // We need to pass positional embeddings (pe) and t5_padding_mask.
        // These are not directly available in the DiffusionModel compute signature.
        // This implies a need to adjust the DiffusionModel interface or how ChromaModel is called.

        // For now, let's assume pe and t5_padding_mask are handled internally by ChromaRunner
        // or passed through a different mechanism.
        // Based on the ChromaRunner::build_graph, it expects t5_padding_mask and pe.
        // The current DiffusionModel interface does not provide these.

        // This is a design conflict. The current DiffusionModel interface is generic.
        // Chroma has specific inputs (T5 embeddings, T5 padding mask, positional embeddings).
        // FluxModel also has specific inputs (pe, guidance).

        // Let's re-evaluate the `compute` method signature for `DiffusionModel`.
        // The `compute` method in `DiffusionModel` is quite generic.
        // `FluxModel` uses `pe` and `guidance`.
        // `ChromaModel` needs `pe` and `t5_padding_mask`.

        // The `stable-diffusion.cpp` calls `diffusion_model->compute`.
        // It passes `context`, `c_concat`, `y`, `guidance`.

        // For Chroma, `context` is `txt_embeddings`.
        // `c_concat` and `y` are not directly used by Chroma's UNet.
        // `guidance` is also not directly used by Chroma's UNet.

        // The `chroma_integration_plan.md` states:
        // `Pass the image latent, the T5 embeddings sequence, the timestep, positional embeddings, and the T5 padding mask to the ChromaUNet_ggml forward function.`

        // This means the `ChromaModel::compute` needs to receive `pe` and `t5_padding_mask`.
        // The current `DiffusionModel::compute` signature does not have these.

        // I need to modify the `DiffusionModel` interface to include `pe` and `t5_padding_mask`.
        // This will affect all other DiffusionModel implementations (UNetModel, MMDiTModel, FluxModel).
        // This is a larger change than just implementing ChromaModel.

        // Let's check the objective again: "Implement the `ChromaModel` class and integrate it into `StableDiffusionGGML`".
        // It doesn't explicitly say to modify the `DiffusionModel` interface.

        // Alternative: `ChromaRunner` could generate `pe` and `t5_padding_mask` internally.
        // `pe` generation depends on image dimensions and context length.
        // `t5_padding_mask` generation depends on T5 token IDs.

        // The `FluxRunner::build_graph` generates `pe_vec` and then creates `pe` tensor.
        // So, `pe` can be generated inside `ChromaRunner`.

        // For `t5_padding_mask`, it needs `token_ids` which come from `T5Embedder`.
        // `T5Embedder` is part of `cond_stage_model`.
        // The `cond_stage_model->get_learned_condition` returns `SDCondition` which contains `c_crossattn` (T5 embeddings).
        // It does not return `token_ids`.

        // This means `t5_padding_mask` cannot be generated inside `ChromaRunner` without access to `token_ids`.
        // The `generate_t5_padding_mask_ggml` function needs `token_ids`.

        // This implies that `token_ids` (or the `t5_padding_mask` itself) needs to be passed to `ChromaModel::compute`.
        // This means the `DiffusionModel::compute` interface *must* change.

        // Let's assume for now that `t5_padding_mask` and `pe` are passed as part of `context` or `c_concat` or `y`
        // or that the `ChromaRunner` can somehow access them.
        // This is a hacky solution.

        // Let's look at `stable-diffusion.cpp` where `diffusion_model->compute` is called.
        // It passes `x`, `timesteps`, `cond.c_crossattn`, `cond.c_concat`, `cond.c_vector`, `guidance_tensor`.
        // For Chroma, `cond.c_crossattn` is `txt_embeddings`.
        // `cond.c_concat` and `cond.c_vector` are currently unused for Chroma.
        // I can repurpose `cond.c_concat` for `t5_padding_mask` and `cond.c_vector` for `pe`.
        // This is a bit of a hack, but avoids changing the `DiffusionModel` interface for now.

        // Let's assume:
        // `context` (original `c_crossattn`) is `txt_embeddings`
        // `c_concat` is `t5_padding_mask`
        // `y` is `pe`

        // This means I need to modify `stable-diffusion.cpp` to pass these correctly.
        // And `ChromaModel::compute` will interpret them as such.

        chroma.compute(n_threads,
                       x,
                       timesteps,
                       context, // T5 embeddings
                       c_concat, // t5_padding_mask
                       y, // pe
                       output,
                       output_ctx,
                       skip_layers);
    }
};

#endif
