#ifndef __DIFFUSION_MODEL_H__
#define __DIFFUSION_MODEL_H__

#include "flux.hpp"
#include "mmdit.hpp"
#include "unet.hpp"
#include "chroma.hpp"
#include "ggml_extend.hpp" // Required for set_timestep_embedding

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
        : chroma(backend, tensor_types, "model.diffusion_model",flash_attn) {
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
                 struct ggml_tensor* x, // img_latent_tokens
                 struct ggml_tensor* timesteps, // raw_timesteps
                 struct ggml_tensor* context, // txt_embeddings (T5 embeddings)
                 struct ggml_tensor* c_concat, // t5_padding_mask
                 struct ggml_tensor* y, // pe (positional embeddings)
                 struct ggml_tensor* guidance, // raw_guidance
                 int num_video_frames                      = -1,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL,
                 std::vector<int> skip_layers              = std::vector<int>()) {
        LOG_DEBUG(" Start compute chroma");

        // Debug input tensor shapes using the same pattern as stable-diffusion.cpp
        LOG_DEBUG(" Input tensor shapes:");
        if (x) LOG_DEBUG("  x (img_latent_tokens) shape: %lld %lld %lld %lld, type: %s", x->ne[0], x->ne[1], x->ne[2], x->ne[3], ggml_type_name(x->type)); else LOG_DEBUG("  x (img_latent_tokens) is NULL");
        if (timesteps) LOG_DEBUG("  timesteps shape: %lld %lld %lld %lld, type: %s", timesteps->ne[0], timesteps->ne[1], timesteps->ne[2], timesteps->ne[3], ggml_type_name(timesteps->type)); else LOG_DEBUG("  timesteps is NULL");
        if (context) LOG_DEBUG("  context (txt_embeddings) shape: %lld %lld %lld %lld, type: %s", context->ne[0], context->ne[1], context->ne[2], context->ne[3], ggml_type_name(context->type)); else LOG_DEBUG("  context (txt_embeddings) is NULL");
        if (y) LOG_DEBUG("  y (pe) shape: %lld %lld %lld %lld, type: %s", y->ne[0], y->ne[1], y->ne[2], y->ne[3], ggml_type_name(y->type)); else LOG_DEBUG("  y (pe) is NULL");
        if (guidance) LOG_DEBUG("  guidance shape: %lld %lld %lld %lld, type: %s", guidance->ne[0], guidance->ne[1], guidance->ne[2], guidance->ne[3], ggml_type_name(guidance->type)); else LOG_DEBUG("  guidance is NULL");
        if (c_concat) LOG_DEBUG("  c_concat (t5_padding_mask) shape: %lld %lld %lld %lld, type: %s", c_concat->ne[0], c_concat->ne[1], c_concat->ne[2], c_concat->ne[3], ggml_type_name(c_concat->type)); else LOG_DEBUG("  c_concat (t5_padding_mask) is NULL");
        // Pass raw values to ChromaRunner - all tensor construction happens in build_graph
        chroma.compute(n_threads,
                       x,           // img_latent_tokens
                       context,     // txt_tokens (T5 embeddings)
                       y,           // pe (positional embeddings)
                       c_concat,    // t5_padding_mask
                       timesteps, // raw timestep value
                       guidance, // raw guidance value
                       output,
                       output_ctx,
                       skip_layers);
        
        LOG_DEBUG(" Chroma compute completed successfully");
    }
};

#endif
