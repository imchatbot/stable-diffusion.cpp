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
        // ... existing comments about repurposing inputs ...

        // Construct timestep_for_approximator_input_vec as per Python logic
        int64_t batch_size = x->ne[3]; // Assuming batch size is the last dim of x (img_latent_tokens)

        // 1. distill_timestep = timestep_embedding(timesteps, 16)
        // raw_timesteps is expected to be a 1D tensor [N] or [1]. Extract the single float value.
        std::vector<float> current_timestep_val = {ggml_get_f32_1d(timesteps, 0)};
        struct ggml_tensor* distill_timestep_tensor = ggml_new_tensor_2d(output_ctx, GGML_TYPE_F32, 16, batch_size);
        set_timestep_embedding(current_timestep_val, distill_timestep_tensor, 16);
        // Permute from [16, batch_size] to [batch_size, 16] to match Python's (batch_size, 16)
        distill_timestep_tensor = ggml_cont(output_ctx, ggml_permute(output_ctx, distill_timestep_tensor, 1, 0, 2, 3));

        // 2. distil_guidance = timestep_embedding(guidance, 16)
        std::vector<float> current_guidance_val = {ggml_get_f32_1d(guidance, 0)};
        struct ggml_tensor* distil_guidance_tensor = ggml_new_tensor_2d(output_ctx, GGML_TYPE_F32, 16, batch_size);
        set_timestep_embedding(current_guidance_val, distil_guidance_tensor, 16);
        // Permute from [16, batch_size] to [batch_size, 16]
        distil_guidance_tensor = ggml_cont(output_ctx, ggml_permute(output_ctx, distil_guidance_tensor, 1, 0, 2, 3));

        // 3. modulation_index = timestep_embedding(torch.arange(mod_index_length), 32)
        // mod_index_length is chroma.chroma_hyperparams.mod_vector_total_indices (344)
        std::vector<float> arange_mod_index(chroma.chroma_hyperparams.mod_vector_total_indices);
        for (int i = 0; i < chroma.chroma_hyperparams.mod_vector_total_indices; ++i) {
            arange_mod_index[i] = (float)i;
        }
        struct ggml_tensor* modulation_index_tensor = ggml_new_tensor_2d(output_ctx, GGML_TYPE_F32, 32, chroma.chroma_hyperparams.mod_vector_total_indices);
        set_timestep_embedding(arange_mod_index, modulation_index_tensor, 32);
        // Permute from [32, mod_index_length] to [mod_index_length, 32] to match Python's (mod_index_length, 32)
        modulation_index_tensor = ggml_cont(output_ctx, ggml_permute(output_ctx, modulation_index_tensor, 1, 0, 2, 3));

        // 4. Broadcast modulation_index: unsqueeze(0).repeat(batch_size, 1, 1)
        // From [mod_index_length, 32] to [batch_size, mod_index_length, 32]
        // Reshape to [32, mod_index_length, 1] then repeat along batch dimension
        modulation_index_tensor = ggml_reshape_3d(output_ctx, modulation_index_tensor, 32, chroma.chroma_hyperparams.mod_vector_total_indices, 1);
        modulation_index_tensor = ggml_repeat(output_ctx, modulation_index_tensor, ggml_new_tensor_3d(output_ctx, GGML_TYPE_F32, 32, chroma.chroma_hyperparams.mod_vector_total_indices, batch_size));
        // Permute back to [batch_size, mod_index_length, 32]
        modulation_index_tensor = ggml_cont(output_ctx, ggml_permute(output_ctx, modulation_index_tensor, 2, 1, 0, 3));

        // 5. Concatenate distill_timestep and distil_guidance: torch.cat([distill_timestep, distil_guidance], dim=1)
        // From [batch_size, 16] and [batch_size, 16] to [batch_size, 32]
        struct ggml_tensor* combined_timestep_guidance = ggml_concat(output_ctx, distill_timestep_tensor, distil_guidance_tensor, 1);

        // 6. Broadcast combined_timestep_guidance: unsqueeze(1).repeat(1, mod_index_length, 1)
        // From [batch_size, 32] to [batch_size, mod_index_length, 32]
        // Reshape to [32, 1, batch_size] then repeat along mod_index_length dimension
        combined_timestep_guidance = ggml_reshape_3d(output_ctx, combined_timestep_guidance, 32, 1, batch_size);
        combined_timestep_guidance = ggml_repeat(output_ctx, combined_timestep_guidance, ggml_new_tensor_3d(output_ctx, GGML_TYPE_F32, 32, chroma.chroma_hyperparams.mod_vector_total_indices, batch_size));
        // Permute back to [batch_size, mod_index_length, 32]
        combined_timestep_guidance = ggml_cont(output_ctx, ggml_permute(output_ctx, combined_timestep_guidance, 2, 1, 0, 3));

        // 7. Final concatenation for input_vec: torch.cat([timestep_guidance, modulation_index], dim=-1)
        // From [batch_size, mod_index_length, 32] and [batch_size, mod_index_length, 32] to [batch_size, mod_index_length, 64]
        struct ggml_tensor* constructed_timestep_for_approximator_input_vec = ggml_concat(output_ctx, combined_timestep_guidance, modulation_index_tensor, 2);

        chroma.compute(n_threads,
                       x, // img_latent_tokens
                       constructed_timestep_for_approximator_input_vec, // constructed input_vec
                       context, // txt_tokens (T5 embeddings)
                       y, // pe (positional embeddings)
                       c_concat, // t5_padding_mask
                       output,
                       output_ctx,
                       skip_layers);
    }
};

#endif
