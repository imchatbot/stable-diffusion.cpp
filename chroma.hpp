#ifndef __CHROMA_HPP__
#define __CHROMA_HPP__

#define CHROMA_GRAPH_SIZE 10240

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"

namespace Chroma {

    struct ChromaParams;

    __STATIC_INLINE__ struct ggml_tensor* attention(struct ggml_context* ctx,
                                                    struct ggml_tensor* q,
                                                    struct ggml_tensor* k,
                                                    struct ggml_tensor* v,
                                                    struct ggml_tensor* pe,
                                                    int64_t num_heads,
                                                    struct ggml_tensor* attn_mask = NULL) {
        int64_t d_head = q->ne[0];
        int64_t L_q = q->ne[1];
        int64_t N_total_q = q->ne[2]; // This is num_heads * N_batch

        int64_t L_k = k->ne[1];
        // int64_t N_total_k = k->ne[2]; // Should be same as N_total_q for k and v

        int64_t N_batch = N_total_q / num_heads;

        LOG_DEBUG("attention - Input shapes: q[%lld,%lld,%lld] k[%lld,%lld,%lld] v[%lld,%lld,%lld] num_heads=%lld d_head=%lld N_batch=%lld",
                  (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2],
                  (long long)k->ne[0], (long long)k->ne[1], (long long)k->ne[2],
                  (long long)v->ne[0], (long long)v->ne[1], (long long)v->ne[2],
                  (long long)num_heads, (long long)d_head, (long long)N_batch);

        // Prepare q_in for ggml_nn_attention_ext: expected shape [d_head*num_heads, L_q, N_batch]
        struct ggml_tensor* q_4d = ggml_reshape_4d(ctx, q, d_head, L_q, num_heads, N_batch);
        // q_4d is [d_head, L_q, num_heads, N_batch]
        struct ggml_tensor* q_perm = ggml_permute(ctx, q_4d, 0, 2, 1, 3); // permutes to [d_head, num_heads, L_q, N_batch]
        struct ggml_tensor* q_in = ggml_reshape_3d(ctx, ggml_cont(ctx, q_perm), d_head * num_heads, L_q, N_batch);
        LOG_DEBUG("attention - q_in prepared for ext: [%lld,%lld,%lld]", q_in->ne[0], q_in->ne[1], q_in->ne[2]);
    
        // Prepare k_in for ggml_nn_attention_ext: expected shape [d_head*num_heads, L_k, N_batch]
        struct ggml_tensor* k_4d = ggml_reshape_4d(ctx, k, d_head, L_k, num_heads, N_batch);
        struct ggml_tensor* k_perm = ggml_permute(ctx, k_4d, 0, 2, 1, 3);
        struct ggml_tensor* k_in = ggml_reshape_3d(ctx, ggml_cont(ctx, k_perm), d_head * num_heads, L_k, N_batch);
        LOG_DEBUG("attention - k_in prepared for ext: [%lld,%lld,%lld]", k_in->ne[0], k_in->ne[1], k_in->ne[2]);
    
        // Prepare v_in for ggml_nn_attention_ext: expected shape [d_head*num_heads, L_k, N_batch]
        struct ggml_tensor* v_4d = ggml_reshape_4d(ctx, v, d_head, L_k, num_heads, N_batch);
        struct ggml_tensor* v_perm = ggml_permute(ctx, v_4d, 0, 2, 1, 3);
        struct ggml_tensor* v_in = ggml_reshape_3d(ctx, ggml_cont(ctx, v_perm), d_head * num_heads, L_k, N_batch);
        LOG_DEBUG("attention - v_in prepared for ext: [%lld,%lld,%lld]", v_in->ne[0], v_in->ne[1], v_in->ne[2]);
    
        struct ggml_tensor* attn_mask_in = NULL;
        if (attn_mask) {
            LOG_DEBUG("attention - original attn_mask shape: [%lld,%lld,%lld,%lld], n_dims: %d",
                      (long long)attn_mask->ne[0], (long long)attn_mask->ne[1],
                      (long long)attn_mask->ne[2], (long long)attn_mask->ne[3],
                      ggml_n_dims(attn_mask));

            // Handle different mask formats and dimensions
            if (ggml_n_dims(attn_mask) == 2) {
                if (attn_mask->ne[0] == L_k && attn_mask->ne[1] == L_q) {
                    // Reshape 2D mask to 3D: [L_k, L_q, 1]
                    attn_mask_in = ggml_reshape_3d(ctx, attn_mask, L_k, L_q, 1);
                    LOG_DEBUG("attention - Reshaped 2D attn_mask to [%lld,%lld,1]",
                              (long long)attn_mask_in->ne[0], (long long)attn_mask_in->ne[1]);
                } else {
                    LOG_DEBUG("attention - 2D mask dimensions [%lld,%lld] incompatible with L_k=%lld, L_q=%lld. Mask not used.",
                              (long long)attn_mask->ne[0], (long long)attn_mask->ne[1],
                              (long long)L_k, (long long)L_q);
                }
            }
            else if (ggml_n_dims(attn_mask) == 3) {
                if (attn_mask->ne[0] == L_k && attn_mask->ne[1] == L_q) {
                    attn_mask_in = attn_mask;
                    LOG_DEBUG("attention - Using 3D attn_mask [%lld,%lld,%lld]",
                              (long long)attn_mask_in->ne[0], (long long)attn_mask_in->ne[1],
                              (long long)attn_mask_in->ne[2]);
                } else {
                    LOG_DEBUG("attention - 3D mask dimensions [%lld,%lld,%lld] incompatible with L_k=%lld, L_q=%lld. Mask not used.",
                              (long long)attn_mask->ne[0], (long long)attn_mask->ne[1],
                              (long long)attn_mask->ne[2], (long long)L_k, (long long)L_q);
                }
            }
            else if (ggml_n_dims(attn_mask) == 4) {
                // Handle 4D mask by selecting first element in extra dimensions
                if (attn_mask->ne[0] == L_k && attn_mask->ne[1] == L_q) {
                    attn_mask_in = ggml_reshape_3d(ctx,
                        ggml_view_3d(ctx, attn_mask, L_k, L_q, 1,
                                     attn_mask->nb[1], attn_mask->nb[2], 0),
                        L_k, L_q, 1);
                    LOG_DEBUG("attention - Reshaped 4D attn_mask to [%lld,%lld,1]",
                              (long long)attn_mask_in->ne[0], (long long)attn_mask_in->ne[1]);
                } else {
                    LOG_DEBUG("attention - 4D mask leading dimensions [%lld,%lld] incompatible with L_k=%lld, L_q=%lld. Mask not used.",
                              (long long)attn_mask->ne[0], (long long)attn_mask->ne[1],
                              (long long)L_k, (long long)L_q);
                }
            }
            else {
                LOG_DEBUG("attention - Unsupported mask dimensionality: %d. Mask not used.", ggml_n_dims(attn_mask));
            }
        }

        // Call ggml_nn_attention_ext with skip_reshape = false
        LOG_DEBUG("attention - Calling ggml_nn_attention_ext with: q_in[%lld,%lld,%lld] k_in[%lld,%lld,%lld] v_in[%lld,%lld,%lld] mask=%p",
                 (long long)q_in->ne[0], (long long)q_in->ne[1], (long long)q_in->ne[2],
                 (long long)k_in->ne[0], (long long)k_in->ne[1], (long long)k_in->ne[2],
                 (long long)v_in->ne[0], (long long)v_in->ne[1], (long long)v_in->ne[2],
                 attn_mask_in);
                 
        struct ggml_tensor* output = ggml_nn_attention_ext(ctx, q_in, k_in, v_in, num_heads, attn_mask_in, false, false, false);
        
        if (output) {
            LOG_DEBUG("attention - ggml_nn_attention_ext output shape: [%lld,%lld,%lld]",
                     (long long)output->ne[0], (long long)output->ne[1], (long long)output->ne[2]);
        } else {
            LOG_ERROR("attention - ggml_nn_attention_ext returned NULL");
        }
        
        // Output of ggml_nn_attention_ext (skip_reshape=false) is [d_head*num_heads, L_q, N_batch].
        // This is the desired final shape for this attention function.
        return output;
    }

    // Add this helper function to chroma.hpp
    __STATIC_INLINE__ void debug_tensor_shapes(const char* operation, struct ggml_tensor* a, struct ggml_tensor* b = nullptr) {
        if (b) {
            LOG_DEBUG("%s - Tensor A shape: [%lld, %lld, %lld, %lld], Tensor B shape: [%lld, %lld, %lld, %lld]",
                operation,
                (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        } else {
            LOG_DEBUG("%s - Tensor shape: [%lld, %lld, %lld, %lld]",
                operation,
                (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3]);
        }
    }


    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* shift,
                                                   struct ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        // shift: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x     = ggml_add(ctx, x, shift);
        return x;
    }


    struct RMSNorm : public UnaryBlock {
    protected:
        int64_t hidden_size;
        float eps;
        void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") override {
            ggml_type wtype = GGML_TYPE_F32;
            if (tensor_types.count(prefix + "scale"))
                wtype = tensor_types[prefix + "scale"];
            params["scale"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
        }

    public:
        RMSNorm(int64_t hs, float e = 1e-6f) : hidden_size(hs), eps(e) {}
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) override {
            struct ggml_tensor* s = params["scale"];
            x                     = ggml_rms_norm(ctx, x, eps);
            x                     = ggml_mul(ctx, x, s);
            return x;
        }
    };

    struct MLPEmbedder : public UnaryBlock {
    public:
        MLPEmbedder(int64_t id, int64_t hd) {
            blocks["in_layer"]  = std::shared_ptr<GGMLBlock>(new Linear(id, hd, true));
            blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hd, hd, true));
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) override {
            auto il = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
            auto ol = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);
            x       = il->forward(ctx, x);
            x       = ggml_silu_inplace(ctx, x);
            x       = ol->forward(ctx, x);
            return x;
        }
    };

    struct Approximator_ggml : public GGMLBlock {
    public:
        int32_t in_dim, out_dim, hidden_dim, n_layers,mod_vector_total_indices;


        Approximator_ggml(int32_t id, int32_t od, int32_t hd, int32_t nl, int32_t mod_vector_total_indices)
            : in_dim(id), out_dim(od), hidden_dim(hd), n_layers(nl), mod_vector_total_indices(mod_vector_total_indices) {
            blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
            for (int i = 0; i < n_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(hidden_dim, hidden_dim));
                blocks["norms." + std::to_string(i)]  = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_dim));
            }
            blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, ggml_tensor* timestep_vals_tensor, ggml_tensor* guidance_vals_tensor, int64_t batch_size)  {
            auto ip                = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto op                = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            LOG_DEBUG("Approximator_ggml::forward - Computing approximator input tensors");
            // 2. Create timestep embeddings (16-dimensional each)
            // Python: (batch_size, 16) -> GGML: (16, batch_size)
            struct ggml_tensor* timestep_emb_local = ggml_timestep_embedding(ctx, timestep_vals_tensor, 16, 10000);
            struct ggml_tensor* guidance_emb_local = ggml_timestep_embedding(ctx, guidance_vals_tensor, 16, 10000);

            LOG_DEBUG("Concatenate timestep and guidance");
            // 3. Concatenate timestep and guidance: [32, batch_size]
            struct ggml_tensor* timestep_guidance_local = ggml_concat(ctx, timestep_emb_local, guidance_emb_local, 0);

            LOG_DEBUG("Create modulation index range");
            // 4. Create modulation index range: [0, 1, ..., 343] -> GGML: [344]
            struct ggml_tensor* mod_indices_local = ggml_arange(ctx, 0.0f, (float)mod_vector_total_indices, 1.0f);

            LOG_DEBUG("Create modulation index embeddings");
            // 5. Create modulation index embeddings (32-dimensional): [344, 32] logical -> GGML: [32, 344]
            struct ggml_tensor* mod_index_emb_local = ggml_timestep_embedding(ctx, mod_indices_local, 32, 10000);

            // Define the target shape for repetition: [32, 344, batch_size]
            struct ggml_tensor* target_shape_for_repeat = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, mod_vector_total_indices, batch_size);

            LOG_DEBUG("Reshape concatenated timestep/guidance for broadcasting");
            // 6. Reshape concatenated timestep/guidance to [32, 1, batch_size] for broadcasting
            struct ggml_tensor* reshaped_tg_local = ggml_reshape_3d(ctx, timestep_guidance_local, 32, 1, batch_size);

            LOG_DEBUG("Repeat concatenated timestep/guidance");
            // 7. Repeat concatenated timestep/guidance to [32, 344, batch_size]
            struct ggml_tensor* repeated_timestep_guidance_local = ggml_repeat(ctx, reshaped_tg_local, target_shape_for_repeat);

            LOG_DEBUG("Reshape modulation index embeddings for broadcasting");
            // 8. Reshape modulation index embeddings to [32, 344, 1] for broadcasting
            struct ggml_tensor* reshaped_mod_local = ggml_reshape_3d(ctx, mod_index_emb_local, 32, mod_vector_total_indices, 1);

            LOG_DEBUG("Repeat modulation index embeddings");
            // 9. Repeat modulation index embeddings to [32, 344, batch_size]
            struct ggml_tensor* repeated_mod_index_emb_local = ggml_repeat(ctx, reshaped_mod_local, target_shape_for_repeat);

            LOG_DEBUG("Concatenate the two repeated tensors along dimension 0");
            // 10. Concatenate the two repeated tensors along dimension 0: [64, 344, batch_size]
            struct ggml_tensor* approximator_input = ggml_concat(ctx, repeated_timestep_guidance_local, repeated_mod_index_emb_local, 0);

            LOG_DEBUG("Approximator_ggml::forward - Approximator input shape: [%lld, %lld, %lld]",
                (long long)approximator_input->ne[2], (long long)approximator_input->ne[1], (long long)approximator_input->ne[0]);
            // === END APPROXIMATOR INPUT CONSTRUCTION ===

            struct ggml_tensor* ci = ip->forward(ctx, approximator_input);
            for (int i = 0; i < n_layers; ++i) {
                auto l = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
                auto n = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);
                ci     = ggml_add(ctx, ci, l->forward(ctx, n->forward(ctx, ci)));
            }
            return op->forward(ctx, ci);
        }
    };

    struct QKNorm : public GGMLBlock {
        int64_t head_dim;
        QKNorm(int64_t hd) : head_dim(hd) {
            blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim));
            blocks["key_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim));
        }
        struct ggml_tensor* query_norm(struct ggml_context* ctx, struct ggml_tensor* x) { return std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"])->forward(ctx, x); }
        struct ggml_tensor* key_norm(struct ggml_context* ctx, struct ggml_tensor* x) { return std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"])->forward(ctx, x); }
    };

    struct SelfAttention : public GGMLBlock {
        int64_t dim, num_heads, head_dim;
        bool qkv_bias;
        SelfAttention(int64_t d, int64_t nh, bool bias = false) : dim(d), num_heads(nh), qkv_bias(bias) {
            GGML_ASSERT(dim % num_heads == 0);
            head_dim       = dim / num_heads;
            blocks["qkv"]  = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
            blocks["norm"] = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
            blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, true));
        }
        std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
            auto qp   = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
            auto nb   = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto qc   = qp->forward(ctx, x);
            int64_t N = x->ne[2], L = x->ne[1];
            qc         = ggml_cont(ctx, ggml_permute(ctx, qc, 2, 1, 0, 3));
            qc         = ggml_reshape_4d(ctx, qc, dim, 3, L, N);
            int64_t s2 = qc->nb[2], s1 = qc->nb[1];
            auto q_ = ggml_view_3d(ctx, qc, dim, L, N, s2, qc->nb[3], 0 * s1);
            auto k_ = ggml_view_3d(ctx, qc, dim, L, N, s2, qc->nb[3], 1 * s1);
            auto v_ = ggml_view_3d(ctx, qc, dim, L, N, s2, qc->nb[3], 2 * s1);
            q_      = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, q_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            k_      = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            v_      = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            return {nb->query_norm(ctx, q_), nb->key_norm(ctx, k_), v_};
        }
        struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) { return std::dynamic_pointer_cast<Linear>(blocks["proj"])->forward(ctx, x); }
    };

    struct ModulationOut {
        ggml_tensor* shift = NULL;
        ggml_tensor* scale = NULL;
        ggml_tensor* gate  = NULL;
        ModulationOut(ggml_tensor* s = 0, ggml_tensor* sc = 0, ggml_tensor* g = 0) : shift(s), scale(sc), gate(g) {}
    };

    enum class ModulationOutputType { SINGLE,
                                      DOUBLE,
                                      FINAL };
    struct BlockModulationOutput {
        ModulationOutputType type;
        ModulationOut single_mod;
        std::pair<ModulationOut, ModulationOut> double_mods;
        std::pair<ggml_tensor*, ggml_tensor*> final_mods;
    };

    struct ChromaParams {
        int32_t in_channels                      = 64;
        int32_t out_channels                     = 64;
        int32_t t5_embed_dim                     = 4096;
        int32_t approximator_input_concat_dim    = 64;  // Changed from (16 + 16 + 32)
        int32_t approximator_internal_hidden_dim = 5120;
        int32_t approximator_feature_dim         = 3072;
        int32_t unet_model_dim                   = 3072;
        int32_t num_heads                        = 24;
        float mlp_ratio                          = 4.0f;
        int32_t depth                            = 19;
        int32_t depth_single_blocks              = 38;
        int32_t mod_vector_total_indices         = 344;
        int32_t head_dim;
        bool flash_attn = false;
        ChromaParams() {
            if (num_heads > 0 && unet_model_dim > 0 && unet_model_dim % num_heads == 0) {
                head_dim = unet_model_dim / num_heads;
            } else {
                head_dim = 128;
            }
        }
    };

    __STATIC_INLINE__ ModulationOut get_modulation_params_from_offset(
            ggml_context* ctx,
            ggml_tensor* mod_vectors_permuted,
            int offset_idx,
            int64_t feature_dim,
            int64_t N_batch_size) {
            size_t col_stride = mod_vectors_permuted->nb[1];
            // Create contiguous 2D tensors
            ggml_tensor* s    = ggml_cont(ctx, ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)offset_idx * col_stride));
            ggml_tensor* sc   = ggml_cont(ctx, ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)(offset_idx + 1) * col_stride));
            ggml_tensor* g    = ggml_cont(ctx, ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)(offset_idx + 2) * col_stride));
            return ModulationOut(s, sc, g);
    }

    __STATIC_INLINE__ BlockModulationOutput get_modulations_for_block(
        ggml_context* ctx,
        const ChromaParams& p,
        ggml_tensor* mod_vectors_input,
        const std::string& bt,
        int bi) {
        BlockModulationOutput o;
        ggml_tensor* mod_vectors = ggml_cont(ctx, ggml_permute(ctx, mod_vectors_input, 2, 1, 0, 3));
        int64_t N_batch          = mod_vectors_input->ne[2];  // N from input [N, L_idx, D_feat]
                                                              // after permute mod_vectors is [D_feat, L_idx, N] -> N is ne[2]
        if (bt == "final") {
            o.type           = ModulationOutputType::FINAL;
            int64_t li       = p.mod_vector_total_indices - 1;
            int64_t sli      = p.mod_vector_total_indices - 2;
            size_t cs        = mod_vectors->nb[1];
            ggml_tensor* scf = ggml_view_2d(ctx, mod_vectors, p.approximator_feature_dim, N_batch, cs, (size_t)sli * cs);
            ggml_tensor* sf  = ggml_view_2d(ctx, mod_vectors, p.approximator_feature_dim, N_batch, cs, (size_t)li * cs);
            scf              = ggml_cont(ctx, ggml_permute(ctx, scf, 1, 0, 2, 3));
            sf               = ggml_cont(ctx, ggml_permute(ctx, sf, 1, 0, 2, 3));
            o.final_mods     = std::make_pair(sf, scf);
            return o;  // Python: (shift, scale)
        }
        int sc  = p.depth_single_blocks;
        int dc  = p.depth;
        int off = 0;
        if (bt == "single") {
            o.type       = ModulationOutputType::SINGLE;
            off          = 3 * bi;
            o.single_mod = get_modulation_params_from_offset(ctx, mod_vectors, off, p.approximator_feature_dim, N_batch);
            return o;
        }
        off = 6 * bi;  // For double blocks, each block idx means 6 params (mod1 + mod2)
        if (bt == "double_img") {
            o.type = ModulationOutputType::DOUBLE;
            off += 3 * sc;
        } else if (bt == "double_txt") {
            o.type = ModulationOutputType::DOUBLE;
            off += 3 * sc + 6 * dc;
        } else {
            throw std::runtime_error("Bad block_type: " + bt);
        }
        ModulationOut m1 = get_modulation_params_from_offset(ctx, mod_vectors, off, p.approximator_feature_dim, N_batch);
        ModulationOut m2 = get_modulation_params_from_offset(ctx, mod_vectors, off + 3, p.approximator_feature_dim, N_batch);
        o.double_mods    = std::make_pair(m1, m2);
        return o;
    }

    struct SingleStreamBlock_ggml : public GGMLBlock {
        int64_t hidden_size, num_heads, head_dim;
        float mlp_ratio;
        SingleStreamBlock_ggml(int64_t hs, int64_t nh, float mr)
            : hidden_size(hs), num_heads(nh), mlp_ratio(mr) {
            GGML_ASSERT(hs % nh == 0);
            head_dim           = hs / nh;
            int64_t mlp_hd     = static_cast<int64_t>(hs * mr);
            blocks["pre_norm"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["linear1"]  = std::make_shared<Linear>(hs, hs * 3 + mlp_hd, true);
            blocks["norm"]     = std::make_shared<QKNorm>(head_dim);
            blocks["linear2"]  = std::make_shared<Linear>(hs + mlp_hd, hs, true);
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pe, const ModulationOut& vm, struct ggml_tensor* am = 0) {
            auto pn                 = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);
            auto l1                 = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
            auto qn                 = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
            auto l2                 = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
            struct ggml_tensor* xn  = pn->forward(ctx, x);
            struct ggml_tensor* xm  = Chroma::modulate(ctx, xn, vm.shift, vm.scale);
            struct ggml_tensor* l1o = l1->forward(ctx, xm);
            int64_t N = x->ne[2], L = x->ne[1], mlp_hda = static_cast<int64_t>(hidden_size * mlp_ratio);
            struct ggml_tensor* pl1o  = ggml_cont(ctx, ggml_permute(ctx, l1o, 2, 1, 0, 3));
            struct ggml_tensor* qkvp  = ggml_view_3d(ctx, pl1o, hidden_size * 3, L, N, pl1o->nb[1], pl1o->nb[2], 0);
            qkvp                      = ggml_cont(ctx, ggml_permute(ctx, qkvp, 2, 1, 0, 3));
            struct ggml_tensor* mlpp  = ggml_view_3d(ctx, pl1o, mlp_hda, L, N, pl1o->nb[1], pl1o->nb[2], pl1o->nb[0] * (hidden_size * 3));
            mlpp                      = ggml_cont(ctx, ggml_permute(ctx, mlpp, 2, 1, 0, 3));
            struct ggml_tensor* qkvpr = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, qkvp, 2, 1, 0, 3)), hidden_size, 3, L, N);
            int64_t s2q = qkvpr->nb[2], s1q = qkvpr->nb[1];
            auto q_                  = ggml_view_3d(ctx, qkvpr, hidden_size, L, N, s2q, qkvpr->nb[3], 0 * s1q);
            auto k_                  = ggml_view_3d(ctx, qkvpr, hidden_size, L, N, s2q, qkvpr->nb[3], 1 * s1q);
            auto v_                  = ggml_view_3d(ctx, qkvpr, hidden_size, L, N, s2q, qkvpr->nb[3], 2 * s1q);
            q_                       = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, q_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            k_                       = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            v_                       = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v_, 1, 2, 0, 3)), head_dim, L, N * num_heads);
            q_                       = qn->query_norm(ctx, q_);
            k_                       = qn->key_norm(ctx, k_);
            struct ggml_tensor* ao   = Chroma::attention(ctx, q_, k_, v_, pe, num_heads, am);
            struct ggml_tensor* mao  = ggml_gelu_inplace(ctx, mlpp);
            // Log tensor shapes before concatenation
            LOG_DEBUG("SingleStreamBlock_ggml::forward - ao shape: [%lld, %lld, %lld]",
                     (long long)ao->ne[0], (long long)ao->ne[1], (long long)ao->ne[2]);
            // Current mao shape is [N, L, C_mlp], e.g., [16, 320, 12288]
            // ao shape is [C_attn, L, N], e.g., [3072, 320, 16]
            // We need to permute mao to [C_mlp, L, N] before concatenating with ao along dim 0.
            struct ggml_tensor* mao_for_concat = ggml_permute(ctx, mao, 2, 1, 0, 3);
            mao_for_concat = ggml_cont(ctx, mao_for_concat);

            struct ggml_tensor* co   = ggml_concat(ctx, ao, mao_for_concat, 0);
            
            struct ggml_tensor* l2o  = l2->forward(ctx, co);
            struct ggml_tensor* gadd = ggml_mul(ctx, l2o, vm.gate);
            struct ggml_tensor* o    = x;
            if (ggml_are_same_shape(x, gadd)) {
                o = ggml_add(ctx, x, gadd);
            } else {
                fprintf(stderr, "SSB skip shape mismatch x:[%lld,%lld,%lld] vs gadd:[%lld,%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)gadd->ne[0], (long long)gadd->ne[1], (long long)gadd->ne[2]);
            }
            LOG_DEBUG("End of SingleStreamBlock_ggml");
            return o;
        }
    };

    struct DoubleStreamBlock_ggml : public GGMLBlock {
        int64_t hidden_size, num_heads;
        float mlp_ratio;
        bool qkv_bias;
        DoubleStreamBlock_ggml(int64_t hs, int64_t nh, float mr, bool qb = true)
            : hidden_size(hs), num_heads(nh), mlp_ratio(mr), qkv_bias(qb) {
            int64_t mlp_hd      = static_cast<int64_t>(hs * mr);
            blocks["img_norm1"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["img_attn"]  = std::make_shared<SelfAttention>(hs, nh, qb);
            blocks["img_norm2"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["img_mlp.0"] = std::make_shared<Linear>(hs, mlp_hd, true);
            blocks["img_mlp.2"] = std::make_shared<Linear>(mlp_hd, hs, true);
            blocks["txt_norm1"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["txt_attn"]  = std::make_shared<SelfAttention>(hs, nh, qb);
            blocks["txt_norm2"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["txt_mlp.0"] = std::make_shared<Linear>(hs, mlp_hd, true);
            blocks["txt_mlp.2"] = std::make_shared<Linear>(mlp_hd, hs, true);
        }
        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(
            struct ggml_context* ctx,
            struct ggml_tensor* img,
            struct ggml_tensor* txt,
            struct ggml_tensor* pe,
            const std::pair<ModulationOut, ModulationOut>& vim,
            const std::pair<ModulationOut, ModulationOut>& vtm,
            struct ggml_tensor* am = 0) {
            ModulationOut im1     = vim.first;
            ModulationOut im2_mod = vim.second;
            ModulationOut tm1     = vtm.first;
            ModulationOut tm2_mod = vtm.second;
            auto in1              = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto ia               = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);
            auto in2              = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto im0_linear       = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.0"]);  // Corrected variable name
            auto im2_linear       = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.2"]);  // Corrected variable name
            auto tn1              = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto ta               = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);
            auto tn2              = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto tm0_linear       = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.0"]);  // Corrected variable name
            auto tm2_linear       = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.2"]);  // Corrected variable name

            // Log input shapes
            LOG_DEBUG("DoubleStreamBlock_ggml::forward - img input shape: [%lld, %lld, %lld]",
                     (long long)img->ne[0], (long long)img->ne[1], (long long)img->ne[2]);
            LOG_DEBUG("DoubleStreamBlock_ggml::forward - txt input shape: [%lld, %lld, %lld]",
                     (long long)txt->ne[0], (long long)txt->ne[1], (long long)txt->ne[2]);
            
            struct ggml_tensor* imna = in1->forward(ctx, img);
            LOG_DEBUG("imna shape: [%lld, %lld, %lld]",
                     (long long)imna->ne[0], (long long)imna->ne[1], (long long)imna->ne[2]);
            
            imna = Chroma::modulate(ctx, imna, im1.shift, im1.scale);
            LOG_DEBUG("imna after modulate: [%lld, %lld, %lld]",
                     (long long)imna->ne[0], (long long)imna->ne[1], (long long)imna->ne[2]);
            
            auto iqkv = ia->pre_attention(ctx, imna);
            LOG_DEBUG("iqkv[0] shape: [%lld, %lld, %lld]",
                     (long long)iqkv[0]->ne[0], (long long)iqkv[0]->ne[1], (long long)iqkv[0]->ne[2]);
            LOG_DEBUG("iqkv[1] shape: [%lld, %lld, %lld]",
                     (long long)iqkv[1]->ne[0], (long long)iqkv[1]->ne[1], (long long)iqkv[1]->ne[2]);
            LOG_DEBUG("iqkv[2] shape: [%lld, %lld, %lld]",
                     (long long)iqkv[2]->ne[0], (long long)iqkv[2]->ne[1], (long long)iqkv[2]->ne[2]);
            
            struct ggml_tensor* tmna = tn1->forward(ctx, txt);
            LOG_DEBUG("tmna shape: [%lld, %lld, %lld]",
                     (long long)tmna->ne[0], (long long)tmna->ne[1], (long long)tmna->ne[2]);
            
            tmna = Chroma::modulate(ctx, tmna, tm1.shift, tm1.scale);
            LOG_DEBUG("tmna after modulate: [%lld, %lld, %lld]",
                     (long long)tmna->ne[0], (long long)tmna->ne[1], (long long)tmna->ne[2]);
            
            auto tqkv = ta->pre_attention(ctx, tmna);
            LOG_DEBUG("tqkv[0] shape: [%lld, %lld, %lld]",
                     (long long)tqkv[0]->ne[0], (long long)tqkv[0]->ne[1], (long long)tqkv[0]->ne[2]);
            LOG_DEBUG("tqkv[1] shape: [%lld, %lld, %lld]",
                     (long long)tqkv[1]->ne[0], (long long)tqkv[1]->ne[1], (long long)tqkv[1]->ne[2]);
            LOG_DEBUG("tqkv[2] shape: [%lld, %lld, %lld]",
                     (long long)tqkv[2]->ne[0], (long long)tqkv[2]->ne[1], (long long)tqkv[2]->ne[2]);
            
            // Process text and image streams separately
            LOG_DEBUG("Calling attention for image stream");
            auto iap = Chroma::attention(ctx, iqkv[0], iqkv[1], iqkv[2], pe, num_heads, am);
            LOG_DEBUG("iap shape: [%lld, %lld, %lld]",
                     (long long)iap->ne[0], (long long)iap->ne[1], (long long)iap->ne[2]);
            
            LOG_DEBUG("Calling attention for text stream");
            auto tap = Chroma::attention(ctx, tqkv[0], tqkv[1], tqkv[2], pe, num_heads, am);
            LOG_DEBUG("tap shape: [%lld, %lld, %lld]",
                     (long long)tap->ne[0], (long long)tap->ne[1], (long long)tap->ne[2]);
            
            auto ira = ia->post_attention(ctx, iap);
            LOG_DEBUG("ira shape: [%lld, %lld, %lld]",
                     (long long)ira->ne[0], (long long)ira->ne[1], (long long)ira->ne[2]);
            
            img = ggml_add(ctx, img, ggml_mul(ctx, ira, im1.gate));
            LOG_DEBUG("img after attention: [%lld, %lld, %lld]",
                     (long long)img->ne[0], (long long)img->ne[1], (long long)img->ne[2]);
            
            auto imn2 = in2->forward(ctx, img);
            LOG_DEBUG("imn2 shape: [%lld, %lld, %lld]",
                     (long long)imn2->ne[0], (long long)imn2->ne[1], (long long)imn2->ne[2]);
            
            imn2 = Chroma::modulate(ctx, imn2, im2_mod.shift, im2_mod.scale);
            LOG_DEBUG("imn2 after modulate: [%lld, %lld, %lld]",
                     (long long)imn2->ne[0], (long long)imn2->ne[1], (long long)imn2->ne[2]);
            
            auto imh = ggml_gelu_inplace(ctx, im0_linear->forward(ctx, imn2));
            LOG_DEBUG("imh shape: [%lld, %lld, %lld]",
                     (long long)imh->ne[0], (long long)imh->ne[1], (long long)imh->ne[2]);
            
            auto imf = im2_linear->forward(ctx, imh);
            LOG_DEBUG("imf shape: [%lld, %lld, %lld]",
                     (long long)imf->ne[0], (long long)imf->ne[1], (long long)imf->ne[2]);
            
            img = ggml_add(ctx, img, ggml_mul(ctx, imf, im2_mod.gate));
            LOG_DEBUG("img final shape: [%lld, %lld, %lld]",
                     (long long)img->ne[0], (long long)img->ne[1], (long long)img->ne[2]);
            
            auto tra = ta->post_attention(ctx, tap);
            LOG_DEBUG("tra shape: [%lld, %lld, %lld]",
                     (long long)tra->ne[0], (long long)tra->ne[1], (long long)tra->ne[2]);
            
            txt = ggml_add(ctx, txt, ggml_mul(ctx, tra, tm1.gate));
            LOG_DEBUG("txt after attention: [%lld, %lld, %lld]",
                     (long long)txt->ne[0], (long long)txt->ne[1], (long long)txt->ne[2]);
            
            auto tmn2 = tn2->forward(ctx, txt);
            LOG_DEBUG("tmn2 shape: [%lld, %lld, %lld]",
                     (long long)tmn2->ne[0], (long long)tmn2->ne[1], (long long)tmn2->ne[2]);
            
            tmn2 = Chroma::modulate(ctx, tmn2, tm2_mod.shift, tm2_mod.scale);
            LOG_DEBUG("tmn2 after modulate: [%lld, %lld, %lld]",
                     (long long)tmn2->ne[0], (long long)tmn2->ne[1], (long long)tmn2->ne[2]);
            
            auto tmh = ggml_gelu_inplace(ctx, tm0_linear->forward(ctx, tmn2));
            LOG_DEBUG("tmh shape: [%lld, %lld, %lld]",
                     (long long)tmh->ne[0], (long long)tmh->ne[1], (long long)tmh->ne[2]);
            
            auto tmf = tm2_linear->forward(ctx, tmh);
            LOG_DEBUG("tmf shape: [%lld, %lld, %lld]",
                     (long long)tmf->ne[0], (long long)tmf->ne[1], (long long)tmf->ne[2]);
            
            txt = ggml_add(ctx, txt, ggml_mul(ctx, tmf, tm2_mod.gate));
            LOG_DEBUG("txt final shape: [%lld, %lld, %lld]",
                     (long long)txt->ne[0], (long long)txt->ne[1], (long long)txt->ne[2]);
            
            return {img, txt};
        }
    };

    struct LastLayer_ggml : public GGMLBlock {
        int64_t hidden_size, out_channels;
        LastLayer_ggml(int64_t hs, int64_t oc) : hidden_size(hs), out_channels(oc) {
            blocks["norm_final"] = std::make_shared<LayerNorm>(hs, 1e-6f, false);
            blocks["linear"]     = std::make_shared<Linear>(hs, oc, true);
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* shift, struct ggml_tensor* scale) {
            LOG_DEBUG("LastLayer_ggml::forward - start forward lastlayer");

            auto nf = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto ln = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto xn = nf->forward(ctx, x);

            print_ggml_tensor(xn,true,"xn");
            struct ggml_tensor* xn_nlc = ggml_permute(ctx, xn, 2, 1, 0, 3);
            // Ensure shift/scale are [N, C] for modulate function to reshape to [N,1,C]
            // Python: shift = shift.squeeze(1) # [N,1,C] -> [N,C]
            // Python: scale = scale.squeeze(1)
            // The get_modulations_for_block for "final" already returns them as [N, C]
            auto xm = Chroma::modulate(ctx, xn_nlc, shift, scale);
            print_ggml_tensor(xm,true,"xm");
            auto xm_nlc = Chroma::modulate(ctx, xn_nlc, shift, scale);
            LOG_DEBUG("LastLayer_ggml::forward - xm_nlc after modulate (N,L,C): [%lld,%lld,%lld]", (long long)xm_nlc->ne[0], (long long)xm_nlc->ne[1], (long long)xm_nlc->ne[2]);
            struct ggml_tensor* xm_cln = ggml_permute(ctx, xm_nlc, 2, 1, 0, 3); // N,L,C -> C,L,N
            xm_cln = ggml_cont(ctx, xm_cln);
            LOG_DEBUG("LastLayer_ggml::forward - before ln last layer");

            return ln->forward(ctx, xm_cln);
        }
    };

    struct ChromaUNet_ggml : public GGMLBlock {
        ChromaParams params_unet;


        ChromaUNet_ggml(){}
        ChromaUNet_ggml(const ChromaParams& p) : params_unet(p) {
            blocks["distilled_guidance_layer"] = std::make_shared<Approximator_ggml>(p.approximator_input_concat_dim, p.approximator_feature_dim, p.approximator_internal_hidden_dim,5,p.mod_vector_total_indices);
            blocks["img_in"]                   = std::make_shared<Linear>(p.in_channels, p.unet_model_dim, true);
            blocks["txt_in"]                   = std::make_shared<Linear>(p.t5_embed_dim, p.unet_model_dim, true);
            for (int i = 0; i < p.depth; ++i)
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock_ggml>(p.unet_model_dim, p.num_heads, p.mlp_ratio, true);
            for (int i = 0; i < p.depth_single_blocks; ++i)
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock_ggml>(p.unet_model_dim, p.num_heads, p.mlp_ratio);
            blocks["final_layer"] = std::make_shared<LastLayer_ggml>(p.unet_model_dim, p.out_channels);
        }

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* ilt, ggml_tensor* timestep_val, ggml_tensor* guidance_val, struct ggml_tensor* te, struct ggml_tensor* pe, struct ggml_tensor* t5pm, std::vector<int> sl = {}) {
            LOG_DEBUG("ChromaUNet_ggml::forward - Starting forward pass");
            
            auto ap                 = std::dynamic_pointer_cast<Approximator_ggml>(blocks["distilled_guidance_layer"]);
            auto img_ip             = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_ip             = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_l            = std::dynamic_pointer_cast<LastLayer_ggml>(blocks["final_layer"]);
            int64_t batch_size = ilt->ne[3];
           
            struct ggml_tensor* mvg = ap->forward(ctx, timestep_val,guidance_val,batch_size); // Use the computed approximator_input
            
            LOG_DEBUG("ChromaUNet_ggml::forward - Processing input layers");
            struct ggml_tensor* cit = img_ip->forward(ctx, ilt);

            struct ggml_tensor* ctt = txt_ip->forward(ctx, te);
            LOG_DEBUG("ChromaUNet_ggml::forward - Starting double blocks processing (depth=%d)", params_unet.depth);
            for (int i = 0; i < params_unet.depth; ++i) {
                if (!sl.empty() && std::find(sl.begin(), sl.end(), i) != sl.end())
                    continue;
                auto b                    = std::dynamic_pointer_cast<DoubleStreamBlock_ggml>(blocks["double_blocks." + std::to_string(i)]);
                BlockModulationOutput imo = get_modulations_for_block(ctx, params_unet, mvg, "double_img", i);
                BlockModulationOutput tmo = get_modulations_for_block(ctx, params_unet, mvg, "double_txt", i);
                auto& imp                 = imo.double_mods;
                auto& tmp                 = tmo.double_mods;
                auto pt                   = b->forward(ctx, cit, ctt, pe, imp, tmp, t5pm);
                cit                       = pt.first;
                ctt                       = pt.second;
            }
            
            LOG_DEBUG("ChromaUNet_ggml::forward - Concatenating for single blocks");
            // Broadcast text tensor to match image tensor's batch size
            // Validate and broadcast text tensor
            if (ctt->ne[2] != cit->ne[2]) {
                if (ctt->ne[2] == 1) {
                    LOG_DEBUG("Broadcasting text tensor from batch size %lld to %lld",
                             (long long)ctt->ne[2], (long long)cit->ne[2]);
                    ctt = ggml_repeat(ctx, ctt, ggml_new_tensor_3d(ctx, ctt->type, ctt->ne[0], ctt->ne[1], cit->ne[2]));
                } else {
                    LOG_ERROR("Text and image batch sizes incompatible: text=%lld, image=%lld",
                             (long long)ctt->ne[2], (long long)cit->ne[2]);
                    // We cannot proceed, so we return a tensor of zeros? Or let it crash?
                    // For now, we'll just use the original ctt and hope for the best, but log an error.
                }
            }
            
            struct ggml_tensor* combt = ggml_concat(ctx, ctt, cit, 1);
            LOG_DEBUG("Concatenated tensor shape: [%lld,%lld,%lld]",
                     (long long)combt->ne[0], (long long)combt->ne[1], (long long)combt->ne[2]);
            
            LOG_DEBUG("ChromaUNet_ggml::forward - Starting single blocks processing (depth_single=%d)", params_unet.depth_single_blocks);
            for (int i = 0; i < params_unet.depth_single_blocks; ++i) {
                if (!sl.empty() && std::find(sl.begin(), sl.end(), i + params_unet.depth) != sl.end())
                    continue;
                auto b                    = std::dynamic_pointer_cast<SingleStreamBlock_ggml>(blocks["single_blocks." + std::to_string(i)]);
                BlockModulationOutput smo = get_modulations_for_block(ctx, params_unet, mvg, "single", i);
                auto& sm                  = smo.single_mod;
                combt                     = b->forward(ctx, combt, pe, sm, t5pm);
            }
            
            LOG_DEBUG("ChromaUNet_ggml::forward - Processing final layer");
            int64_t ntt               = ctt->ne[1];
            
            struct ggml_tensor* fit   = ggml_view_3d(ctx, combt, combt->ne[0], combt->ne[1] - ntt, combt->ne[2], combt->nb[1], combt->nb[2], ntt * combt->nb[1]);
            print_ggml_tensor(fit,true,"fit");
            BlockModulationOutput fmo = get_modulations_for_block(ctx, params_unet, mvg, "final", 0);
            auto& fmp                 = fmo.final_mods;


            struct ggml_tensor* result = final_l->forward(ctx, fit, fmp.first, fmp.second);
            print_ggml_tensor(result,true,"fit");
            LOG_DEBUG("ChromaUNet_ggml::forward - Forward pass completed successfully");
            return result;
        }
    };

    struct ChromaRunner : public GGMLRunner {
        ChromaParams chroma_hyperparams;
        ChromaUNet_ggml chroma;


        ChromaRunner(ggml_backend_t b, std::map<std::string, enum ggml_type>& tt, const std::string pr = "", bool ufa = false)
            : GGMLRunner(b), chroma_hyperparams(), chroma(chroma_hyperparams) {
            chroma_hyperparams.flash_attn = ufa;
            chroma.init(params_ctx, tt, pr);

        }


        std::string get_desc()  { return "chroma"; }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& t, const std::string p) { chroma.get_param_tensors(t, p); }
        
        struct ggml_cgraph* build_graph(
            struct ggml_tensor* img_latent_tokens, // [N, L_img, C_in_channels]
            struct ggml_tensor* timesteps,         // [N] or scalar
            struct ggml_tensor* guidance,          // [N] or scalar
            struct ggml_tensor* txt_tokens,        // [N, L_txt, C_txt_in_to_unet] (after T5 + projection)
            struct ggml_tensor* pe,                // Positional embeddings
            struct ggml_tensor* t5_padding_mask,   // Attention mask
            std::vector<int> skip_layers = {}
        ) {
            LOG_DEBUG("ChromaRunner::build_graph - Starting graph construction");
            print_ggml_tensor(img_latent_tokens,true,"img_latent_tokens");
            print_ggml_tensor(timesteps,true,"timesteps");
            print_ggml_tensor(guidance,true,"guidance");
            print_ggml_tensor(txt_tokens,true,"txt_tokens");
            print_ggml_tensor(pe,true,"pe");
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, CHROMA_GRAPH_SIZE, false);
  

            img_latent_tokens = to_backend(img_latent_tokens);
            timesteps         = to_backend(timesteps);
            guidance          = to_backend(guidance);
            txt_tokens        = to_backend(txt_tokens);
            if (pe) pe        = to_backend(pe);
            if (t5_padding_mask) t5_padding_mask = to_backend(t5_padding_mask);
 

            struct ggml_tensor* output = chroma.forward(
                compute_ctx, img_latent_tokens, timesteps,guidance, txt_tokens, pe, t5_padding_mask, skip_layers
            );
            LOG_DEBUG("ChromaRunner::build_graph - chroma_unet.forward completed");

            ggml_build_forward_expand(gf, output);
            LOG_DEBUG("ChromaRunner::build_graph - Graph construction completed successfully");
            return gf;
        }
        void compute(
                int n_threads,
                struct ggml_tensor* img_latent_tokens,
                struct ggml_tensor* timesteps,
                struct ggml_tensor* guidance,
                struct ggml_tensor* txt_tokens,
                struct ggml_tensor* pe,
                struct ggml_tensor* t5_padding_mask,
                struct ggml_tensor** output = NULL,
                ggml_context* output_ctx = NULL, // Fix: Removed struct
                std::vector<int> skip_layers = {}
            ) {
                auto get_graph_fn = [&]() -> struct ggml_cgraph* { // Renamed lambda variable
                    return build_graph(img_latent_tokens, timesteps, guidance, txt_tokens, pe, t5_padding_mask, skip_layers);
                };
                GGMLRunner::compute(get_graph_fn, n_threads, false, output, output_ctx); // Use renamed lambda
            }
    };

}  // namespace Chroma
#endif  // __CHROMA_HPP__ 
-> 
struct ChromaT5Embedder : public Conditioner {
    T5UniGramTokenizer t5_tokenizer;
    std::shared_ptr<T5Runner> t5;

    ChromaT5Embedder(ggml_backend_t backend,
                    std::map<std::string, enum ggml_type>& tensor_types)
    { // Initialize prefix_
        t5 = std::make_shared<T5Runner>(backend, tensor_types, "text_encoders.t5xxl.transformer");
    }

    SDCondition get_learned_condition(ggml_context* work_ctx,
                                      int n_threads,
                                      const std::string& text,
                                      int clip_skip, // Not used by T5
                                      int width,     // Not used by T5
                                      int height,    // Not used by T5
                                      int adm_in_channels        = -1, // Not used by T5
                                      bool force_zero_embeddings = false)  {
        // Tokenize the text using T5UniGramTokenizer
        auto parsed_attention = parse_prompt_attention(text);
        std::vector<int> tokens;
        std::vector<float> weights;

        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = t5_tokenizer.Encode(curr_text, false);
            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        // Add EOS token and pad
        int EOS_TOKEN_ID = 1; // Assuming EOS_TOKEN_ID for T5 is 1
        tokens.push_back(EOS_TOKEN_ID);
        weights.push_back(1.0);
        t5_tokenizer.pad_tokens(tokens, weights, 256, true); // Max length 256 for T5, enable padding

        // Create input_ids tensor from tokens
        struct ggml_tensor* input_ids = vector_to_ggml_tensor_i32(work_ctx, tokens);
        struct ggml_tensor* hidden_states = NULL;

        // Compute T5 embeddings
        t5->compute(n_threads, input_ids, &hidden_states, work_ctx);

        // Apply weights to hidden_states, similar to FluxCLIPEmbedder
        if (!force_zero_embeddings) {
            auto tensor = hidden_states;
            float original_mean = ggml_tensor_mean(tensor);
            // T5 output is [N, n_token, model_dim], so ne[0] is model_dim, ne[1] is n_token
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) { // Iterate over tokens
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) { // Iterate over hidden_size
                    float value = ggml_tensor_get_f32(tensor, i0, i1, 0); // Assuming 2D tensor
                    value *= weights[i1]; // Apply weight for this token
                    ggml_tensor_set_f32(tensor, value, i0, i1, 0);
                }
            }
            float new_mean = ggml_tensor_mean(tensor);
            ggml_tensor_scale(tensor, (original_mean / new_mean));
        } else {
            float* vec = (float*)hidden_states->data;
            for (int i = 0; i < ggml_nelements(hidden_states); i++) {
                vec[i] = 0;
            }
        }

        // Generate T5 padding mask (c_concat)
        struct ggml_tensor* c_concat_tensor = NULL;
        std::vector<float> padding_mask_vec(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            padding_mask_vec[i] = (tokens[i] == 0) ? 0.0f : 1.0f;
        }
        c_concat_tensor = vector_to_ggml_tensor(work_ctx, padding_mask_vec);
        c_concat_tensor = ggml_reshape_2d(work_ctx, c_concat_tensor, 1, tokens.size()); // Reshape to [1, N_tokens]

        return SDCondition(hidden_states, NULL, c_concat_tensor);
    }

    void alloc_params_buffer()  {
        t5->alloc_params_buffer();
    }

    void free_params_buffer()  {
        t5->free_params_buffer();
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
        t5->get_param_tensors(tensors, "text_encoders.t5xxl.transformer");
    }

    size_t get_params_buffer_size() {
        return t5->get_params_buffer_size();
    }

    std::tuple<SDCondition, std::vector<bool>> get_learned_condition_with_trigger(ggml_context* work_ctx,
                                                                                  int n_threads,
                                                                                  const std::string& text,
                                                                                  int clip_skip,
                                                                                  int width,
                                                                                  int height,
                                                                                  int num_input_imgs,
                                                                                  int adm_in_channels        = -1,
                                                                                  bool force_zero_embeddings = false) override {
        GGML_ASSERT(0 && "Not implemented yet!");
        return std::make_tuple(SDCondition(), std::vector<bool>());
    }

    std::string remove_trigger_from_prompt(ggml_context* work_ctx,
                                           const std::string& prompt) override {
        GGML_ASSERT(0 && "Not implemented yet!");
        return "";
    }
};

-> #ifndef __DIFFUSION_MODEL_H__
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
        return chroma.compute(n_threads, x, timesteps, context, c_concat, y, guidance, output, output_ctx, skip_layers);
    }
};

#endif
