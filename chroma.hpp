#ifndef __CHROMA_HPP__
#define __CHROMA_HPP__

#define CHROMA_GRAPH_SIZE 10240
#define T5_GRAPH_SIZE 10240  // Define if not present in t5.hpp or elsewhere

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
        int64_t d_head                  = q->ne[0];
        int64_t L_q                     = q->ne[1];
        int64_t N                       = q->ne[2] / num_heads;
        float scale                     = 1.0f / std::sqrt((float)d_head);
        struct ggml_tensor* attn_output = ggml_flash_attn_ext(ctx, q, k, v, attn_mask, scale, 0, 0);
        attn_output                     = ggml_reshape_4d(ctx, attn_output, d_head, L_q, num_heads, N);
        attn_output                     = ggml_cont(ctx, ggml_permute(ctx, attn_output, 0, 2, 1, 3));
        attn_output                     = ggml_reshape_3d(ctx, attn_output, d_head * num_heads, L_q, N);
        return attn_output;
    }

    // Corrected modulate function
    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,      // Expected [N, L, C_model_dim]
                                                   struct ggml_tensor* shift,  // Expected [N, C_model_dim]
                                                   struct ggml_tensor* scale)  // Expected [N, C_model_dim]
    {
        GGML_ASSERT(ggml_n_dims(x) == 3);                                   // N, L, C
        GGML_ASSERT(ggml_n_dims(shift) == 2);                               // N, C
        GGML_ASSERT(ggml_n_dims(scale) == 2);                               // N, C
        GGML_ASSERT(x->ne[2] == shift->ne[1] && x->ne[0] == shift->ne[0]);  // x(N,L,C) shift(N,C) -> C must match
        GGML_ASSERT(x->ne[2] == scale->ne[1] && x->ne[0] == scale->ne[0]);

        // Reshape shift and scale to be broadcastable with x: [N, 1, C_model_dim]
        // Assumes x is [N, L, C], shift/scale are [N, C]
        struct ggml_tensor* shift_b = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);  // [N, 1, C_model_dim]
        struct ggml_tensor* scale_b = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C_model_dim]

        struct ggml_tensor* ones = ggml_new_f32(ctx, 1.0f);  // Scalar 1.0f

        // (1 + scale_b)
        struct ggml_tensor* one_plus_scale = ggml_add(ctx, scale_b, ones);  // scale_b is [N,1,C], ones is scalar -> one_plus_scale is [N,1,C]

        // x * (1 + scale_b)
        struct ggml_tensor* term1 = ggml_mul(ctx, x, one_plus_scale);  // x is [N,L,C], one_plus_scale is [N,1,C] -> term1 is [N,L,C]

        // x * (1 + scale_b) + shift_b
        struct ggml_tensor* output = ggml_add(ctx, term1, shift_b);  // term1 is [N,L,C], shift_b is [N,1,C] -> output is [N,L,C]
        return output;
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

    struct Approximator_ggml : public UnaryBlock {
        int in_dim, out_dim, hidden_dim, n_layers;
        Approximator_ggml(int id, int od, int hd, int nl = 5) : in_dim(id), out_dim(od), hidden_dim(hd), n_layers(nl) {
            blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
            for (int i = 0; i < n_layers; ++i) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(hidden_dim, hidden_dim));
                blocks["norms." + std::to_string(i)]  = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_dim));
            }
            blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true));
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* t) override {
            auto ip                = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
            auto op                = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);
            struct ggml_tensor* ci = ip->forward(ctx, t);
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
        int32_t approximator_input_concat_dim    = (16 + 16 + 32);  // Timestep + Guidance + Index Embeddings
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
        ggml_tensor* s    = ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)offset_idx * col_stride);
        ggml_tensor* sc   = ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)(offset_idx + 1) * col_stride);
        ggml_tensor* g    = ggml_view_2d(ctx, mod_vectors_permuted, feature_dim, N_batch_size, col_stride, (size_t)(offset_idx + 2) * col_stride);
        s                 = ggml_cont(ctx, ggml_permute(ctx, s, 1, 0, 2, 3));
        sc                = ggml_cont(ctx, ggml_permute(ctx, sc, 1, 0, 2, 3));
        g                 = ggml_cont(ctx, ggml_permute(ctx, g, 1, 0, 2, 3));
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
            struct ggml_tensor* co   = ggml_concat(ctx, ao, mao, 2);
            struct ggml_tensor* l2o  = l2->forward(ctx, co);
            struct ggml_tensor* gadd = ggml_mul(ctx, l2o, vm.gate);
            struct ggml_tensor* o    = x;
            if (ggml_are_same_shape(x, gadd)) {
                o = ggml_add(ctx, x, gadd);
            } else {
                fprintf(stderr, "SSB skip shape mismatch x:[%lld,%lld,%lld] vs gadd:[%lld,%lld,%lld]\n", (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)gadd->ne[0], (long long)gadd->ne[1], (long long)gadd->ne[2]);
            }
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

            struct ggml_tensor* imna = in1->forward(ctx, img);
            imna                     = Chroma::modulate(ctx, imna, im1.shift, im1.scale);
            auto iqkv                = ia->pre_attention(ctx, imna);
            struct ggml_tensor* tmna = tn1->forward(ctx, txt);
            tmna                     = Chroma::modulate(ctx, tmna, tm1.shift, tm1.scale);
            auto tqkv                = ta->pre_attention(ctx, tmna);
            auto qc                  = ggml_concat(ctx, tqkv[0], iqkv[0], 1);
            auto kc                  = ggml_concat(ctx, tqkv[1], iqkv[1], 1);
            auto vc                  = ggml_concat(ctx, tqkv[2], iqkv[2], 1);
            auto joa                 = Chroma::attention(ctx, qc, kc, vc, pe, num_heads, am);
            int64_t Lt = txt->ne[1], Li = img->ne[1];
            auto pjoa = ggml_cont(ctx, ggml_permute(ctx, joa, 2, 1, 0, 3));
            auto tap  = ggml_view_3d(ctx, pjoa, hidden_size, Lt, joa->ne[2], pjoa->nb[1], pjoa->nb[2], 0);
            tap       = ggml_cont(ctx, ggml_permute(ctx, tap, 2, 1, 0, 3));
            auto iap  = ggml_view_3d(ctx, pjoa, hidden_size, Li, joa->ne[2], pjoa->nb[1], pjoa->nb[2], pjoa->nb[0] * Lt);
            iap       = ggml_cont(ctx, ggml_permute(ctx, iap, 2, 1, 0, 3));
            auto ira  = ia->post_attention(ctx, iap);
            img       = ggml_add(ctx, img, ggml_mul(ctx, ira, im1.gate));
            auto imn2 = in2->forward(ctx, img);
            imn2      = Chroma::modulate(ctx, imn2, im2_mod.shift, im2_mod.scale);  // Use im2_mod
            auto imh  = ggml_gelu_inplace(ctx, im0_linear->forward(ctx, imn2));     // Use im0_linear
            auto imf  = im2_linear->forward(ctx, imh);                              // Use im2_linear
            img       = ggml_add(ctx, img, ggml_mul(ctx, imf, im2_mod.gate));       // Use im2_mod
            auto tra  = ta->post_attention(ctx, tap);
            txt       = ggml_add(ctx, txt, ggml_mul(ctx, tra, tm1.gate));
            auto tmn2 = tn2->forward(ctx, txt);
            tmn2      = Chroma::modulate(ctx, tmn2, tm2_mod.shift, tm2_mod.scale);  // Use tm2_mod
            auto tmh  = ggml_gelu_inplace(ctx, tm0_linear->forward(ctx, tmn2));     // Use tm0_linear
            auto tmf  = tm2_linear->forward(ctx, tmh);                              // Use tm2_linear
            txt       = ggml_add(ctx, txt, ggml_mul(ctx, tmf, tm2_mod.gate));       // Use tm2_mod
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
            auto nf = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto ln = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto xn = nf->forward(ctx, x);
            // Ensure shift/scale are [N, C] for modulate function to reshape to [N,1,C]
            // Python: shift = shift.squeeze(1) # [N,1,C] -> [N,C]
            // Python: scale = scale.squeeze(1)
            // The get_modulations_for_block for "final" already returns them as [N, C]
            auto xm = Chroma::modulate(ctx, xn, shift, scale);
            return ln->forward(ctx, xm);
        }
    };

    struct ChromaUNet_ggml : public GGMLBlock {
        ChromaParams params_unet;
        ChromaUNet_ggml(const ChromaParams& p) : params_unet(p) {
            blocks["distilled_guidance_layer"] = std::make_shared<Approximator_ggml>(p.approximator_input_concat_dim, p.approximator_feature_dim, p.approximator_internal_hidden_dim);
            blocks["img_in"]                   = std::make_shared<Linear>(p.in_channels, p.unet_model_dim, true);
            blocks["txt_in"]                   = std::make_shared<Linear>(p.t5_embed_dim, p.unet_model_dim, true);
            for (int i = 0; i < p.depth; ++i)
                blocks["double_blocks." + std::to_string(i)] = std::make_shared<DoubleStreamBlock_ggml>(p.unet_model_dim, p.num_heads, p.mlp_ratio, true);
            for (int i = 0; i < p.depth_single_blocks; ++i)
                blocks["single_blocks." + std::to_string(i)] = std::make_shared<SingleStreamBlock_ggml>(p.unet_model_dim, p.num_heads, p.mlp_ratio);
            blocks["final_layer"] = std::make_shared<LastLayer_ggml>(p.unet_model_dim, p.out_channels);
        }
        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* ilt, struct ggml_tensor* ts_approx_in, struct ggml_tensor* te, struct ggml_tensor* pe, struct ggml_tensor* t5pm, std::vector<int> sl = {}) {
            auto ap                 = std::dynamic_pointer_cast<Approximator_ggml>(blocks["distilled_guidance_layer"]);
            auto img_ip             = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_ip             = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto final_l            = std::dynamic_pointer_cast<LastLayer_ggml>(blocks["final_layer"]);
            struct ggml_tensor* mvg = ap->forward(ctx, ts_approx_in);
            struct ggml_tensor* cit = img_ip->forward(ctx, ilt);
            struct ggml_tensor* ctt = txt_ip->forward(ctx, te);
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
            struct ggml_tensor* combt = ggml_concat(ctx, ctt, cit, 1);
            for (int i = 0; i < params_unet.depth_single_blocks; ++i) {
                if (!sl.empty() && std::find(sl.begin(), sl.end(), i + params_unet.depth) != sl.end())
                    continue;
                auto b                    = std::dynamic_pointer_cast<SingleStreamBlock_ggml>(blocks["single_blocks." + std::to_string(i)]);
                BlockModulationOutput smo = get_modulations_for_block(ctx, params_unet, mvg, "single", i);
                auto& sm                  = smo.single_mod;
                combt                     = b->forward(ctx, combt, pe, sm, t5pm);
            }
            int64_t ntt               = ctt->ne[1];
            struct ggml_tensor* fit   = ggml_view_3d(ctx, combt, combt->ne[0], combt->ne[1] - ntt, combt->ne[2], combt->nb[1], combt->nb[2], ntt * combt->nb[1]);
            BlockModulationOutput fmo = get_modulations_for_block(ctx, params_unet, mvg, "final", 0);
            auto& fmp                 = fmo.final_mods;
            return final_l->forward(ctx, fit, fmp.first, fmp.second);
        }
    };

    struct ChromaRunner : public GGMLRunner {
        ChromaParams chroma_hyperparams;
        ChromaUNet_ggml chroma_unet;
        ChromaRunner(ggml_backend_t b, std::map<std::string, enum ggml_type>& tt, const std::string pr = "", bool ufa = false)
            : GGMLRunner(b), chroma_hyperparams(), chroma_unet(chroma_hyperparams) {
            chroma_hyperparams.flash_attn = ufa;
            chroma_unet.init(params_ctx, tt, pr);
        }
        std::string get_desc() override { return "chroma"; }
        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& t, const std::string p) { chroma_unet.get_param_tensors(t, p); }
        struct ggml_cgraph* build_graph(struct ggml_tensor* ilt, struct ggml_tensor* ts_approx_in, struct ggml_tensor* txt, struct ggml_tensor* pe, struct ggml_tensor* t5pm, std::vector<int> sl = {}) {
            struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, CHROMA_GRAPH_SIZE, false);
            ilt                    = to_backend(ilt);
            ts_approx_in           = to_backend(ts_approx_in);
            txt                    = to_backend(txt);
            pe                     = to_backend(pe);
            if (t5pm)
                t5pm = to_backend(t5pm);
            struct ggml_tensor* o = chroma_unet.forward(compute_ctx, ilt, ts_approx_in, txt, pe, t5pm, sl);
            ggml_build_forward_expand(gf, o);
            return gf;
        }
        void compute(int nt, struct ggml_tensor* ilt, struct ggml_tensor* ts_approx_in, struct ggml_tensor* txt, struct ggml_tensor* pe, struct ggml_tensor* t5pm, struct ggml_tensor** o = 0, struct ggml_context* oc = 0, std::vector<int> sl = {}) {
            auto get_gf = [&]() -> struct ggml_cgraph* { return build_graph(ilt, ts_approx_in, txt, pe, t5pm, sl); };
            GGMLRunner::compute(get_gf, nt, false, o, oc);
        }
    };

}  // namespace Chroma
#endif  // __CHROMA_HPP__