#ifndef __CHROMA_HPP__
#define __CHROMA_HPP__

#define CHROMA_GRAPH_SIZE 10240

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>     // For std::sqrt, std::exp, std::log, std::min, std::max
#include <algorithm> // For std::min, std::max, std::find
#include <utility>   // For std::pair

#include "ggml_extend.hpp" // Assuming this contains Linear, LayerNorm, UnaryBlock, GGMLBlock etc.
#include "model.h"         // For base classes like UnaryBlock, GGMLBlock

namespace Chroma {

struct ChromaParams {
    // From your GGUF and Python code
    int32_t in_channels = 64;
    int32_t out_channels = 64;
    int32_t t5_embed_dim = 4096; // context_in_dim in Python

    // Approximator specific (distilled_guidance_layer)
    // in_dim for Approximator is determined by concatenation of timestep_guidance and modulation_index embeddings.
    // E.g., (16*2 + 16*2 for ts/guidance + 32 for index) = 64 + 32 = 96. (Python: input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1))
    // Let's call this `approximator_input_concat_dim`. This must be correctly calculated.
    // For now, let's assume a value, needs to be exact from Python logic.
    int32_t approximator_input_concat_dim = 64; // Based on user clarification (16+16+32)
    int32_t approximator_internal_hidden_dim = 5120; // hidden_dim for Approximator
    int32_t approximator_feature_dim = 3072;       // out_dim for Approximator, this IS hidden_size of UNet

    int32_t unet_model_dim = 3072; // hidden_size in Python UNet blocks
    // ... other params like num_heads, mlp_ratio, depth etc. ...
    int32_t num_heads = 24;
    float mlp_ratio = 4.0f;
    int32_t depth = 19;
    int32_t depth_single_blocks = 38;

    // For get_modulations logic
    int32_t mod_vector_total_indices = 344; // from python mod_index_length

    int32_t head_dim; // Will be unet_model_dim / num_heads
    bool flash_attn = false;


    ChromaParams() { // Default constructor to calculate derived values
        if (num_heads > 0 && unet_model_dim > 0 && unet_model_dim % num_heads == 0) {
            head_dim = unet_model_dim / num_heads;
        } else {
            head_dim = 128; // A sensible default if not calculable, but should be derived
        }
    }
};

__STATIC_INLINE__ struct ggml_tensor* attention(struct ggml_context* ctx,
                                                struct ggml_tensor* q, // Expected: [d_head, L_q, N*n_head]
                                                struct ggml_tensor* k, // Expected: [d_head, L_k, N*n_head]
                                                struct ggml_tensor* v, // Expected: [d_head, L_v, N*n_head] (L_v == L_k)
                                                struct ggml_tensor* pe, // Positional embeddings, may not be directly used by ggml_flash_attn_ext
                                                int64_t num_heads,      // Added num_heads to derive original N
                                                struct ggml_tensor* attn_mask = NULL) {
    // q, k, v are typically [d_head, L, N*n_head] for flash_attn
    // pe: [L, d_head/2, 2, 2] (RoPE applied before this call if needed by flash_attn_ext version)
    // return: [N, L_q, n_head*d_head]

    int64_t d_head = q->ne[0];
    int64_t L_q = q->ne[1];
    // N_x_n_head = q->ne[2]
    int64_t N = q->ne[2] / num_heads; // Derive N

    float scale = 1.0f / std::sqrt((float)d_head);

    // ggml_flash_attn_ext expects q,k,v as [N_merged, L, D_head_actual] where N_merged can be B*H
    // The output of ggml_flash_attn_ext is [N_merged, L_q, D_head_actual]
    struct ggml_tensor* attn_output = ggml_flash_attn_ext(ctx, q, k, v, attn_mask, scale, 0, 0); // Assuming RoPE handled if flash_attn_ext needs it

    // Reshape the output from [N*n_head, L_q, d_head] to [N, L_q, n_head*d_head]
    attn_output = ggml_reshape_4d(ctx, attn_output, d_head, L_q, num_heads, N); // [N, num_heads, L_q, d_head] (interpret N*num_heads as N, num_heads)
    attn_output = ggml_cont(ctx, ggml_permute(ctx, attn_output, 0, 2, 1, 3));   // [N, L_q, num_heads, d_head]
    attn_output = ggml_reshape_3d(ctx, attn_output, d_head * num_heads, L_q, N); // [N, L_q, num_heads*d_head]

    return attn_output;
}

__STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                               struct ggml_tensor* x,
                                               struct ggml_tensor* shift,
                                               struct ggml_tensor* scale) {
    // x: [N, L, C] (target tensor)
    // scale: [N, C] (modulation params)
    // shift: [N, C] (modulation params)

    // Reshape scale and shift to [N, 1, C] for broadcasting with x [N, L, C]
    struct ggml_tensor* scale_reshaped = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
    struct ggml_tensor* shift_reshaped = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);

    // Explicitly repeat for broadcasting (more robust with some ggml backends)
    struct ggml_tensor* scale_repeated = ggml_repeat(ctx, scale_reshaped, x);
    struct ggml_tensor* shift_repeated = ggml_repeat(ctx, shift_reshaped, x);
    
    // Create a tensor of ones with the same shape as x for (1 + scale)
    // Or, more directly, (1 + scale_repeated) * x + shift_repeated
    // struct ggml_tensor* ones_like_x = ggml_dup(ctx, x); // Create a tensor with same metadata
    // ggml_set_f32(ones_like_x, 1.0f); // this is wrong, ones should be like scale_repeated
    
    struct ggml_tensor* ones_tensor = ggml_new_tensor(ctx, scale_repeated->type, ggml_n_dims(scale_repeated), scale_repeated->ne);
    ggml_set_f32(ones_tensor, 1.0f);


    struct ggml_tensor* one_plus_scale = ggml_add(ctx, ones_tensor, scale_repeated);
    struct ggml_tensor* modulated_x = ggml_mul(ctx, x, one_plus_scale); // Element-wise: x * (1 + scale)
    struct ggml_tensor* output = ggml_add(ctx, modulated_x, shift_repeated);    // Element-wise: x * (1 + scale) + shift

    return output;
}

struct RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") override {
        ggml_type wtype = GGML_TYPE_F32;
        if (tensor_types.count(prefix + "scale")) {
            wtype = tensor_types[prefix + "scale"];
        }
        params["scale"] = ggml_new_tensor_1d(ctx, wtype, hidden_size); // Changed "weight" to "scale"
    }

public:
    RMSNorm(int64_t hidden_size, float eps = 1e-06f)
        : hidden_size(hidden_size), eps(eps) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) override {
        struct ggml_tensor* s = params["scale"]; // Changed "w" to "s" and "weight" to "scale"
        x = ggml_rms_norm(ctx, x, eps);
        x = ggml_mul(ctx, x, s); // Apply scaling
        return x;
    }
};

struct MLPEmbedder : public UnaryBlock {
public:
    MLPEmbedder(int64_t in_dim, int64_t hidden_dim) {
        blocks["in_layer"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, true));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) override {
        auto in_layer = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
        auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

        x = in_layer->forward(ctx, x);
        x = ggml_silu_inplace(ctx, x);
        x = out_layer->forward(ctx, x);
        return x;
    }
};

struct Approximator_ggml : public UnaryBlock {
    int in_dim;     // e.g., params.approximator_input_concat_dim
    int out_dim;    // e.g., params.approximator_feature_dim (mod_vectors_global dim)
    int hidden_dim; // e.g., params.approximator_internal_hidden_dim
    int n_layers;

    Approximator_ggml(int in_dim, int out_dim, int hidden_dim, int n_layers = 5)
        : in_dim(in_dim), out_dim(out_dim), hidden_dim(hidden_dim), n_layers(n_layers) {
        blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        for (int i = 0; i < n_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(hidden_dim, hidden_dim));
            blocks["norms." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_dim));
        }
        blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, out_dim, true)); // Bias true for out_proj
    }

    // timestep_for_approximator_input_vec: [N, L_indices_for_approximator, C_concat_for_approximator_input]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* timestep_for_approximator_input_vec) override {
        auto in_proj = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

        struct ggml_tensor* current_input = in_proj->forward(ctx, timestep_for_approximator_input_vec);

        for (int i = 0; i < n_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);
            
            struct ggml_tensor* norm_output = norm->forward(ctx, current_input);
            struct ggml_tensor* mlp_output = layer->forward(ctx, norm_output);
            current_input = ggml_add(ctx, current_input, mlp_output);
        }
        struct ggml_tensor* out_proj_output = out_proj->forward(ctx, current_input);
        return out_proj_output; // This is mod_vectors_global
    }
};

struct QKNorm : public GGMLBlock {
    int64_t head_dim; // QKNorm operates on head_dim
    QKNorm(int64_t head_dim) : head_dim(head_dim) {
        blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim));
        blocks["key_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim));
    }

    struct ggml_tensor* query_norm(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);
        return norm->forward(ctx, x);
    }

    struct ggml_tensor* key_norm(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
        return norm->forward(ctx, x);
    }
};

struct SelfAttention : public GGMLBlock {
    int64_t dim; // model_dim (e.g., 3072)
    int64_t num_heads;
    int64_t head_dim; // dim / num_heads
    bool qkv_bias;

    SelfAttention(int64_t dim, int64_t num_heads, bool qkv_bias = false)
        : dim(dim), num_heads(num_heads), qkv_bias(qkv_bias) {
        GGML_ASSERT(dim % num_heads == 0);
        this->head_dim = dim / num_heads;
        blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim)); // QKNorm uses head_dim
        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim, true)); // Proj bias is true
    }

    // Returns: q, k, v (each [d_head, L, N*n_head])
    std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, L, dim]
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
        auto norm_block = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

        auto qkv_combined = qkv_proj->forward(ctx, x); // [N, L, dim*3]

        int64_t N = x->ne[2];
        int64_t L = x->ne[1];

        // Reshape and split
        // qkv_combined: [N, L, dim*3] -> [dim*3, L, N]
        qkv_combined = ggml_cont(ctx, ggml_permute(ctx, qkv_combined, 2, 1, 0, 3)); 
        // -> [dim, 3, L, N]
        qkv_combined = ggml_reshape_4d(ctx, qkv_combined, dim, 3, L, N);
        
        int64_t s2 = qkv_combined->nb[2]; // stride for L
        int64_t s1 = qkv_combined->nb[1]; // stride for 3

        struct ggml_tensor* q = ggml_view_3d(ctx, qkv_combined, dim, L, N, s2, qkv_combined->nb[3], 0*s1);
        struct ggml_tensor* k = ggml_view_3d(ctx, qkv_combined, dim, L, N, s2, qkv_combined->nb[3], 1*s1);
        struct ggml_tensor* v = ggml_view_3d(ctx, qkv_combined, dim, L, N, s2, qkv_combined->nb[3], 2*s1);
        
        // Reshape q, k, v for QKNorm and attention: [d_head, L, N*n_head]
        q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)), head_dim, L, N * num_heads); // [N, L, dim] -> [d_head, L, N*n_head]
        k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)), head_dim, L, N * num_heads);
        v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)), head_dim, L, N * num_heads);


        struct ggml_tensor* q_normed = norm_block->query_norm(ctx, q);
        struct ggml_tensor* k_normed = norm_block->key_norm(ctx, k);
        
        return {q_normed, k_normed, v};
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, L, dim] (after attention and reshape)
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
        return proj->forward(ctx, x);
    }
};


struct ModulationOut { // This struct is fine
    ggml_tensor* shift = NULL;
    ggml_tensor* scale = NULL;
    ggml_tensor* gate  = NULL;
    ModulationOut(ggml_tensor*s=0, ggml_tensor*sc=0, ggml_tensor*g=0):shift(s),scale(sc),gate(g){}
};

enum class ModulationOutputType {
    SINGLE,
    DOUBLE,
    FINAL
};

struct BlockModulationOutput {
    ModulationOutputType type;
    ModulationOut single_mod;
    std::pair<ModulationOut, ModulationOut> double_mods;
    std::pair<ggml_tensor*, ggml_tensor*> final_mods;
};


// C++ equivalent of ChromaModulationOut.from_offset and part of get_modulations
// mod_vectors is [N, mod_vector_total_indices, approximator_feature_dim]
// approximator_feature_dim should be == unet_model_dim (3072)
__STATIC_INLINE__ ModulationOut get_modulation_params_from_offset(
    ggml_context* ctx,
    ggml_tensor* mod_vectors, // Shape: [approximator_feature_dim, mod_vector_total_indices, N] (after permute)
    int offset_idx // Index along the mod_vector_total_indices dimension
) {
    // mod_vectors is [N, total_indices, feature_dim]
    // We want to slice along total_indices dimension
    // After permute to [feature_dim, total_indices, N] for easier viewing with ggml_view_2d

    int64_t feature_dim = mod_vectors->ne[0]; // Should be unet_model_dim
    int64_t total_indices = mod_vectors->ne[1];
    int64_t N = mod_vectors->ne[2]; // Batch size

    // Stride for the 'total_indices' dimension
    size_t stride_indices = mod_vectors->nb[1];

    // Shift: mod_vectors[:, offset_idx, :]
    ggml_tensor* shift = ggml_view_2d(ctx, mod_vectors,
                                     feature_dim, N,
                                     stride_indices, // Stride to jump one full feature_dim * N block
                                     offset_idx * mod_vectors->nb[1]); // Offset to the start of the desired "row"

    // Scale: mod_vectors[:, offset_idx + 1, :]
    ggml_tensor* scale = ggml_view_2d(ctx, mod_vectors,
                                     feature_dim, N,
                                     stride_indices,
                                     (offset_idx + 1) * mod_vectors->nb[1]);

    // Gate: mod_vectors[:, offset_idx + 2, :]
    ggml_tensor* gate  = ggml_view_2d(ctx, mod_vectors,
                                     feature_dim, N,
                                     stride_indices,
                                     (offset_idx + 2) * mod_vectors->nb[1]);
    // These views are [feature_dim, N]. Need to permute to [N, feature_dim] for modulate function
    shift = ggml_cont(ctx, ggml_permute(ctx, shift, 1,0,2,3));
    scale = ggml_cont(ctx, ggml_permute(ctx, scale, 1,0,2,3));
    gate  = ggml_cont(ctx, ggml_permute(ctx, gate,  1,0,2,3));

    return ModulationOut(shift, scale, gate);
}


// Wrapper similar to Python's get_modulations
// mod_vectors_input is [N, mod_vector_total_indices, approximator_feature_dim]
__STATIC_INLINE__ BlockModulationOutput get_modulations_for_block(
    ggml_context* ctx,
    const ChromaParams& params,      // To get depth_single_blocks, depth_double_blocks
    ggml_tensor* mod_vectors_input,  // Output of Approximator
    const std::string& block_type,
    int block_idx
) {
    BlockModulationOutput output;

    // Permute mod_vectors_input from [N, total_indices, feature_dim] to [feature_dim, total_indices, N]
    // for easier slicing with ggml_view_2d.
    ggml_tensor* mod_vectors = ggml_cont(ctx, ggml_permute(ctx, mod_vectors_input, 2, 1, 0, 3));


    if (block_type == "final") {
        output.type = ModulationOutputType::FINAL;
        // Python: return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        // Assuming -1 is total_indices-1, -2 is total_indices-2
        int64_t last_idx = params.mod_vector_total_indices - 1;
        int64_t second_last_idx = params.mod_vector_total_indices - 2;
        
        ggml_tensor* scale_final = ggml_view_2d(ctx, mod_vectors, params.approximator_feature_dim, mod_vectors->ne[2], mod_vectors->nb[1], second_last_idx * mod_vectors->nb[1]);
        ggml_tensor* shift_final = ggml_view_2d(ctx, mod_vectors, params.approximator_feature_dim, mod_vectors->ne[2], mod_vectors->nb[1], last_idx * mod_vectors->nb[1]);
        
        scale_final = ggml_cont(ctx, ggml_permute(ctx, scale_final, 1,0,2,3)); // [N, feature_dim]
        shift_final = ggml_cont(ctx, ggml_permute(ctx, shift_final, 1,0,2,3)); // [N, feature_dim]

        output.final_mods = std::make_pair(shift_final, scale_final); // Order might be shift, scale based on LastLayer.forward
        return output;
    }

    int single_block_count = params.depth_single_blocks;
    int double_block_count = params.depth; // Python uses self.params.depth
    int offset_s_s_g = 0; // This is the 'idx' argument for from_offset in python

    if (block_type == "single") {
        output.type = ModulationOutputType::SINGLE;
        offset_s_s_g = 3 * block_idx;
        output.single_mod = get_modulation_params_from_offset(ctx, mod_vectors, offset_s_s_g);
        return output;
    }

    offset_s_s_g = 3 * block_idx * 2; // Python: offset *= 2 (which means 3*idx*2)

    if (block_type == "double_img" || block_type == "double_txt") {
        output.type = ModulationOutputType::DOUBLE;
        offset_s_s_g += 3 * single_block_count; // Advance past single block modulations
        if (block_type == "double_txt") {
            offset_s_s_g += 6 * double_block_count; // Advance past double_img block modulations (6 params per double block: mod1(s,s,g) + mod2(s,s,g))
        }
        ModulationOut mod1 = get_modulation_params_from_offset(ctx, mod_vectors, offset_s_s_g);
        ModulationOut mod2 = get_modulation_params_from_offset(ctx, mod_vectors, offset_s_s_g + 3);
        output.double_mods = std::make_pair(mod1, mod2);
        return output;
    }

    throw std::runtime_error("Bad block_type in get_modulations_for_block: " + block_type);
}


// SingleStreamBlock uses the new modulation
struct SingleStreamBlock_ggml : public GGMLBlock {
    int64_t hidden_size, num_heads, head_dim; float mlp_ratio;
    // No internal Modulation block needed anymore
    SingleStreamBlock_ggml(int64_t hs, int64_t nh, float mr)
        : hidden_size(hs), num_heads(nh), mlp_ratio(mr) {
        GGML_ASSERT(hs % nh == 0); head_dim = hs / nh;
        int64_t mlp_hd = static_cast<int64_t>(hs * mr);
        blocks["pre_norm"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["linear1"]=std::shared_ptr<GGMLBlock>(new Linear(hs,hs*3+mlp_hd,true));
        blocks["norm"]=std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
        blocks["linear2"]=std::shared_ptr<GGMLBlock>(new Linear(hs+mlp_hd,hs,true));
    }

    // vec_mod_params IS the ModulationOut struct with s,s,g for this block
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pe, const ModulationOut& vec_mod_params, struct ggml_tensor* attn_mask = NULL) {
        auto pn=std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);
        auto l1=std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
        auto qn=std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
        auto l2=std::dynamic_pointer_cast<Linear>(blocks["linear2"]);

        struct ggml_tensor* xn=pn->forward(ctx,x);
        // Ensure shift/scale are [N, C] or broadcastable from [N,1,C] to xn [N,L,C]
        // Python: mod.scale is [N,1,C], shift is [N,1,C] after ChromaModulationOut.from_offset
        // The get_modulation_params_from_offset already returns them as [N, C].
        // The modulate function will reshape them to [N,1,C] if x is [N,L,C]
        struct ggml_tensor* xm = Chroma::modulate(ctx, xn, vec_mod_params.shift, vec_mod_params.scale);
        
        struct ggml_tensor* l1o=l1->forward(ctx,xm);
        int64_t N=x->ne[2], L=x->ne[1], mlp_hda=static_cast<int64_t>(hidden_size*mlp_ratio);
        struct ggml_tensor* pl1o=ggml_cont(ctx,ggml_permute(ctx,l1o,2,1,0,3));
        struct ggml_tensor* qkvp=ggml_view_3d(ctx,pl1o,hidden_size*3,L,N,pl1o->nb[1],pl1o->nb[2],0);
        qkvp=ggml_cont(ctx,ggml_permute(ctx,qkvp,2,1,0,3));
        struct ggml_tensor* mlpp=ggml_view_3d(ctx,pl1o,mlp_hda,L,N,pl1o->nb[1],pl1o->nb[2],pl1o->nb[0]*(hidden_size*3));
        mlpp=ggml_cont(ctx,ggml_permute(ctx,mlpp,2,1,0,3));
        struct ggml_tensor* qkvpr=ggml_reshape_4d(ctx,ggml_cont(ctx,ggml_permute(ctx,qkvp,2,1,0,3)),hidden_size,3,L,N);
        int64_t s2q=qkvpr->nb[2],s1q=qkvpr->nb[1];
        auto q_=ggml_view_3d(ctx,qkvpr,hidden_size,L,N,s2q,qkvpr->nb[3],0*s1q);
        auto k_=ggml_view_3d(ctx,qkvpr,hidden_size,L,N,s2q,qkvpr->nb[3],1*s1q);
        auto v_=ggml_view_3d(ctx,qkvpr,hidden_size,L,N,s2q,qkvpr->nb[3],2*s1q);
        q_=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_permute(ctx,q_,1,2,0,3)),head_dim,L,N*num_heads);
        k_=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_permute(ctx,k_,1,2,0,3)),head_dim,L,N*num_heads);
        v_=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_permute(ctx,v_,1,2,0,3)),head_dim,L,N*num_heads);
        q_=qn->query_norm(ctx,q_); k_=qn->key_norm(ctx,k_);
        struct ggml_tensor* ao=Chroma::attention(ctx,q_,k_,v_,pe,num_heads,attn_mask);
        struct ggml_tensor* mao=ggml_gelu_inplace(ctx,mlpp);
        struct ggml_tensor* co=ggml_concat(ctx,ao,mao,2);
        struct ggml_tensor* l2o=l2->forward(ctx,co);
        struct ggml_tensor* gadd = ggml_mul(ctx,l2o,vec_mod_params.gate);
        struct ggml_tensor* output = x;
        if (ggml_are_same_shape(x, gadd)) { output = ggml_add(ctx, x, gadd); }
        else { fprintf(stderr, "SSB skip shape mismatch\n"); }
        return output;
    }
};


struct DoubleStreamBlock_ggml : public GGMLBlock {
    int64_t hidden_size, num_heads; float mlp_ratio; bool qkv_bias;
    // No internal Modulation blocks
    DoubleStreamBlock_ggml(int64_t hs, int64_t nh, float mr, bool qb=true)
        : hidden_size(hs), num_heads(nh), mlp_ratio(mr), qkv_bias(qb) {
        int64_t mlp_hd = static_cast<int64_t>(hs*mr);
        blocks["img_norm1"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["img_attn"]=std::shared_ptr<GGMLBlock>(new SelfAttention(hs,nh,qb));
        blocks["img_norm2"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["img_mlp.0"]=std::shared_ptr<GGMLBlock>(new Linear(hs,mlp_hd,true));
        blocks["img_mlp.2"]=std::shared_ptr<GGMLBlock>(new Linear(mlp_hd,hs,true));
        blocks["txt_norm1"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["txt_attn"]=std::shared_ptr<GGMLBlock>(new SelfAttention(hs,nh,qb));
        blocks["txt_norm2"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["txt_mlp.0"]=std::shared_ptr<GGMLBlock>(new Linear(hs,mlp_hd,true));
        blocks["txt_mlp.2"]=std::shared_ptr<GGMLBlock>(new Linear(mlp_hd,hs,true));
    }

    // vec_img_mods and vec_txt_mods are std::pair<ModulationOut, ModulationOut>
    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(
        struct ggml_context* ctx, struct ggml_tensor* img, struct ggml_tensor* txt,
        struct ggml_tensor* pe,
        const std::pair<ModulationOut, ModulationOut>& vec_img_mods,
        const std::pair<ModulationOut, ModulationOut>& vec_txt_mods,
        struct ggml_tensor* attn_mask = NULL
    ) {
        ModulationOut img_mod1 = vec_img_mods.first;
        ModulationOut img_mod2 = vec_img_mods.second;
        ModulationOut txt_mod1 = vec_txt_mods.first;
        ModulationOut txt_mod2 = vec_txt_mods.second;

        auto in1=std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
        auto ia=std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);
        auto in2=std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
        auto im0=std::dynamic_pointer_cast<Linear>(blocks["img_mlp.0"]);
        auto im2=std::dynamic_pointer_cast<Linear>(blocks["img_mlp.2"]);
        auto tn1=std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
        auto ta=std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);
        auto tn2=std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
        auto tm0=std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.0"]);
        auto tm2=std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.2"]);

        struct ggml_tensor* imna=in1->forward(ctx,img);
        imna=Chroma::modulate(ctx,imna,img_mod1.shift,img_mod1.scale);
        auto iqkv=ia->pre_attention(ctx,imna);
        struct ggml_tensor* tmna=tn1->forward(ctx,txt);
        tmna=Chroma::modulate(ctx,tmna,txt_mod1.shift,txt_mod1.scale);
        auto tqkv=ta->pre_attention(ctx,tmna);
        auto qc=ggml_concat(ctx,tqkv[0],iqkv[0],1);
        auto kc=ggml_concat(ctx,tqkv[1],iqkv[1],1);
        auto vc=ggml_concat(ctx,tqkv[2],iqkv[2],1);
        auto joa=Chroma::attention(ctx,qc,kc,vc,pe,num_heads,attn_mask);
        int64_t Lt=txt->ne[1], Li=img->ne[1];
        auto pjoa=ggml_cont(ctx,ggml_permute(ctx,joa,2,1,0,3));
        auto tap=ggml_view_3d(ctx,pjoa,hidden_size,Lt,joa->ne[2],pjoa->nb[1],pjoa->nb[2],0);
        tap=ggml_cont(ctx,ggml_permute(ctx,tap,2,1,0,3));
        auto iap=ggml_view_3d(ctx,pjoa,hidden_size,Li,joa->ne[2],pjoa->nb[1],pjoa->nb[2],pjoa->nb[0]*Lt);
        iap=ggml_cont(ctx,ggml_permute(ctx,iap,2,1,0,3));
        auto ira=ia->post_attention(ctx,iap);
        img=ggml_add(ctx,img,ggml_mul(ctx,ira,img_mod1.gate));
        auto imn2=in2->forward(ctx,img);
        imn2=Chroma::modulate(ctx,imn2,img_mod2.shift,img_mod2.scale);
        auto imh=ggml_gelu_inplace(ctx,im0->forward(ctx,imn2));
        auto imf=im2->forward(ctx,imh);
        img=ggml_add(ctx,img,ggml_mul(ctx,imf,img_mod2.gate));
        auto tra=ta->post_attention(ctx,tap);
        txt=ggml_add(ctx,txt,ggml_mul(ctx,tra,txt_mod1.gate));
        auto tmn2=tn2->forward(ctx,txt);
        tmn2=Chroma::modulate(ctx,tmn2,txt_mod2.shift,txt_mod2.scale);
        auto tmh=ggml_gelu_inplace(ctx,tm0->forward(ctx,tmn2));
        auto tmf=tm2->forward(ctx,tmh);
        txt=ggml_add(ctx,txt,ggml_mul(ctx,tmf,txt_mod2.gate));
        return {img,txt};
    }
};

struct LastLayer_ggml : public GGMLBlock {
    int64_t hidden_size, out_channels;
    // No internal Modulation block
    LastLayer_ggml(int64_t hs, int64_t oc) : hidden_size(hs), out_channels(oc) {
        blocks["norm_final"]=std::shared_ptr<GGMLBlock>(new LayerNorm(hs,1e-6f,false));
        blocks["linear"]=std::shared_ptr<GGMLBlock>(new Linear(hs,oc,true));
    }
    // shift and scale are passed directly
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* shift, struct ggml_tensor* scale) {
        auto nf=std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto ln=std::dynamic_pointer_cast<Linear>(blocks["linear"]);
        auto xn=nf->forward(ctx,x);
        // Ensure shift/scale are [N, C] for modulate function to reshape to [N,1,C]
        // Python: shift = shift.squeeze(1) # [N,1,C] -> [N,C]
        // Python: scale = scale.squeeze(1)
        // The get_modulations_for_block for "final" already returns them as [N, C]
        auto xm=Chroma::modulate(ctx,xn,shift,scale);
        return ln->forward(ctx,xm);
    }
};


struct ChromaUNet_ggml : public GGMLBlock {
    ChromaParams params_unet;

    ChromaUNet_ggml(const ChromaParams& params_arg) {
        this->params_unet = params_arg; // Store a copy
        if (this->params_unet.num_heads > 0 && this->params_unet.unet_model_dim > 0 && this->params_unet.unet_model_dim % this->params_unet.num_heads == 0) {
             this->params_unet.head_dim = this->params_unet.unet_model_dim / this->params_unet.num_heads;
        } else {
            // Handle error or set default, though ChromaParams constructor should do this
            this->params_unet.head_dim = 128; 
        }


        blocks["distilled_guidance_layer"] = std::shared_ptr<GGMLBlock>(
            new Approximator_ggml(params_unet.approximator_input_concat_dim, // Calculated input dim
                                  params_unet.approximator_feature_dim,    // Output feature dim per index
                                  params_unet.approximator_internal_hidden_dim));

        // img_in takes raw VAE latent channels (e.g., 4 or 64 if already projected by VAE encoder)
        // The GGUF has img_in.weight: [64 3072], so input is 64 channels, output 3072 model_dim
        blocks["img_in"] = std::shared_ptr<GGMLBlock>(new Linear(params_unet.in_channels, params_unet.unet_model_dim, true));
        blocks["txt_in"] = std::shared_ptr<GGMLBlock>(new Linear(params_unet.t5_embed_dim, params_unet.unet_model_dim, true));

        for (int i = 0; i < params_unet.depth; ++i) {
            blocks["double_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new DoubleStreamBlock_ggml(params_unet.unet_model_dim, params_unet.num_heads, params_unet.mlp_ratio,
                                           true)); // Removed vec_dim
        }
        for (int i = 0; i < params_unet.depth_single_blocks; ++i) {
            blocks["single_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                new SingleStreamBlock_ggml(params_unet.unet_model_dim, params_unet.num_heads, params_unet.mlp_ratio)); // Removed vec_dim
        }
        
        // NO "final_layer_mod" block, as LastLayer gets its s/s from get_modulations_for_block

        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(
            new LastLayer_ggml(params_unet.unet_model_dim, params_unet.out_channels));
    }

    // img_latent_tokens: [N, L_img, C_in_channels] (before img_in projection)
    // timestep_for_approximator_input_vec: [N, L_indices_for_approximator, C_concat_for_approximator_input] (this is the 'input_vec' from Python)
    // txt_embeddings: [N, L_txt, C_t5_embed_dim] (before txt_in projection)
    // TODO: Add pe, t5_padding_mask generation and input preparation
    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* img_latent_tokens,
                                struct ggml_tensor* timestep_for_approximator_input_vec,
                                struct ggml_tensor* txt_embeddings,
                                struct ggml_tensor* pe,
                                struct ggml_tensor* t5_padding_mask,
                                std::vector<int> skip_layers = {}) {
        auto approximator = std::dynamic_pointer_cast<Approximator_ggml>(blocks["distilled_guidance_layer"]);
        auto img_in_proj = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
        auto txt_in_proj = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
        auto final_layer = std::dynamic_pointer_cast<LastLayer_ggml>(blocks["final_layer"]);

        // This is mod_vectors from Python: [N, mod_vector_total_indices, approximator_feature_dim]
        struct ggml_tensor* mod_vectors_global = approximator->forward(ctx, timestep_for_approximator_input_vec);

        struct ggml_tensor* current_img_tokens = img_in_proj->forward(ctx, img_latent_tokens);
        struct ggml_tensor* current_txt_tokens = txt_in_proj->forward(ctx, txt_embeddings);

        for (int i = 0; i < params_unet.depth; ++i) {
            // ... skip_layers check ...
            auto block = std::dynamic_pointer_cast<DoubleStreamBlock_ggml>(blocks["double_blocks."+std::to_string(i)]);
            BlockModulationOutput img_mods_output = get_modulations_for_block(ctx, params_unet, mod_vectors_global, "double_img", i);
            BlockModulationOutput txt_mods_output = get_modulations_for_block(ctx, params_unet, mod_vectors_global, "double_txt", i);
            
            // Assuming the type is DOUBLE for these blocks
            auto& img_mods_pair = img_mods_output.double_mods;
            auto& txt_mods_pair = txt_mods_output.double_mods;

            auto pair_tokens = block->forward(ctx, current_img_tokens, current_txt_tokens, pe, img_mods_pair, txt_mods_pair, t5_padding_mask);
            current_img_tokens = pair_tokens.first;
            current_txt_tokens = pair_tokens.second;
        }

        struct ggml_tensor* combined_tokens = ggml_concat(ctx, current_txt_tokens, current_img_tokens, 1);

        for (int i = 0; i < params_unet.depth_single_blocks; ++i) {
            // ... skip_layers check ...
            auto block = std::dynamic_pointer_cast<SingleStreamBlock_ggml>(blocks["single_blocks."+std::to_string(i)]);
            BlockModulationOutput single_mod_output = get_modulations_for_block(ctx, params_unet, mod_vectors_global, "single", i);
            
            // Assuming the type is SINGLE for these blocks
            auto& single_mod = single_mod_output.single_mod;
            combined_tokens = block->forward(ctx, combined_tokens, pe, single_mod, t5_padding_mask);
        }
        
        // Extract relevant part for final layer (output of SingleStreamBlocks corresponding to image tokens)
        // This assumes txt tokens are first in combined_tokens
        int64_t num_txt_tokens = current_txt_tokens->ne[1]; // Or get from input txt_embeddings
        struct ggml_tensor* final_img_tokens = ggml_view_3d(ctx, combined_tokens,
            combined_tokens->ne[0], // hidden_size
            combined_tokens->ne[1] - num_txt_tokens, // L_img
            combined_tokens->ne[2], // N
            combined_tokens->nb[1], // stride L
            combined_tokens->nb[2], // stride N
            num_txt_tokens * combined_tokens->nb[1] // offset by L_txt * stride L
        );


        BlockModulationOutput final_mod_output = get_modulations_for_block(ctx, params_unet, mod_vectors_global, "final", 0);
        // Assuming the type is FINAL for the final layer
        auto& final_mod_pair = final_mod_output.final_mods;
        // Python uses vec=(shift,scale), so order is shift, scale for LastLayer.forward()
        struct ggml_tensor* output_tokens = final_layer->forward(ctx, final_img_tokens, final_mod_pair.first, final_mod_pair.second);
        
        return output_tokens;
    }
};

struct ChromaRunner : public GGMLRunner {
    ChromaParams chroma_hyperparams; // Store the configured hyperparameters
    ChromaUNet_ggml chroma_unet;

    ChromaRunner(
        ggml_backend_t backend,
        std::map<std::string, enum ggml_type>& tensor_types,
        const std::string prefix =  "",
        bool use_flash_attn = false
    ) :
        GGMLRunner(backend),
        chroma_hyperparams({}),
        chroma_unet(chroma_hyperparams)         // Initialize UNet with these params
    {
        chroma_unet.init(params_ctx, tensor_types, prefix);
    }

    std::string get_desc() override {
        return "chroma";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix)  {
        chroma_unet.get_param_tensors(tensors, prefix);
    }

    // Assuming img_latent_tokens, timestep_for_approximator_input_vec, txt_tokens are prepared outside
    struct ggml_cgraph* build_graph(
        struct ggml_tensor* img_latent_tokens, // [N, L_img, C_img_in_to_unet]
        struct ggml_tensor* timestep_for_approximator_input_vec,    // [N, L_indices_for_approximator, C_concat_for_approximator_input]
        struct ggml_tensor* txt_tokens,      // [N, L_txt, C_txt_in_to_unet] (after T5 + projection)
        struct ggml_tensor* pe,                // Positional embeddings
        struct ggml_tensor* t5_padding_mask,   // Attention mask
        std::vector<int> skip_layers = {}
    ) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, CHROMA_GRAPH_SIZE, false);

        img_latent_tokens = to_backend(img_latent_tokens);
        timestep_for_approximator_input_vec    = to_backend(timestep_for_approximator_input_vec);
        txt_tokens        = to_backend(txt_tokens);
        pe                = to_backend(pe);
        if (t5_padding_mask) t5_padding_mask = to_backend(t5_padding_mask);

        struct ggml_tensor* output = chroma_unet.forward(
            compute_ctx, img_latent_tokens, timestep_for_approximator_input_vec, txt_tokens, pe, t5_padding_mask, skip_layers
        );

        ggml_build_forward_expand(gf, output);
        return gf;
    }

    void compute(
        int n_threads,
        struct ggml_tensor* img_latent_tokens,
        struct ggml_tensor* timestep_for_approximator_input_vec,
        struct ggml_tensor* txt_tokens,
        struct ggml_tensor* pe,
        struct ggml_tensor* t5_padding_mask,
        struct ggml_tensor** output = NULL,
        struct ggml_context* output_ctx = NULL,
        std::vector<int> skip_layers = {}
    ) {
        auto get_graph_fn = [&]() -> struct ggml_cgraph* { // Renamed lambda variable
            return build_graph(img_latent_tokens, timestep_for_approximator_input_vec, txt_tokens, pe, t5_padding_mask, skip_layers);
        };
        GGMLRunner::compute(get_graph_fn, n_threads, false, output, output_ctx); // Use renamed lambda
    }
};


} // namespace Chroma
#endif // __CHROMA_HPP__