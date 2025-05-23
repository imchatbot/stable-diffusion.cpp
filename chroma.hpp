#ifndef __CHROMA_HPP__
#define __CHROMA_HPP__

#define CHROMA_GRAPH_SIZE 10240

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath> // For std::sqrt, std::exp, std::log, std::min, std::max
#include <algorithm> // For std::min, std::max

#include "ggml_extend.hpp"
#include "model.h"

namespace Chroma { // Added comment to force recompile

__STATIC_INLINE__ struct ggml_tensor* attention(struct ggml_context* ctx,
                                                struct ggml_tensor* q,
                                                struct ggml_tensor* k,
                                                struct ggml_tensor* v,
                                                struct ggml_tensor* pe,
                                                struct ggml_tensor* attn_mask = NULL) {
    // q,k,v: [N, L, n_head, d_head]
    // pe: [L, d_head/2, 2, 2]
    // return: [N, L, n_head*d_head]

    // Apply RoPE (Rotary Positional Embeddings) - Adaptation from Flux
    // Note: Chroma's RoPE might differ slightly, this is based on Flux's implementation for now.
    // Need to verify Chroma's specific RoPE application if different.
    // Assuming pe is already in the correct format [L, d_head/2, 2, 2] or similar for broadcasting

    // Reshape q, k, v for RoPE application if needed (Flux's apply_rope expects [N, L, n_head, d_head])
    // Assuming q, k, v are already in the expected shape [N, L, n_head, d_head] based on SelfAttention pre_attention output

    // Apply RoPE to q and k
    // Note: The exact implementation of apply_rope is not in ggml_extend.hpp or provided.
    // We will use a placeholder or assume a compatible ggml operation exists or will be added.
    // For now, let's assume a function `ggml_apply_rope` exists that takes q, k, and pe.
    // If not, we'll need to implement the RoPE logic using basic ggml operations.

    // Placeholder for RoPE application - assuming ggml_apply_rope exists or similar logic is used
    // struct ggml_tensor* q_rope = ggml_apply_rope(ctx, q, pe);
    // struct ggml_tensor* k_rope = ggml_apply_rope(ctx, k, pe);

    // For now, let's proceed without explicit RoPE application in this placeholder,
    // as the plan's pseudo-code for SelfAttention forward doesn't explicitly show RoPE calls,
    // but the DoubleStreamBlock pseudo-code does. This suggests RoPE might be handled differently
    // or within the attention function itself in Chroma.
    // Let's use the ggml_flash_attn_ext directly with the provided q, k, v, pe, and mask.

    // Compute attention using ggml_flash_attn_ext
    // ggml_flash_attn_ext expects q, k, v in shape [N*n_head, L, d_head] or similar depending on implementation
    // The plan's pseudo-code for SingleStreamBlock shows q, k, v reshaped to [batch_size, num_heads, sequence_length, head_dim]
    // and then permuted before attention. Let's follow that pattern for the inputs to ggml_flash_attn_ext.

    int64_t N = q->ne[3]; // Batch size
    int64_t L_q = q->ne[2]; // Sequence length Q
    int64_t L_k = k->ne[2]; // Sequence length K
    int64_t n_head = q->ne[1]; // Number of heads
    int64_t d_head = q->ne[0]; // Head dimension

    // Reshape and permute q, k, v for ggml_flash_attn_ext if necessary
    // Assuming the inputs q, k, v to this function are already in the required format for ggml_flash_attn_ext
    // based on the SelfAttention pre_attention output.
    // If pre_attention outputs [N, L, n_head, d_head], we need to reshape/permute here.
    // Let's assume pre_attention outputs [N*n_head, L, d_head] for now, consistent with ggml_nn_attention_ext usage in flux.hpp.

    float scale = 1.0f / std::sqrt((float)d_head);

    struct ggml_tensor* attn_output = ggml_flash_attn_ext(ctx, q, k, v, attn_mask, scale, 0, 0);

    // Reshape the output back to [N, L, n_head*d_head]
    // Assuming attn_output is [N*n_head, L, d_head]
    attn_output = ggml_reshape_4d(ctx, attn_output, d_head, L_q, n_head, N); // [N, n_head, L, d_head]
    attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3); // [N, L, n_head, d_head]
    attn_output = ggml_cont(ctx, attn_output);
    attn_output = ggml_reshape_3d(ctx, attn_output, d_head * n_head, L_q, N); // [N, L, n_head*d_head]


    return attn_output;
}


__STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                               struct ggml_tensor* x,
                                               struct ggml_tensor* shift,
                                               struct ggml_tensor* scale) {
    // x: [N, L, C] or [N, C]
    // scale: [N, C] or [N, 1, C]
    // shift: [N, C] or [N, 1, C]

    // Reshape scale and shift for broadcasting if x has more dimensions than scale/shift
    struct ggml_tensor* scale_reshaped = scale;
    struct ggml_tensor* shift_reshaped = shift;

    if (ggml_n_dims(x) == 3 && ggml_n_dims(scale) == 2) {
        scale_reshaped = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]); // [C, 1, N]
        shift_reshaped = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]); // [C, 1, N]
    } else if (ggml_n_dims(x) == 4 && ggml_n_dims(scale) == 2) {
         scale_reshaped = ggml_reshape_4d(ctx, scale, scale->ne[0], 1, 1, scale->ne[1]); // [C, 1, 1, N]
         shift_reshaped = ggml_reshape_4d(ctx, shift, shift->ne[0], 1, 1, shift->ne[1]); // [C, 1, 1, N]
    }

    // Explicitly repeat scale_reshaped and shift_reshaped to match the shape of x
    struct ggml_tensor* scale_repeated = ggml_repeat(ctx, scale_reshaped, x);
    struct ggml_tensor* shift_repeated = ggml_repeat(ctx, shift_reshaped, x);

    // Apply modulation: (1 + scale) * x + shift
    // Create a tensor of ones with the same shape as scale_repeated for addition
    struct ggml_tensor* ones_tensor = ggml_new_tensor(ctx, scale_repeated->type, ggml_n_dims(scale_repeated), scale_repeated->ne);
    ggml_set_f32(ones_tensor, 1.0f); // Set all elements to 1.0f

    struct ggml_tensor* one_plus_scale = ggml_add(ctx, ones_tensor, scale_repeated);
    struct ggml_tensor* modulated_x = ggml_mul(ctx, x, one_plus_scale);
    struct ggml_tensor* output = ggml_add(ctx, modulated_x, shift_repeated);

    return output;
}
// Adapted from Flux::RMSNorm
struct RMSNorm : public UnaryBlock {
protected:
    int64_t hidden_size;
    float eps;

    void init_params(struct ggml_context* ctx, std::map<std::string, enum ggml_type>& tensor_types, const std::string prefix = "") {
        // Note: Chroma's LayerNorm/RMSNorm with elementwise_affine=False do not have learnable weights/biases.
        // This implementation includes weights for potential future use or if the plan's interpretation of elementwise_affine=False is slightly off for RMSNorm.
        // Based on the plan (Phase 1.2, Approximator), RMSNorm *does* have weights.
        ggml_type wtype = GGML_TYPE_F32;
        params["weight"] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
    }

public:
    RMSNorm(int64_t hidden_size,
            float eps = 1e-06f)
        : hidden_size(hidden_size),
          eps(eps) {}

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        struct ggml_tensor* w = params["weight"];
        x = ggml_rms_norm(ctx, x, eps);
        x = ggml_mul(ctx, x, w);
        return x;
    }
};

// Adapted from Flux::MLPEmbedder
struct MLPEmbedder : public UnaryBlock {
public:
    MLPEmbedder(int64_t in_dim, int64_t hidden_dim) {
        blocks["in_layer"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        blocks["out_layer"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim, hidden_dim, true));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [..., in_dim]
        // return: [..., hidden_dim]
        auto in_layer = std::dynamic_pointer_cast<Linear>(blocks["in_layer"]);
        auto out_layer = std::dynamic_pointer_cast<Linear>(blocks["out_layer"]);

        x = in_layer->forward(ctx, x);
        x = ggml_silu_inplace(ctx, x); // Using ggml_silu_inplace as seen in Flux
        x = out_layer->forward(ctx, x);
        return x;
    }
};


// Based on the plan (Phase 1.2 and 2.5)
struct Approximator_ggml : public UnaryBlock {
    int in_dim;
    int out_dim;
    int hidden_dim;
    int n_layers;

    Approximator_ggml(int in_dim, int out_dim, int hidden_dim, int n_layers = 5)
        : in_dim(in_dim), out_dim(out_dim), hidden_dim(hidden_dim), n_layers(n_layers) {
        blocks["in_proj"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden_dim, true));
        for (int i = 0; i < n_layers; ++i) {
            blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new MLPEmbedder(hidden_dim, hidden_dim));
            blocks["norms." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new RMSNorm(hidden_dim));
        }
        blocks["out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_dim , out_dim));
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* timestep) {
        // Implement forward pass based on the pseudo-code in the plan (Phase 2.2)
        auto in_proj = std::dynamic_pointer_cast<Linear>(blocks["in_proj"]);
        auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

        // 1. in_proj_output = ggml_mul_mat(ctx, model.in_proj_weight, timestep);
        // 2. in_proj_output = ggml_add(ctx, in_proj_output, model.in_proj_bias);
        struct ggml_tensor* current_input = in_proj->forward(ctx, timestep);

        // 3. Loop through layers:
        for (int i = 0; i < n_layers; ++i) {
            auto layer = std::dynamic_pointer_cast<MLPEmbedder>(blocks["layers." + std::to_string(i)]);
            auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["norms." + std::to_string(i)]);

            // *   norm_output = ggml_rms_norm(ctx, current_input, 1e-6f); (using the appropriate RMSNorm weight and an epsilon of 1e-6f)
            struct ggml_tensor* norm_output = norm->forward(ctx, current_input); // RMSNorm forward handles weight and epsilon

            // *   mlp_output = ggml_mul_mat(ctx, model.mlp_embedder_weights[i], norm_output);
            // *   mlp_output = ggml_add(ctx, mlp_output, model.mlp_embedder_biases[i]);
            struct ggml_tensor* mlp_output = layer->forward(ctx, norm_output);

            // *   current_input = ggml_add(ctx, current_input, mlp_output); (skip connection)
            current_input = ggml_add(ctx, current_input, mlp_output);
        }
        
        // 4. out_proj_output = ggml_mul_mat(ctx, model.out_proj_weight, current_input);
        // 5. out_proj_output = ggml_add(ctx, out_proj_output, model.out_proj_bias);
        struct ggml_tensor* out_proj_output = out_proj->forward(ctx, current_input);

        // 6. Return out_proj_output.
        return out_proj_output;
    }
};

// Basic struct definitions for other Chroma modules, inheriting from appropriate base classes

// Based on the plan (Phase 1.2 and 2.5)
struct QKNorm : public GGMLBlock {
    int64_t dim;
    QKNorm(int64_t dim) : dim(dim) {
        blocks["query_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
        blocks["key_norm"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim));
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

// Based on the plan (Phase 1.2 and 2.5)
struct SelfAttention : public GGMLBlock {
    int64_t dim;
    int64_t num_heads;
    bool qkv_bias;

    SelfAttention(int64_t dim, int64_t num_heads, bool qkv_bias = false)
        : dim(dim), num_heads(num_heads), qkv_bias(qkv_bias) {
        int64_t head_dim = dim / num_heads;
        blocks["qkv"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * 3, qkv_bias));
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
        blocks["proj"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
    }
    std::vector<struct ggml_tensor*> pre_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
        auto norm = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);

        auto qkv = qkv_proj->forward(ctx, x); // [N, L, dim*3]
        // Split qkv into q, k, v [N, L, dim]
        int64_t N = x->ne[2];
        int64_t L = x->ne[1];
        int64_t dim = x->ne[0];
        int64_t head_dim = dim / num_heads;

        auto qkv_split = ggml_reshape_4d(ctx, qkv, dim, 3, L, N); // [N, L, dim*3] -> [dim, 3, L, N]
        qkv_split = ggml_cont(ctx, ggml_permute(ctx, qkv_split, 0, 2, 1, 3)); // [dim, 3, L, N] -> [dim, L, 3, N]

        int64_t offset = qkv_split->nb[2] * qkv_split->ne[2];
        auto q = ggml_view_3d(ctx, qkv_split, dim, L, N, qkv_split->nb[1], qkv_split->nb[2], offset * 0); // [dim, L, N]
        auto k = ggml_view_3d(ctx, qkv_split, dim, L, N, qkv_split->nb[1], qkv_split->nb[2], offset * 1); // [dim, L, N]
        auto v = ggml_view_3d(ctx, qkv_split, dim, L, N, qkv_split->nb[1], qkv_split->nb[2], offset * 2); // [dim, L, N]

        // Reshape q, k, v for QKNorm and ggml_nn_attention_ext
        // QKNorm expects [..., dim], ggml_nn_attention_ext expects [d_head, L, N*n_head] or similar depending on implementation
        // Let's reshape q, k, v to [d_head, L, N*n_head] and [d_head, L, n_head, N] respectively

        auto q_reshaped = ggml_reshape_3d(ctx, q, head_dim, L, N * num_heads); // [dim, L, N] -> [d_head, L, N*n_head]
        auto k_reshaped = ggml_reshape_3d(ctx, k, head_dim, L, N * num_heads); // [dim, L, N] -> [d_head, L, N*n_head]
        auto v_reshaped = ggml_reshape_4d(ctx, v, head_dim, L, num_heads, N); // [dim, L, N] -> [d_head, L, n_head, N]

        // Apply QKNorm to q and k
        auto q_normed = norm->query_norm(ctx, q_reshaped);
        auto k_normed = norm->key_norm(ctx, k_reshaped);

        return {q_normed, k_normed, v_reshaped};
    }

    struct ggml_tensor* post_attention(struct ggml_context* ctx, struct ggml_tensor* x) {
        auto proj = std::dynamic_pointer_cast<Linear>(blocks["proj"]);
        return proj->forward(ctx, x); // [N, L, dim]
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pe, struct ggml_tensor* attn_mask = NULL) {
        // x: [N, L, dim]
        // pe: Positional embeddings (shape TBD)
        // attn_mask: Attention mask (shape TBD)
        // return [N, L, dim]

        auto qkv_vec = pre_attention(ctx, x); // q, k: [d_head, L, N*n_head], v: [d_head, L, n_head, N]
        auto q = qkv_vec[0];
        auto k = qkv_vec[1];
        auto v = qkv_vec[2];

        // Compute attention using ggml_nn_attention_ext
        // ggml_nn_attention_ext handles internal reshaping/permuting and RoPE if skip_reshape is false
        struct ggml_tensor* attn_output = ggml_nn_attention_ext(ctx, q, k, v, num_heads, attn_mask, false, false, true); // Assuming flash_attn=true

        // Apply post-attention projection
        struct ggml_tensor* output = post_attention(ctx, attn_output); // [N, L, dim]

        return output;
    }
};

// Based on the plan (Phase 1.2 and 2.5)
struct ModulationOut {
    ggml_tensor* shift = NULL;
    ggml_tensor* scale = NULL;
    ggml_tensor* gate  = NULL;

    ModulationOut(ggml_tensor* shift = NULL, ggml_tensor* scale = NULL, ggml_tensor* gate = NULL)
        : shift(shift), scale(scale), gate(gate) {}
};

// Struct to hold Chroma-specific conditioning data
struct ChromaCondition {
    ggml_tensor* txt_embeddings = NULL;
    ggml_tensor* t5_padding_mask = NULL;
};

// Based on the plan (Phase 3.1)
__STATIC_INLINE__ struct ggml_tensor* generate_t5_padding_mask_ggml(ggml_context* ctx, const std::vector<int>& tokens, int image_sequence_length, int num_heads) {
    // The mask should be 0 for valid tokens and -INFINITY for padding tokens.
    // The shape of the mask should be (1, num_heads, image_sequence_length, tokens.size()).
    // This implies ne0 = tokens.size(), ne1 = image_sequence_length, ne2 = num_heads, ne3 = 1.

    // Create a new tensor for the mask
    ggml_tensor* mask_tensor = ggml_new_tensor_4d(
        ctx,
        GGML_TYPE_F32, // Mask is typically float
        tokens.size(),         // ne0: text sequence length
        image_sequence_length, // ne1: image sequence length
        num_heads,             // ne2: number of heads
        1                      // ne3: batch size (always 1 for this context)
    );

    // Initialize the mask tensor with zeros (representing valid tokens)
    ggml_set_zero(mask_tensor);

    // Define a large negative value to represent masking
    const float negative_infinity = -1e9; // Use a large negative number representable in F32

    // T5 padding token ID is 0 based on t5.hpp
    const int t5_padding_token_id = 0;

    float* mask_data = ggml_get_data_f32(mask_tensor);

    // Iterate through the text tokens to identify padding tokens
    for (int j = 0; j < tokens.size(); ++j) {
        if (tokens[j] == t5_padding_token_id) {
            // If the j-th token is a padding token, mask all attention scores
            // from image tokens to this padding token across all heads.
            // The mask shape is (1, num_heads, image_sequence_length, tokens.size()).
            // This means for each head (k), and for each image token (i),
            // if the j-th text token is padding, set mask_tensor[0][k][i][j] to -INFINITY.

            // The ggml_tensor data is typically stored in row-major order for 4D tensors:
            // data[n3 * s3 + n2 * s2 + n1 * s1 + n0 * s0]
            // where s0 = 1, s1 = ne0, s2 = ne0 * ne1, s3 = ne0 * ne1 * ne2
            // For our mask_tensor:
            // ne0 = tokens.size()
            // ne1 = image_sequence_length
            // ne2 = num_heads
            // ne3 = 1 (batch size)

            // Index calculation: mask_data[ (batch_idx * ne2 * ne1 * ne0) + (head_idx * ne1 * ne0) + (image_idx * ne0) + text_idx ]
            // Since batch_idx is always 0:
            // mask_data[ (head_idx * image_sequence_length * tokens.size()) + (image_idx * tokens.size()) + text_idx ]

            for (int k = 0; k < num_heads; ++k) { // Iterate over heads
                for (int i = 0; i < image_sequence_length; ++i) { // Iterate over image sequence length
                    // Calculate the index in the flattened mask data
                    int index = (k * image_sequence_length * tokens.size()) + (i * tokens.size()) + j;
                    mask_data[index] = negative_infinity;
                }
            }
        }
    }

    return mask_tensor;
}


// Based on the plan (Phase 1.2 and 2.5)
struct Modulation : public GGMLBlock {
    int64_t dim;
    bool is_double;
    int multiplier;

    Modulation(int64_t dim, bool is_double)
        : dim(dim), is_double(is_double) {
        multiplier = is_double ? 6 : 3;
        blocks["lin"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * multiplier));
    }

    std::vector<ModulationOut> forward(struct ggml_context* ctx, struct ggml_tensor* vec) {
        auto lin = std::dynamic_pointer_cast<Linear>(blocks["lin"]);

        auto out = ggml_silu(ctx, vec);
        out      = lin->forward(ctx, out);  // [N, multiplier*dim]

        auto m = ggml_reshape_3d(ctx, out, vec->ne[0], multiplier, vec->ne[1]);  // [N, multiplier, dim]
        m      = ggml_cont(ctx, ggml_permute(ctx, m, 0, 2, 1, 3));               // [multiplier, N, dim]

        int64_t offset = m->nb[1] * m->ne[1];
        auto shift_0   = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 0);  // [N, dim]
        auto scale_0   = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 1);  // [N, dim]
        auto gate_0    = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 2);  // [N, dim]

        if (is_double) {
            auto shift_1 = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 3);  // [N, dim]
            auto scale_1 = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 4);  // [N, dim]
            auto gate_1  = ggml_view_2d(ctx, m, m->ne[0], m->ne[1], m->nb[1], offset * 5);  // [N, dim]
            return {ModulationOut(shift_0, scale_0, gate_0), ModulationOut(shift_1, scale_1, gate_1)};
        }

        return {ModulationOut(shift_0, scale_0, gate_0), ModulationOut()};
    }
};

// Based on the plan (Phase 1.2 and 2.5)
struct SingleStreamBlock_ggml : public GGMLBlock {
    int64_t hidden_size;
    int64_t num_heads;
    float mlp_ratio;
    float qk_scale;

    SingleStreamBlock_ggml(int64_t hidden_size, int64_t num_heads, float mlp_ratio = 4.0f, float qk_scale = 0.f)
        : hidden_size(hidden_size), num_heads(num_heads), mlp_ratio(mlp_ratio), qk_scale(qk_scale) {
        int64_t head_dim = hidden_size / num_heads;
        float scale = qk_scale;
        if (scale <= 0.f) {
            scale = 1.0f / std::sqrt((float)head_dim);
        }

        int64_t mlp_hidden_dim = hidden_size * mlp_ratio;

        blocks["linear1"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim));
        blocks["linear2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size + mlp_hidden_dim, hidden_size));
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new QKNorm(head_dim));
        blocks["pre_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false)); // elementwise_affine=False
        // mlp_act is nn.GELU(approximate="tanh") - handled by ggml_gelu
        // Modulation block is created and called in the forward pass based on the plan
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pe, struct ModulationOut& mod, struct ggml_tensor* attn_mask = NULL) {
        // x: [N, L, hidden_size]
        // pe: Positional embeddings (shape TBD)
        // mod: Modulation parameters (scale, shift, gate)
        // attn_mask: Attention mask (optional)
        // return: [N, L, hidden_size]

        auto linear1 = std::dynamic_pointer_cast<Linear>(blocks["linear1"]);
        auto linear2 = std::dynamic_pointer_cast<Linear>(blocks["linear2"]);
        auto norm = std::dynamic_pointer_cast<QKNorm>(blocks["norm"]);
        auto pre_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["pre_norm"]);

        // 1. Apply pre-normalization and modulation
        struct ggml_tensor* x_norm = pre_norm->forward(ctx, x); // [N, L, hidden_size]
        struct ggml_tensor* x_mod = Chroma::modulate(ctx, x_norm, mod.shift, mod.scale); // [N, L, hidden_size]

        // 2. Pass modulated input through linear1 to get combined QKV and MLP input
        struct ggml_tensor* linear1_output = linear1->forward(ctx, x_mod); // [N, L, hidden_size * 3 + mlp_hidden_dim]
        
        // 3. Split linear1 output into QKV and MLP input tensors
        // 4. Reshape QKV tensors and apply QKNorm to Q and K
        int64_t N = x->ne[2];
        int64_t L = x->ne[1];
        int64_t hidden_size = x->ne[0];
        int64_t head_dim = hidden_size / num_heads;
        int64_t mlp_hidden_dim = linear1_output->ne[0] - hidden_size * 3;

        struct ggml_tensor* qkv_mlp_split = ggml_reshape_4d(ctx, linear1_output, hidden_size * 3 + mlp_hidden_dim, 1, L, N); // [N, L, total_dim] -> [total_dim, 1, L, N]
        qkv_mlp_split = ggml_cont(ctx, ggml_permute(ctx, qkv_mlp_split, 0, 2, 1, 3)); // [total_dim, 1, L, N] -> [total_dim, L, 1, N]

        struct ggml_tensor* qkv = ggml_view_3d(ctx, qkv_mlp_split, hidden_size * 3, L, N, qkv_mlp_split->nb[1], qkv_mlp_split->nb[2], 0); // [hidden_size*3, L, N]
        struct ggml_tensor* mlp_input = ggml_view_3d(ctx, qkv_mlp_split, mlp_hidden_dim, L, N, qkv_mlp_split->nb[1], qkv_mlp_split->nb[2], qkv_mlp_split->nb[2] * hidden_size * 3); // [mlp_hidden_dim, L, N]

        qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 1, 2, 0, 3)); // [hidden_size*3, L, N] -> [N, L, hidden_size*3]
        mlp_input = ggml_cont(ctx, ggml_permute(ctx, mlp_input, 1, 2, 0, 3)); // [mlp_hidden_dim, L, N] -> [N, L, mlp_hidden_dim]


        struct ggml_tensor* qkv_reshaped = ggml_reshape_4d(ctx, qkv, head_dim, num_heads, L, N); // [N, L, hidden_size*3] -> [d_head, n_head, L, N]
        qkv_reshaped = ggml_cont(ctx, ggml_permute(ctx, qkv_reshaped, 0, 2, 1, 3)); // [d_head, n_head, L, N] -> [d_head, L, n_head, N]

        int64_t qkv_offset = qkv_reshaped->nb[2] * qkv_reshaped->ne[2];
        struct ggml_tensor* q = ggml_view_3d(ctx, qkv_reshaped, head_dim, L, N * num_heads, qkv_reshaped->nb[1], qkv_reshaped->nb[2], qkv_offset * 0); // [d_head, L, N*n_head]
        struct ggml_tensor* k = ggml_view_3d(ctx, qkv_reshaped, head_dim, L, N * num_heads, qkv_reshaped->nb[1], qkv_reshaped->nb[2], qkv_offset * 1); // [d_head, L, N*n_head]
        struct ggml_tensor* v = ggml_view_3d(ctx, qkv_reshaped, head_dim, L, N * num_heads, qkv_reshaped->nb[1], qkv_reshaped->nb[2], qkv_offset * 2); // [d_head, L, N*n_head]

        struct ggml_tensor* q_normed = norm->query_norm(ctx, q);
        struct ggml_tensor* k_normed = norm->key_norm(ctx, k);

        // 5. Compute self-attention
        struct ggml_tensor* attn_output = attention(ctx, q_normed, k_normed, v, pe, attn_mask); // [N, L, hidden_size]

        // 6. Apply GELU activation to MLP input
        struct ggml_tensor* mlp_act_output = ggml_gelu_inplace(ctx, mlp_input); // [N, L, mlp_hidden_dim]

        // 7. Concatenate attention output and activated MLP output
        struct ggml_tensor* concatenated_output = ggml_concat(ctx, attn_output, mlp_act_output, 0); // [N, L, hidden_size + mlp_hidden_dim]

        // 8. Pass the concatenated tensor through linear2
        struct ggml_tensor* linear2_output = linear2->forward(ctx, concatenated_output); // [N, L, hidden_size]

        // 9. Apply modulation and add a skip connection
        struct ggml_tensor* output = ggml_add(ctx, x, ggml_mul(ctx, linear2_output, mod.gate)); // [N, L, hidden_size]

        // 10. Handle potential NaN values (monitoring needed)

        return output;
    }
};

// Based on the plan (Phase 1.2 and 2.5)
struct DoubleStreamBlock_ggml : public GGMLBlock { // DoubleStreamBlock forward returns a pair, so it doesn't fit UnaryBlock
    int64_t hidden_size;
    int64_t num_heads;
    float mlp_ratio;
    bool qkv_bias;
    bool flash_attn;

    DoubleStreamBlock_ggml(int64_t hidden_size,
                           int64_t num_heads,
                           float mlp_ratio,
                           bool qkv_bias   = false,
                           bool flash_attn = false)
        : hidden_size(hidden_size), num_heads(num_heads), mlp_ratio(mlp_ratio), qkv_bias(qkv_bias), flash_attn(flash_attn) {
        int64_t mlp_hidden_dim = hidden_size * mlp_ratio;
        blocks["img_norm1"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
        blocks["img_attn"]     = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias));

        blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
        blocks["img_mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, mlp_hidden_dim));
        blocks["img_mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(mlp_hidden_dim, hidden_size));

        blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
        blocks["txt_attn"]  = std::shared_ptr<GGMLBlock>(new SelfAttention(hidden_size, num_heads, qkv_bias));

        blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
        blocks["txt_mlp.0"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, mlp_hidden_dim));
        blocks["txt_mlp.2"] = std::shared_ptr<GGMLBlock>(new Linear(mlp_hidden_dim, hidden_size));
    }

    std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* img, struct ggml_tensor* txt, struct ggml_tensor* pe, const std::vector<ModulationOut>& img_mods, const std::vector<ModulationOut>& txt_mods, struct ggml_tensor* attn_mask = NULL) {
        auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
        auto img_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["img_attn"]);

        auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
        auto img_mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.0"]);
        auto img_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.2"]);

        auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
        auto txt_attn  = std::dynamic_pointer_cast<SelfAttention>(blocks["txt_attn"]);

        auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
        auto txt_mlp_0 = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.0"]);
        auto txt_mlp_2 = std::dynamic_pointer_cast<Linear>(blocks["txt_mlp.2"]);

        ModulationOut img_mod1 = img_mods[0];
        ModulationOut img_mod2 = img_mods[1];
        ModulationOut txt_mod1 = txt_mods[0];
        ModulationOut txt_mod2 = txt_mods[1];

        // prepare image for attention
        auto img_modulated = img_norm1->forward(ctx, img);
        img_modulated      = Chroma::modulate(ctx, img_modulated, img_mod1.shift, img_mod1.scale);
        auto img_qkv       = img_attn->pre_attention(ctx, img_modulated);
        auto img_q         = img_qkv[0];
        auto img_k         = img_qkv[1];
        auto img_v         = img_qkv[2];

        // prepare txt for attention
        auto txt_modulated = txt_norm1->forward(ctx, txt);
        txt_modulated      = Chroma::modulate(ctx, txt_modulated, txt_mod1.shift, txt_mod1.scale);
        auto txt_qkv       = txt_attn->pre_attention(ctx, txt_modulated);
        auto txt_q         = txt_qkv[0];
        auto txt_k         = txt_qkv[1];
        auto txt_v         = txt_qkv[2];

        // run actual attention
        auto q = ggml_concat(ctx, txt_q, img_q, 2);
        auto k = ggml_concat(ctx, txt_k, img_k, 2);
        auto v = ggml_concat(ctx, txt_v, img_v, 2);

        auto attn = attention(ctx, q, k, v, pe, attn_mask);

        int64_t N = img->ne[2];
        int64_t L_txt = txt->ne[1];
        int64_t L_img = img->ne[1];
        int64_t hidden_size_attn = attn->ne[0]; // This is n_head * d_head

        // Split attention output back into image and text streams
        // attn is [N, L_txt + L_img, hidden_size_attn]
        auto txt_attn_out = ggml_view_3d(ctx, attn, hidden_size_attn, L_txt, N, attn->nb[1], attn->nb[2], 0);
        auto img_attn_out = ggml_view_3d(ctx, attn, hidden_size_attn, L_img, N, attn->nb[1], attn->nb[2], attn->nb[2] * L_txt);

        // calculate the img blocks
        img = ggml_add(ctx, img, ggml_mul(ctx, img_attn->post_attention(ctx, img_attn_out), img_mod1.gate));

        auto img_mlp_out = img_mlp_0->forward(ctx, Chroma::modulate(ctx, img_norm2->forward(ctx, img), img_mod2.shift, img_mod2.scale));
        img_mlp_out      = ggml_gelu_inplace(ctx, img_mlp_out);
        img_mlp_out      = img_mlp_2->forward(ctx, img_mlp_out);

        img = ggml_add(ctx, img, ggml_mul(ctx, img_mlp_out, img_mod2.gate));

        // calculate the txt blocks
        txt = ggml_add(ctx, txt, ggml_mul(ctx, txt_attn->post_attention(ctx, txt_attn_out), txt_mod1.gate));

        auto txt_mlp_out = txt_mlp_0->forward(ctx, Chroma::modulate(ctx, txt_norm2->forward(ctx, txt), txt_mod2.shift, txt_mod2.scale));
        txt_mlp_out      = ggml_gelu_inplace(ctx, txt_mlp_out);
        txt_mlp_out      = txt_mlp_2->forward(ctx, txt_mlp_out);

        txt = ggml_add(ctx, txt, ggml_mul(ctx, txt_mlp_out, txt_mod2.gate));

        return {img, txt};
    }
};

// Based on the plan (Phase 1.2 and 2.5)
struct LastLayer_ggml : public GGMLBlock {
    int64_t hidden_size;
    int64_t patch_size;
    int64_t out_channels;

    LastLayer_ggml(int64_t hidden_size, int64_t patch_size, int64_t out_channels)
        : hidden_size(hidden_size), patch_size(patch_size), out_channels(out_channels) {
        blocks["norm_final"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size, 1e-6f, false));
        blocks["linear"]     = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, patch_size * patch_size * out_channels));
    }



    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* shift, struct ggml_tensor* scale) {
        auto norm_final = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
        auto linear     = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

        // 1. Apply final normalization
        struct ggml_tensor* x_norm = norm_final->forward(ctx, x);

        // 2. Apply modulation: (1 + scale) * norm(x) + shift
        // Reshape shift and scale for broadcasting
        // x_norm: [N, L, hidden_size]
        // shift, scale: [N, hidden_size]
        struct ggml_tensor* shift_reshaped = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]); // [N, 1, hidden_size]
        struct ggml_tensor* scale_reshaped = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]); // [N, 1, hidden_size]

        struct ggml_tensor* ones_tensor = ggml_new_tensor(ctx, scale_reshaped->type, ggml_n_dims(scale_reshaped), scale_reshaped->ne);
        ggml_set_f32(ones_tensor, 1.0f);
        struct ggml_tensor* one_plus_scale = ggml_add(ctx, ones_tensor, scale_reshaped);
        struct ggml_tensor* modulated_x    = ggml_mul(ctx, x_norm, one_plus_scale);
        modulated_x                        = ggml_add(ctx, modulated_x, shift_reshaped);

        // 3. Pass the modulated input through the linear layer
        struct ggml_tensor* output = linear->forward(ctx, modulated_x);

        return output;
    }
};

// Based on the plan (Phase 2.1 and 2.5)
struct ChromaUNet_ggml : public GGMLBlock {
    int64_t hidden_size;
    int64_t num_heads;
    float mlp_ratio;
    int64_t depth;
    int64_t single_depth;
    int64_t in_channels;
    int64_t out_channels;
    bool flash_attn;

    ChromaUNet_ggml(int64_t hidden_size,
                    int64_t num_heads,
                    float mlp_ratio,
                    int64_t depth,
                    int64_t single_depth,
                    int64_t in_channels,
                    int64_t out_channels,
                    bool flash_attn)
        : hidden_size(hidden_size), num_heads(num_heads), mlp_ratio(mlp_ratio),
          depth(depth), single_depth(single_depth), in_channels(in_channels),
          out_channels(out_channels), flash_attn(flash_attn) {
        blocks["distilled_guidance_layer"] = std::shared_ptr<GGMLBlock>(new Approximator_ggml(64, hidden_size * 6 , hidden_size)); // out_dim is hidden_size * 6 for 2 sets of scale/shift/gate

        blocks["img_in"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, hidden_size, true));
        blocks["txt_in"] = std::shared_ptr<GGMLBlock>(new Linear(hidden_size, hidden_size, true)); // T5 embeddings are already hidden_size

        for (int i = 0; i < depth; ++i) {
            blocks["double_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new DoubleStreamBlock_ggml(hidden_size, num_heads, mlp_ratio, true, flash_attn));
        }

        for (int i = 0; i < single_depth; ++i) {
            blocks["single_blocks." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(new SingleStreamBlock_ggml(hidden_size, num_heads, mlp_ratio, 0.f));
        }

        blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new LastLayer_ggml(hidden_size, 1, out_channels)); // patch_size is 1 for Chroma
    }


    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* img_latent,
                                struct ggml_tensor* timestep,
                                struct ggml_tensor* txt_embeddings,
                                struct ggml_tensor* pe,
                                struct ggml_tensor* t5_padding_mask,
                                std::vector<int> skip_layers = std::vector<int>()) {
        auto approximator = std::dynamic_pointer_cast<Approximator_ggml>(blocks["distilled_guidance_layer"]);
        auto img_in       = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
        auto txt_in       = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
        auto final_layer  = std::dynamic_pointer_cast<LastLayer_ggml>(blocks["final_layer"]);

        // 1. Process timestep through Approximator to get modulation signals
        struct ggml_tensor* approx_output = approximator->forward(ctx, timestep); // [N, hidden_size * 6]

        // Initialize image and text inputs
        img_latent = img_in->forward(ctx, img_latent);
        txt_embeddings = txt_in->forward(ctx, txt_embeddings);

        // DoubleStreamBlocks
        for (int i = 0; i < depth; ++i) {
            if (std::find(skip_layers.begin(), skip_layers.end(), i) != skip_layers.end()) {
                continue;
            }
            auto block = std::dynamic_pointer_cast<DoubleStreamBlock_ggml>(blocks["double_blocks." + std::to_string(i)]);

            // Need to create ModulationOut vectors for img and txt
            // This implies the approx_output needs to be split into 4 sets of (scale, shift, gate)
            // Or the Modulation class needs to be called multiple times.

            // Let's create two Modulation instances here, one for img and one for txt.
            // And pass the relevant part of approx_output to them.
            // This means approx_output needs to be split into two halves.

            int64_t N = img_latent->ne[2]; // Batch size

            struct ggml_tensor* approx_img_vec = ggml_view_2d(ctx, approx_output, hidden_size * 6, N, approx_output->nb[1], 0);
            struct ggml_tensor* approx_txt_vec = ggml_view_2d(ctx, approx_output, hidden_size * 6, N, approx_output->nb[1], approx_output->nb[1] * hidden_size * 6);

            Modulation img_mod_block(hidden_size, true);
            Modulation txt_mod_block(hidden_size, true);

            std::vector<ModulationOut> img_mods = img_mod_block.forward(ctx, approx_img_vec);
            std::vector<ModulationOut> txt_mods = txt_mod_block.forward(ctx, approx_txt_vec);

            auto img_txt = block->forward(ctx, img_latent, txt_embeddings, pe, img_mods, txt_mods, t5_padding_mask);
            img_latent   = img_txt.first;
            txt_embeddings = img_txt.second;
        }

        // Concatenate image and text tokens for SingleStreamBlocks
        // The plan says: `auto txt_img = ggml_concat(ctx, txt, img, 1);`
        // This means txt_embeddings and img_latent are concatenated.
        struct ggml_tensor* combined_tokens = ggml_concat(ctx, txt_embeddings, img_latent, 1);

        // SingleStreamBlocks
        for (int i = 0; i < single_depth; ++i) {
            if (std::find(skip_layers.begin(), skip_layers.end(), i + depth) != skip_layers.end()) {
                continue;
            }
            auto block = std::dynamic_pointer_cast<SingleStreamBlock_ggml>(blocks["single_blocks." + std::to_string(i)]);

            // SingleStreamBlock takes one ModulationOut.
            // This means approx_output needs to be split into one set of (scale, shift, gate).
            // This is inconsistent with the DoubleStreamBlock.

            // Re-reading the plan:
            // Approximator output: `conditioning signal (Tensor of shape [batch_size, 1, out_dim])`, likely split into scale, shift, gate.
            // SingleStreamBlock: `mod = vec`
            // DoubleStreamBlock: `(img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec`

            // This implies `vec` is a complex structure.
            // Let's assume `approx_output` is the `vec` and it contains all necessary modulation parameters.
            // And the `Modulation` class will be used to extract them.

            // Let's assume `approx_output` is `hidden_size * 6` (for 2 sets of modulations).
            // And `SingleStreamBlock` will use the first set.

            Modulation single_mod_block(hidden_size, false);
            std::vector<ModulationOut> single_mods = single_mod_block.forward(ctx, approx_output);
            ModulationOut single_mod = single_mods[0];

            combined_tokens = block->forward(ctx, combined_tokens, pe, single_mod, t5_padding_mask);
        }

        // Final Layer
        // The plan says: `LastLayer(nn.Module)` takes `x: Tensor, vec: Tensor`.
        // `shift, scale = vec`.
        // So, `final_layer` needs `shift` and `scale`.
        // This means `approx_output` needs to contain `shift` and `scale` for the final layer.

        // Let's assume `approx_output` contains `shift` and `scale` for the final layer.
        // And we extract them.

        // This is inconsistent. The `Approximator` output is `out_dim`.
        // And `out_dim` is `hidden_size * 6`.
        // This means `approx_output` contains 2 sets of (scale, shift, gate).
        // But `LastLayer` needs only `shift` and `scale`.

        // Let's assume the `approx_output` is directly used as `vec` for `LastLayer`.
        // And `LastLayer` will extract `shift` and `scale` from it.
        // This means `approx_output` should contain `hidden_size * 2` for `LastLayer`.

        // Let's simplify the `Approximator` output to be just `hidden_size`.
        // And then we will use `Modulation` blocks to generate the actual modulation parameters.
        // This is how Flux does it.

        // Let's revert `Approximator_ggml` `out_dim` to `hidden_size`.
        // And then create `Modulation` blocks in `ChromaUNet_ggml`.

        // Reverting `Approximator_ggml` `out_dim` to `hidden_size`.
        // And then creating `Modulation` blocks in `ChromaUNet_ggml`.

        // This means `ChromaUNet_ggml` needs to have `Modulation` blocks as members.
        // And `Approximator` output is just a `vec` that is passed to these `Modulation` blocks.

        // Let's assume `approx_output` is the `vec` for all modulation.
        // And `Modulation` class will be used to extract them.

        // Let's assume `approx_output` is `hidden_size * 6` (for 2 sets of modulations).
        // And `LastLayer` will use the first set of `shift` and `scale`.

        // Extract shift and scale for LastLayer from approx_output
        int64_t N = img_latent->ne[2]; // Batch size
        struct ggml_tensor* final_shift = ggml_view_2d(ctx, approx_output, hidden_size, N, approx_output->nb[1], 0);
        struct ggml_tensor* final_scale = ggml_view_2d(ctx, approx_output, hidden_size, N, approx_output->nb[1], approx_output->nb[1] * hidden_size);

        img_latent = final_layer->forward(ctx, combined_tokens, final_shift, final_scale);

        return img_latent;
    }
};

// Struct to hold Chroma model configuration parameters
struct ChromaParams {
    int32_t in_channels = 64;
    int32_t out_channels = 64;
    int32_t vec_in_dim = 768; // Not directly used by ChromaUNet, but kept for consistency with Flux
    int32_t hidden_size = 5120;
    float mlp_ratio = 4.0f;
    int32_t num_heads = 24;
    int32_t depth = 19;
    int32_t single_depth = 38;
    bool flash_attn; // This will be set by the constructor
};

// ChromaRunner struct, inheriting from GGMLRunner
struct ChromaRunner : public GGMLRunner {
    ChromaParams chroma_params;
    ChromaUNet_ggml chroma_unet;

    ChromaRunner(
        ggml_backend_t backend,
        std::map<std::string, enum ggml_type>& tensor_types,
        const std::string prefix                            = "",
        bool flash_attn = false
    ) :
        GGMLRunner(backend),
        chroma_params({
            .flash_attn = flash_attn
        }),
        chroma_unet(
            chroma_params.hidden_size, chroma_params.num_heads, chroma_params.mlp_ratio,
            chroma_params.depth, chroma_params.single_depth,
            chroma_params.in_channels, chroma_params.out_channels, chroma_params.flash_attn
        )
    {
        chroma_unet.init(params_ctx, tensor_types,prefix);
    }

    std::string get_desc() override {
        return "chroma";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) { // Removed override
        chroma_unet.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(
        struct ggml_tensor* x,
        struct ggml_tensor* timesteps,
        struct ggml_tensor* context,
        struct ggml_tensor* t5_padding_mask,
        struct ggml_tensor* pe,
        std::vector<int> skip_layers
    ) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, CHROMA_GRAPH_SIZE, false);

        x = to_backend(x);
        timesteps = to_backend(timesteps);
        context = to_backend(context);
        t5_padding_mask = to_backend(t5_padding_mask);
        pe = to_backend(pe);

        struct ggml_tensor* output = chroma_unet.forward(
            compute_ctx, x, timesteps, context, pe, t5_padding_mask, skip_layers
        );

        ggml_build_forward_expand(gf, output);

        return gf;
    }

    void compute(
        int n_threads,
        struct ggml_tensor* x,
        struct ggml_tensor* timesteps,
        struct ggml_tensor* context,
        struct ggml_tensor* t5_padding_mask,
        struct ggml_tensor* pe,
        struct ggml_tensor** output = NULL,
        struct ggml_context* output_ctx = NULL,
        std::vector<int> skip_layers = std::vector<int>()
    ) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, t5_padding_mask, pe, skip_layers);
        };

        GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

} // namespace Chroma
#endif // __CHROMA_HPP__