// Conceptual examples/linear_layer_test/main.c

// Include the necessary GGML header file.
// This header provides the definitions for GGML structures and functions.
#include "ggml_extend.hpp"
// Include standard I/O for printing output.
#include <stdio.h>
// Include assert for basic sanity checks (optional but good practice).
#include <assert.h>
// Include math.h for fabs (absolute value for floating-point comparison).
#include <math.h>

int main() {
    // Initialize GGML context.
    // The context manages the memory and computation graph for GGML operations.
    // We define initialization parameters, including the memory size.
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024, // Allocate 16MB of memory for the context.
        .mem_buffer = NULL, // NULL means GGML will allocate the buffer on the heap.
        .no_alloc   = false, // Allow GGML to allocate memory for tensors within the context.
    };
    struct ggml_context * ctx = ggml_init(params); // Initialize the context.

    // Check if context initialization was successful.
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1; // Return an error code if initialization fails.
    }

    // Define dimensions for the linear layer.
    int n_input = 3;  // Number of input features.
    int n_output = 2; // Number of output features.

    // Create tensors for weights (W), bias (b), and input (x).
    // ggml_new_tensor_2d creates a 2D tensor (matrix) for weights.
    // Dimensions are (number of columns, number of rows) for matrix multiplication Wx.
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_output);
    // ggml_new_tensor_1d creates a 1D tensor (vector) for bias.
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_output);
    // ggml_new_tensor_1d creates a 1D tensor (vector) for input.
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_input);

    // Check if tensor creation was successful.
    if (!W || !b || !x) {
         fprintf(stderr, "%s: ggml_new_tensor() failed\n", __func__);
         ggml_free(ctx); // Free the context before exiting.
         return 1;
    }

    // Set tensor data with example values.
    // ggml_set_f32 sets all elements of a tensor to a single float value.
    // For W (2x3 matrix):
    // [[1.0, 1.0, 1.0],
    //  [1.0, 1.0, 1.0]]
    // Set tensor data with example values.
    // ggml_set_f32_inplace sets all elements of a tensor to a single float value.
    // For W (2x3 matrix):
    // [[1.0, 1.0, 1.0],
    //  [1.0, 1.0, 1.0]]
    // Set tensor data with example values.
    // Use ggml_set_f32 to set all elements of a tensor to a single float value.
    // For W (2x3 matrix):
    // [[1.0, 1.0, 1.0],
    //  [1.0, 1.0, 1.0]]
    ggml_set_f32(W, 1.0f);
    // For b (2-element vector):
    // [0.5, 0.5]
    ggml_set_f32(b, 0.5f);
    // For x (3-element vector):
    // [2.0, 2.0, 2.0]
    ggml_set_f32(x, 2.0f);

    // Implement linear layer operation (output = Wx + b).
    // ggml_mul_mat performs matrix multiplication.
    // Note: GGML's ggml_mul_mat(ctx, A, B) computes A * B.
    // To compute Wx where W is 2x3 and x is 3x1 (vector), we need to multiply W by x.
    // The result Wx will be a 2x1 vector.
    struct ggml_tensor * Wx = ggml_mul_mat(ctx, W, x);
    // Check if matrix multiplication was successful.
    if (!Wx) {
        fprintf(stderr, "%s: ggml_mul_mat() failed\n", __func__);
        ggml_free(ctx);
        return 1;
    }

    // ggml_add performs element-wise addition.
    // Add the bias vector 'b' to the result of the matrix multiplication 'Wx'.
    // The result 'output' will be a 2x1 vector.
    struct ggml_tensor * output = ggml_add(ctx, Wx, b);
     // Check if addition was successful.
    if (!output) {
        fprintf(stderr, "%s: ggml_add() failed\n", __func__);
        ggml_free(ctx);
        return 1;
    }

    // Build the computation graph.
    // ggml_new_graph creates a new computation graph.
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    // ggml_build_forward_expand builds the forward computation graph starting from the 'output' tensor.
    // It automatically adds all dependencies of 'output' to the graph.
    ggml_build_forward_expand(gf, output);

    // Allocate graph memory and compute the graph.
    // ggml_graph_compute_with_ctx executes the computation graph.
    // The second argument is the graph to compute.
    // The third argument is the number of threads to use (1 in this case).
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Expected output (manual calculation for this simple example).
    // W = [[1, 1, 1], [1, 1, 1]]
    // x = [2, 2, 2]
    // Wx = [1*2 + 1*2 + 1*2, 1*2 + 1*2 + 1*2] = [6, 6]
    // b = [0.5, 0.5]
    // output = [6 + 0.5, 6 + 0.5] = [6.5, 6.5]
    float expected_output[] = { 6.5f, 6.5f };

    // Access the computed output data.
    // ggml_get_data returns a pointer to the tensor's data.
    float * output_data = (float *)ggml_get_data(output);

    // Compare computed output with expected output.
    int errors = 0;
    // Iterate through the elements of the output tensor.
    for (int i = 0; i < n_output; ++i) {
        // Use fabs for floating-point comparison with a tolerance (1e-6).
        if (fabs(output_data[i] - expected_output[i]) > 1e-6) {
            // Print an error message if the computed value is significantly different from the expected value.
            printf("Error at index %d: Expected %f, Got %f\n", i, expected_output[i], output_data[i]);
            errors++; // Increment the error count.
        }
    }

    // Print the test result based on the number of errors.
    if (errors == 0) {
        printf("Linear layer test passed!\n");
    } else {
        printf("Linear layer test failed with %d errors.\n", errors);
    }

    // Free the GGML context.
    // This releases all memory allocated by the context.
    ggml_free(ctx);

    return 0; // Return 0 to indicate successful execution.
}