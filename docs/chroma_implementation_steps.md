# Step-by-Step Implementation Plan with Layer-by-Layer Testing

This plan outlines the steps to integrate the Chroma model into `diffusion.cpp`, focusing on an initial setup phase followed by a layer-by-layer implementation and testing approach for the UNet.

## Step 1: Initial Setup: Prepare Inputs (VAE, Text Encoder, Latent)

This step focuses on setting up the necessary components and inputs for the Chroma model's forward pass.

*   **Load VAE:** Implement the logic within the `ChromaRunner` or a related structure to load the VAE model weights from the GGUF file.
*   **Load Text Encoder:** Implement the logic within the `ChromaRunner` or a related structure to load the T5 XXL text encoder model weights from the GGUF file.
*   **Condition Text Input:** Implement the process to take a text prompt, tokenize it using the T5 tokenizer, and pass the token IDs through the T5 XXL encoder to obtain text embeddings. This will likely involve adapting existing T5 processing logic in [`diffusion.cpp`](stable-diffusion.cpp).
*   **Generate T5 Padding Mask:** Implement the `generate_t5_padding_mask_ggml` function as outlined in the technical plan to create the attention mask for T5 padding tokens.
*   **Create Initial Latent Tensor:** Utilize existing logic within `diffusion.cpp` to create the initial empty latent tensor, which serves as the starting point for the diffusion process.
*   **Verify VAE Compatibility:** Confirm that the existing VAE encoding and decoding logic is compatible with the Chroma model's requirements.

## Step 2: Implement and Test Approximator

*   **Component:** `Approximator_ggml` struct and its `forward` method.
*   **Code:**
    *   Define the `Approximator_ggml` struct in [`chroma.hpp`](chroma.hpp).
    *   Implement the `init_params` method to create GGML tensors for the weights and biases.
    *   Implement the `forward` method using GGML operations ([`ggml_mul_mat`](language.constructs://ggml_mul_mat), [`ggml_add`](language.constructs://ggml_add), [`ggml_rms_norm`](language.constructs://ggml_rms_norm), [`ggml_mul`](language.constructs://ggml_mul)).
*   **Isolated Testing:**
    *   Create a new test function (e.g., `test_approximator` in a new test file or within an existing test structure).
    *   Initialize a GGML context.
    *   Create an instance of `Approximator_ggml`.
    *   Call `init_params` to allocate tensors.
    *   Manually set dummy values for the Approximator's weights and biases.
    *   Create a dummy input tensor for the timestep (e.g., a single value tensor).
    *   Call the `forward` method with the dummy input.
    *   Build and compute the GGML graph for the forward pass.
    *   Verify the shape of the output tensor.
    *   Compare the output tensor values against expected values computed manually or using a simple Python script with the same dummy inputs and weights.
*   **Expected Outcome:** The test should run without errors, and the output tensor's shape and values should match the expected results.

## Step 3: Implement and Test Helper Functions (`ModulationOut`, `modulate`, `QKNorm`)

*   **Component:** `ModulationOut` struct, the `modulate` helper function, and the `QKNorm` struct.
*   **Code:**
    *   Define the `ModulationOut` struct in [`chroma.hpp`](chroma.hpp).
    *   Implement the `modulate` helper function using GGML operations ([`ggml_add`](language.constructs://ggml_add), [`ggml_mul`](language.constructs://ggml_mul), [`ggml_add_const`](language.constructs://ggml_add_const), [`ggml_reshape_3d`](language.constructs://ggml_reshape_3d) or [`ggml_repeat`](language.constructs://ggml_repeat) for broadcasting).
    *   Define the `QKNorm` struct in [`chroma.hpp`](chroma.hpp).
    *   Implement the constructor for `QKNorm` to create nested `RMSNorm` blocks.
    *   Implement `query_norm` and `key_norm` methods for `QKNorm`.
*   **Isolated Testing:**
    *   Create a test function (e.g., `test_modulate`).
    *   Initialize a GGML context.
    *   Create dummy input tensors for `x`, `shift`, and `scale` with various shapes to test broadcasting.
    *   Call the `modulate` function with the dummy inputs.
    *   Build and compute the GGML graph.
    *   Verify the shape and values of the output tensor.
    *   Create a test function (e.g., `test_qknorm`).
    *   Initialize a GGML context.
    *   Create an instance of `QKNorm`.
    *   Call `init_params` on the `QKNorm` instance.
    *   Manually set dummy weights for the nested `RMSNorm` blocks.
    *   Create dummy input tensors for Q and K.
    *   Call `query_norm` and `key_norm`.
    *   Build and compute the GGML graph.
    *   Verify the shape and values of the normalized Q and K tensors.
*   **Expected Outcome:** The tests should run without errors, and the output shapes and values for modulation and QKNorm should be correct.

## Step 4: Implement and Test UNet Blocks Layer by Layer

This step involves implementing the core UNet blocks incrementally and testing the output after each implemented layer or group of layers.

*   **Component:** `SingleStreamBlock_ggml`, `DoubleStreamBlock_ggml`, and `LastLayer_ggml` structs and their `forward` methods.
*   **Code:**
    *   Define the C++ structs for `SingleStreamBlock_ggml`, `DoubleStreamBlock_ggml`, and `LastLayer_ggml` in [`chroma.hpp`](chroma.hpp).
    *   Implement the constructors and `init_params` methods for these blocks, creating nested components (`Linear`, `SelfAttention`, `LayerNorm`, etc.).
    *   Implement the `forward` method for the initial input layers (if any) and the first `DoubleStreamBlock`, utilizing the helper functions and GGML operations as outlined in the technical plan.
    *   Implement the `forward` methods for the remaining `SingleStreamBlock` and `DoubleStreamBlock` layers, following the Chroma UNet architecture.
    *   Implement the `forward` method for the `LastLayer`.
*   **Layer-by-Layer Testing:**
    *   After implementing the forward pass for the first block(s), create a test case that feeds dummy inputs (image latent, text embeddings, timestep, positional embeddings, padding mask) up to the output of these blocks. Build and compute the GGML graph and verify the output shape. If possible, compare output values against a reference implementation.
    *   Incrementally add the implementation of the next block(s) and extend the test case to include these new layers. Repeat the process of building and computing the graph and verifying the output.
    *   Continue this iterative process until the entire UNet forward pass is implemented and tested layer by layer, culminating in testing the output of the `LastLayer`.
*   **Expected Outcome:** Each incremental test should run without errors, and the output shapes at each stage should be correct. Value verification at each layer will help pinpoint implementation issues early.

## Step 5: Implement ChromaUNet_ggml Structure and Orchestration

*   **Component:** `ChromaUNet_ggml` struct and its `forward` method.
*   **Code:**
    *   Define the `ChromaUNet_ggml` struct in [`chroma.hpp`](chroma.hpp) to contain instances of the `Approximator_ggml`, `SingleStreamBlock_ggml`, `DoubleStreamBlock_ggml`, and `LastLayer_ggml` structs.
    *   Implement the `ChromaUNet_ggml::init_params` method to call `init_params` on all nested blocks.
    *   Implement the `ChromaUNet_ggml::forward` method to orchestrate the complete forward pass of the UNet. This involves calling the `Approximator` to get modulation parameters, iterating through the `SingleStreamBlock` and `DoubleStreamBlock` layers in the correct sequence, passing the appropriate inputs (image latent, text embeddings, timestep, positional embeddings, padding mask) and modulation signals to each block's forward method, and finally calling the `LastLayer` forward method.
*   **Testing:** Test the full `ChromaUNet_ggml::forward` method with dummy inputs. Verify the final output shape. Value verification will be more challenging here without a full reference implementation, but shape correctness is a good initial check.
*   **Expected Outcome:** The `ChromaUNet_ggml` struct should correctly represent the Chroma UNet architecture, and its forward method should correctly orchestrate the computation graph.

## Step 6: Implement ChromaRunner and Weight Loading

*   **Component:** `ChromaRunner` class and its weight loading logic.
*   **Code:**
    *   Define the `ChromaRunner` class inheriting from `GGMLRunner`.
    *   Implement the constructor to initialize the `ChromaUNet_ggml` instance.
    *   Implement the `get_param_tensors` method to expose the learnable tensors of the `ChromaUNet_ggml` for loading.
    *   Implement the weight loading logic within the runner, adapting the pattern from `FluxRunner::load_from_file_and_test` to handle Chroma's specific parameter names and structure. This involves using `ModelLoader` to read the GGUF file and map the tensors to the `ChromaUNet_ggml` struct.
    *   Implement the `ChromaRunner::compute` method to manage the GGML context, build the computation graph by calling `ChromaUNet_ggml::forward`, and execute the graph.
*   **Testing:** Test the weight loading process with a Chroma GGUF file. Verify that the tensors are loaded correctly. Test the `ChromaRunner::compute` method with actual inputs to ensure the full model runs without errors.
*   **Expected Outcome:** The `ChromaRunner` should successfully load the Chroma model weights and be able to run the full UNet forward pass.

## Step 7: Integrate ChromaRunner into Inference Loop

*   **Component:** Integration of `ChromaRunner` into the main diffusion inference loop (likely in [`stable-diffusion.cpp`](stable-diffusion.cpp)).
*   **Code:**
    *   Modify the inference loop to load the Chroma model using `ChromaRunner` when the Chroma model type is selected.
    *   Adapt the input preparation within the inference loop to match the `ChromaRunner::compute` requirements, including obtaining the image latent, timestep, text embeddings, and padding mask.
    *   Call the `ChromaRunner::compute` method with the prepared inputs at each step of the diffusion process.
    *   Handle the output of the Chroma UNet (the predicted noise) within the inference loop.
*   **Testing:** Run the full diffusion process with a Chroma GGUF model and a text prompt. Verify that the process completes and generates an output image. Compare the output with a reference implementation if possible.
*   **Expected Outcome:** The `diffusion.cpp` application should be able to generate images using the integrated Chroma model.

## Step 8: Consider Sampler/Scheduler Compatibility

*   **Component:** Samplers and schedulers used in the inference loop.
*   **Code:**
    *   Research if Chroma utilizes any specific samplers or schedulers in ComfyUI.
    *   Compare with the samplers/schedulers available in [`diffusion.cpp`](stable-diffusion.cpp).
    *   Implement new samplers/schedulers or adapt existing ones if necessary to ensure compatibility with Chroma's diffusion process.
*   **Testing:** Test the inference process with the required samplers/schedulers to ensure correct image generation.
*   **Expected Outcome:** The diffusion process should run correctly with the appropriate samplers/schedulers for Chroma, producing high-quality images.