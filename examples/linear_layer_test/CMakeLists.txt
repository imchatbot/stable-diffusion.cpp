# Define the executable for the linear layer test
# Add the source file main.c to the executable
add_executable(linear_layer_test main.cpp)

# Link the executable against the ggml library
# This ensures that the executable can use GGML functions
target_link_libraries(linear_layer_test stable-diffusion)