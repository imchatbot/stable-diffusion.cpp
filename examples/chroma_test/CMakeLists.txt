# Define the executable for the chroma test
add_executable(chroma_test main.cpp)

# Link the executable against the stable-diffusion library
target_link_libraries(chroma_test stable-diffusion)
target_include_directories(chroma_test PUBLIC ${CMAKE_SOURCE_DIR})