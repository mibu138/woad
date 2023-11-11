cmake_minimum_required(VERSION 3.17)
project(Woad)

include(FetchContent)

option(WOAD_SKIP_EXAMPLES OFF)

add_subdirectory(pome)

# So we can use the author_shaders function
# list(APPEND CMAKE_MODULE_PATH ${obsidian_SOURCE_DIR}/cmake)

add_subdirectory(src)
add_subdirectory(shaders)
if (NOT ${WOAD_SKIP_EXAMPLES})
    add_subdirectory(examples)
endif()
