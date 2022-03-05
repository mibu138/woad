cmake_minimum_required(VERSION 3.17)
project(Woad)

include(FetchContent)

option(WOAD_SKIP_EXAMPLES OFF)

FetchContent_Declare(obsidian GIT_REPOSITORY https://github.com/mokchira/obsidian GIT_TAG woad)

FetchContent_MakeAvailable(obsidian)

# So we can use the author_shaders function
list(APPEND CMAKE_MODULE_PATH ${obsidian_SOURCE_DIR}/cmake)

add_subdirectory(src)
add_subdirectory(shaders)
if (NOT ${WOAD_SKIP_EXAMPLES})
    add_subdirectory(examples)
endif()
