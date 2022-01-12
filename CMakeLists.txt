cmake_minimum_required(VERSION 3.17)
project(Woad)

include(FetchContent)

option(WOAD_SKIP_EXAMPLES OFF)

set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/deps)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(obsidian GIT_REPOSITORY /home/michaelb/dev/obsidian GIT_TAG leaves)

FetchContent_MakeAvailable(obsidian)

# So we can use the author_shaders function
list(APPEND CMAKE_MODULE_PATH ${obsidian_SOURCE_DIR}/cmake)

add_subdirectory(source)
add_subdirectory(shaders)
if (NOT ${WOAD_SKIP_EXAMPLES})
    add_subdirectory(examples)
endif()
