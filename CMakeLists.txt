cmake_minimum_required(VERSION 3.17)
project(Woad)

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/deps)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

FetchContent_Declare(obsidian GIT_REPOSITORY /home/michaelb/dev/obsidian GIT_TAG leaves)

FetchContent_MakeAvailable(obsidian)

add_library(woad woad.c)
target_link_libraries(woad PUBLIC Obsidian::Obsidian)
