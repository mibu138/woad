add_library(woad woad.c)
target_link_libraries(woad PUBLIC Obsidian::Obsidian)
target_include_directories(
    woad
    PRIVATE   ../include/woad
    INTERFACE ../include)
add_library(Woad::Woad ALIAS woad)
