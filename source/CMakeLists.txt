set(COMMON_DEFS
    DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    STATS_DIR="${PROJECT_SOURCE_DIR}/stats"
)


add_executable(kd-tree kd_tree/kd_tree.cc)
target_include_directories(kd-tree PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(kd-tree PRIVATE ${COMMON_DEFS})
target_link_libraries(kd-tree PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(kd-tree PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# When teammates add their algorithm directories, append here:
# add_executable(hnsw  hnsw/hnsw.cc)
# add_executable(lsh   lsh/lsh.cc)
