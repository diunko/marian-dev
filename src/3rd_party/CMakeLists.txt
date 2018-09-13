
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(SQLiteCpp)

set(TBB_BUILD_STATIC ON CACHE BOOL "")
set(TBB_BUILD_SHARED OFF CACHE BOOL "")
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "")
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "")
set(TBB_BUILD_TESTS OFF CACHE BOOL "")
add_subdirectory(tbb)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)

include_directories(./tbb/include)
