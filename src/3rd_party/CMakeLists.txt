
include_directories(.)

add_subdirectory(yaml-cpp)
add_subdirectory(SQLiteCpp)

set(TBB_BUILD_STATIC ON)
set(TBB_BUILD_SHARED OFF)
set(TBB_BUILD_TBBMALLOC OFF)
set(TBB_BUILD_TBBMALLOC_PROXY OFF)
set(TBB_BUILD_TESTS OFF)
add_subdirectory(tbb)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)

include_directories(./tbb/include)
