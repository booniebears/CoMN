cmake_minimum_required(VERSION 3.10)
project(CoMN_Refactor)

set(CMAKE_CXX_STANDARD 17)

# file(GLOB_RECURSE SOURCES
#     "*.cpp"
# )

# file(GLOB_RECURSE DIR_SRC "src/*.cpp")

# # list(FILTER SOURCES EXCLUDE REGEX "^.*build/.*$")

# add_executable(main ${SOURCES})

# 编译当前目录下的所有.cpp文件
file(GLOB SOURCES *.cpp)

# 添加编译源文件
add_executable(main ${SOURCES})

# 添加 ./CoMN_Sim/src 目录下的所有.cpp文件
file(GLOB_RECURSE COMN_SIM_SOURCES ./CoMN_Sim/src/*.cpp)

# 将 CoMN_Sim 源文件添加到可执行文件中
target_sources(main PRIVATE ${COMN_SIM_SOURCES})