cmake_minimum_required(VERSION 3.20)

project(mapper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mapper_lib
  hardware_topology/json_io.cpp
  hardware_topology/topology.cpp
  llm/model_config.cpp
  llm/taskgraph_builder.cpp
  mapping/cost_model.cpp
  mapping/graph.cpp
  mapping/mapper.cpp
  mapping/schedule_model.cpp
  mapping/strategies.cpp
  mapper/mapper.cpp
  taskflow/json.cpp
  taskflow/taskflow.cpp
  workload/json_io.cpp
  workload/workload.cpp
)

target_include_directories(mapper_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (MSVC)
  target_compile_options(mapper_lib PRIVATE /W4)
else()
  target_compile_options(mapper_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(mapper_demo main.cpp)
target_link_libraries(mapper_demo PRIVATE mapper_lib)
