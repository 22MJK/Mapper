CXX ?= clang++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic -I.

BUILD_DIR := build
OBJ := \
	$(BUILD_DIR)/topology.o \
	$(BUILD_DIR)/json_io.o \
	$(BUILD_DIR)/llm_model_config.o \
	$(BUILD_DIR)/llm_taskgraph_builder.o \
	$(BUILD_DIR)/graph.o \
	$(BUILD_DIR)/cost_model.o \
	$(BUILD_DIR)/mapper.o \
	$(BUILD_DIR)/schedule_model.o \
	$(BUILD_DIR)/strategies.o \
	$(BUILD_DIR)/json.o \
	$(BUILD_DIR)/taskflow.o \
	$(BUILD_DIR)/mapper_core.o \
	$(BUILD_DIR)/workload_json.o \
	$(BUILD_DIR)/workload.o \
	$(BUILD_DIR)/main.o

all: mapper_demo

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

mapper_demo: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

$(BUILD_DIR)/topology.o: hardware_topology/topology.cpp hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ hardware_topology/topology.cpp

$(BUILD_DIR)/json_io.o: hardware_topology/json_io.cpp hardware_topology/json_io.h hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ hardware_topology/json_io.cpp

$(BUILD_DIR)/llm_model_config.o: llm/model_config.cpp llm/model_config.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ llm/model_config.cpp

$(BUILD_DIR)/llm_taskgraph_builder.o: llm/taskgraph_builder.cpp llm/taskgraph_builder.h llm/model_config.h mapping/graph.h hardware_topology/topology.h taskflow/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ llm/taskgraph_builder.cpp

$(BUILD_DIR)/graph.o: mapping/graph.cpp mapping/graph.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapping/graph.cpp

$(BUILD_DIR)/cost_model.o: mapping/cost_model.cpp mapping/cost_model.h mapping/graph.h hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapping/cost_model.cpp

$(BUILD_DIR)/mapper.o: mapping/mapper.cpp mapping/mapper.h mapping/cost_model.h mapping/graph.h mapping/schedule_model.h hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapping/mapper.cpp

$(BUILD_DIR)/schedule_model.o: mapping/schedule_model.cpp mapping/schedule_model.h mapping/cost_model.h mapping/graph.h mapping/mapper.h hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapping/schedule_model.cpp

$(BUILD_DIR)/strategies.o: mapping/strategies.cpp mapping/strategies.h mapping/graph.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapping/strategies.cpp

$(BUILD_DIR)/json.o: taskflow/json.cpp taskflow/json.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ taskflow/json.cpp

$(BUILD_DIR)/taskflow.o: taskflow/taskflow.cpp taskflow/taskflow.h taskflow/json.h hardware_topology/topology.h mapping/cost_model.h mapping/graph.h mapping/mapper.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ taskflow/taskflow.cpp

$(BUILD_DIR)/mapper_core.o: mapper/mapper.cpp mapper/mapper.h llm/taskgraph_builder.h llm/model_config.h taskflow/taskflow.h mapping/cost_model.h mapping/mapper.h mapping/schedule_model.h mapping/strategies.h workload/workload.h hardware_topology/topology.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ mapper/mapper.cpp

$(BUILD_DIR)/workload.o: workload/workload.cpp workload/workload.h mapping/graph.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ workload/workload.cpp

$(BUILD_DIR)/workload_json.o: workload/json_io.cpp workload/json_io.h workload/workload.h mapping/operator_catalog.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ workload/json_io.cpp

$(BUILD_DIR)/main.o: main.cpp hardware_topology/topology.h mapper/mapper.h hardware_topology/json_io.h workload/json_io.h workload/workload.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ main.cpp

clean:
	rm -rf $(BUILD_DIR) mapper_demo

.PHONY: all clean
