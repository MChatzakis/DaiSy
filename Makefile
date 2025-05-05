# Compiler and flags
CXX = g++
CXXFLAGS = -Ilib -Icommons -march=native -mavx -mavx2 -msse3 -O2 -g -fopenmp -Wuninitialized -Wunused-variable -Wparentheses

# Directories
SRC_DIR = lib
COMMONS_DIR = commons
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
LIB_DIR = $(BUILD_DIR)/lib
DEMO_DIR = demos
BIN_DIR = $(BUILD_DIR)/bin

# Library name
LIB_NAME = libdiNoLib.a

# Find all .cpp source files in lib, commons and demos
LIB_SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
COMMONS_SRCS = $(shell find $(COMMONS_DIR) -name "*.cpp")
DEMO_SRCS = $(shell find $(DEMO_DIR) -name "*.cpp")

# Convert source file paths to object file paths
LIB_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))
COMMONS_OBJS = $(patsubst $(COMMONS_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMONS_SRCS))
DEMO_OBJS = $(patsubst $(DEMO_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(DEMO_SRCS))

# Demo executables
DEMOS = $(patsubst $(DEMO_DIR)/%.cpp,$(BIN_DIR)/%,$(DEMO_SRCS))

# Default target
all: $(LIB_DIR)/$(LIB_NAME) $(DEMOS)

# Rule to build the static library
$(LIB_DIR)/$(LIB_NAME): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# Rule to compile .cpp to .o for library
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cpp to .o for commons
$(OBJ_DIR)/%.o: $(COMMONS_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile .cpp to .o for demos
$(OBJ_DIR)/%.o: $(DEMO_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to link demo executables (link demo object + commons objects + library)
$(BIN_DIR)/%: $(OBJ_DIR)/%.o $(COMMONS_OBJS) $(LIB_DIR)/$(LIB_NAME)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -L$(LIB_DIR) -ldiNoLib -o $@

# Clean target
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean