CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -Wall -Wextra
LDFLAGS  = -fopenmp

BIN = nndescent smerge lsh_merge make_qgt main
BIN_DIR = bin

BIN_FILES = $(addprefix $(BIN_DIR)/, $(BIN))

all: $(BIN_DIR) $(BIN_FILES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(BIN_DIR)/$@ $(LDFLAGS)


clean:
	rm -f $(BIN)
