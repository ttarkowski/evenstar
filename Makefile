SHELL      = /bin/bash
CXX        = g++-10
CXXFLAGS   = -Wall -Wextra -pedantic -O3 -std=c++20 -fconcepts -g -ggdb -I../
LDFLAGS    = -L../libbear/ -lbear
TARGET     = evenstar
SOURCES    = $(shell echo main.cc src/*.cc)
OBJECTS    = $(SOURCES:.cc=.o)
DEPENDENCY = $(OBJECTS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

-include $(DEPENDENCY)

%.o : %.cc
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Example:
#   make ARGS="3 f" run
run:
	LD_LIBRARY_PATH=../libbear/:$LD_LIBRARY_PATH ./$(TARGET) $(ARGS)

check:
	LD_LIBRARY_PATH=../libbear/:$LD_LIBRARY_PATH \
	valgrind --leak-check=full --show-leak-kinds=all \
	--errors-for-leak-kinds=all --run-cxx-freeres=yes \
	./$(TARGET) $(ARGS)

.PHONY: clean
clean:
	rm -f *~ */*~ $(TARGET) $(OBJECTS) $(DEPENDENCY)
