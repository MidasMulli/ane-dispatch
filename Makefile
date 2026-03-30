CC = clang
CFLAGS = -O2 -fno-objc-arc -Iinclude
FRAMEWORKS = -framework Foundation -framework Metal -framework IOSurface -framework IOKit
SRCDIR = src
EXDIR = examples
TESTDIR = tests

LIB_SRC = $(SRCDIR)/ANEDispatch.m
LIB_OBJ = $(SRCDIR)/ANEDispatch.o

# Static library
LIBANE = libANEDispatch.a

.PHONY: all lib examples tests clean

all: lib examples

lib: $(LIBANE)

$(LIBANE): $(LIB_OBJ)
	ar rcs $@ $^

$(LIB_OBJ): $(LIB_SRC) include/ANEDispatch.h
	$(CC) $(CFLAGS) -c $< -o $@

# Examples
examples: $(EXDIR)/basic_eval $(EXDIR)/shared_events $(EXDIR)/chaining

$(EXDIR)/basic_eval: $(EXDIR)/basic_eval.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/shared_events: $(EXDIR)/shared_events.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/chaining: $(EXDIR)/chaining.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

# Tests
tests: $(TESTDIR)/test_basic

$(TESTDIR)/test_basic: $(TESTDIR)/test_basic.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

clean:
	rm -f $(LIB_OBJ) $(LIBANE)
	rm -f $(EXDIR)/basic_eval $(EXDIR)/shared_events
	rm -f $(TESTDIR)/test_basic
