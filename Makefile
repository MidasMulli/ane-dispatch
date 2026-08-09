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

# Main 58 probes — GPU→ANE wait gate investigation
$(EXDIR)/probe1_native_wait: $(EXDIR)/probe1_native_wait.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe2_introspect: $(EXDIR)/probe2_introspect.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe3_cpu_wait: $(EXDIR)/probe3_cpu_wait.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe4_fresh_event: $(EXDIR)/probe4_fresh_event.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe5_2_unload_reload: $(EXDIR)/probe5_2_unload_reload.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe5_4_pool_drain: $(EXDIR)/probe5_4_pool_drain.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe6_write_no_wait: $(EXDIR)/probe6_write_no_wait.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe7_async_check: $(EXDIR)/probe7_async_check.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/probe1_v2_wait_with_settle: $(EXDIR)/probe1_v2_wait_with_settle.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

$(EXDIR)/gpu_ane_sync: $(EXDIR)/gpu_ane_sync.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

# Tests
tests: $(TESTDIR)/test_basic

$(TESTDIR)/test_basic: $(TESTDIR)/test_basic.m $(LIBANE)
	$(CC) $(CFLAGS) $(FRAMEWORKS) -L. -lANEDispatch $< -o $@

clean:
	rm -f $(LIB_OBJ) $(LIBANE)
	rm -f $(EXDIR)/basic_eval $(EXDIR)/shared_events
	rm -f $(TESTDIR)/test_basic
