# ane-dispatch

Direct Neural Engine programming for Apple Silicon. Bypass CoreML entirely.

First open library for ANE dispatch with cross-accelerator synchronization (SharedEvents).

## What this does

`ane-dispatch` gives you direct access to the Apple Neural Engine without CoreML overhead:

- **37% faster dispatch** via `doEvaluateDirectWithModel:` (0.12ms vs 0.19ms CoreML)
- **SharedEvents signal-back** — ANE signals an IOSurfaceSharedEvent on completion (first external implementation)
- **Metal bridge** — ANE completion events are bridgeable to MTLSharedEvent for GPU↔ANE synchronization
- **Zero-copy IOSurface I/O** — direct buffer sharing between CPU, GPU, and ANE
- **Chaining API** (experimental) — pipelined multi-step execution with firmware-level enqueue delay

## Quick start

```bash
make
./examples/basic_eval /path/to/model.mlmodelc
./examples/shared_events /path/to/model.mlmodelc
```

## Usage

```objc
#import "ANEDispatch.h"

// Load a compiled CoreML model
ANEModel *model = [ANEModel modelWithCompiledURL:modelURL error:&error];
[model prepareWithError:&error];  // compile + load

// Create I/O buffers
ANEBuffer *input = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
[input fillFloat16:data count:8];

// Create request and map buffers
ANERequest *request = [ANERequest requestWithInputs:@[input] outputs:@[output]];
[[ANEDispatch shared] mapBuffers:model request:request error:nil];

// Evaluate (direct dispatch, no CoreML)
[[ANEDispatch shared] evaluate:model request:request error:nil];

// Read output
uint16_t result[8];
[output readFloat16:result count:8];
```

### SharedEvents (completion signaling)

```objc
// Create completion event
ANEEvent *completion = [ANEEvent event];
completion.signaledValue = 0;

// Attach to request — ANE will signal 42 after execution
[request setCompletionSignal:completion value:42];

[[ANEDispatch shared] evaluate:model request:request error:nil];
// completion.signaledValue == 42 ✓

// Bridge to Metal for GPU observation
id<MTLSharedEvent> metalEvent = [completion bridgeToMetalDevice:device];
// GPU command buffer can encodeWaitForEvent:metalEvent value:42
```

### Wait gate (pre-signaled)

```objc
ANEEvent *gate = [ANEEvent event];
gate.signaledValue = 50;  // Already at threshold

[request setWaitGate:gate value:50];      // ANE checks: 50 >= 50 → proceed
[request setCompletionSignal:comp value:100];

[[ANEDispatch shared] evaluate:model request:request error:nil];
// comp.signaledValue == 100 ✓
```

### Chaining (experimental)

```objc
ANEChainRequest *chain = [ANEChainRequest
    chainWithInputs:@[input]
         outputSets:@[@[out1], @[out2]]
    loopbackInputIndices:@[@0]
    loopbackOutputIndices:@[@0]
         signalEvent:completion
      fwEnqueueDelay:@(100)    // Firmware delay (units TBD)
        memoryPoolId:nil];

[[ANEDispatch shared] prepareChain:model chainRequest:chain error:nil];
```

## API

| Class | Purpose |
|-------|---------|
| `ANEDispatch` | Singleton dispatch interface (wraps `_ANEClient`) |
| `ANEModel` | Model lifecycle: compile → load → evaluate → unload |
| `ANEBuffer` | Zero-copy IOSurface wrapper with shape/dtype |
| `ANERequest` | Evaluation request with optional SharedEvents |
| `ANEEvent` | Cross-accelerator event (IOSurfaceSharedEvent ↔ MTLSharedEvent bridge) |
| `ANEChainRequest` | Pipelined execution with firmware enqueue delay (experimental) |

## Known issues

### SharedEvents XPC crash
ANE SharedEvents cause a background SIGSEGV in `aned`'s XPC completion handler after evaluation. This does **not** affect execution results or signal values — the eval succeeds and the completion signal is correct. However, subsequent evaluations with SharedEvents through the same connection may hang.

**Workaround:** `signal(SIGSEGV, SIG_IGN)` suppresses the background crash. Use one-shot SharedEvents (one eval with events per process), or fork for multiple tests.

### Wait events
- **Pre-signaled waits work** — if the event value already meets the threshold at eval time, ANE proceeds and signals back.
- **Cross-accelerator waits (GPU → ANE) are blocked** — Metal-bridged events with unmet thresholds cause the eval to block indefinitely. The kernel wait mechanism exists but GPU → ANE signal routing is unresolved.
- **Standalone events with unmet thresholds are skipped** — ANE executes but skips all events (no signal-back either).

### MIL compilation
The `_ANEInMemoryModelDescriptor` API exists but the MIL text format for direct compilation is undocumented. Currently, models must be pre-compiled via `coremltools` (Python). The library accepts compiled `.mlmodelc` directories.

## Requirements

- macOS 26+ (Apple Silicon with ANE)
- Xcode Command Line Tools
- A compiled CoreML model (`.mlmodelc`)

## Architecture

```
Your code
    ↓
ANEDispatch (this library)
    ↓ _ANEClient private API
aned (firmware daemon)
    ↓ IOKit / H11ANE
ANE hardware (16 cores, ~111 GB/s DMA)
```

CoreML is not in the path. The library communicates directly with `aned` via `_ANEClient`'s XPC interface.

## Prior art

- **maderix** ([github.com/maderix/ANE](https://github.com/maderix/ANE)) — ANE characterization, in-memory compilation, IOSurface I/O. Listed SharedEvents as "unexplored."
- **Orion** (arXiv 2603.06728) — Direct `_ANEClient` dispatch for training. Did not use SharedEvents.
- **ane-toolkit** ([github.com/MidasMulli/ane-toolkit](https://github.com/MidasMulli/ane-toolkit)) — ANE binary format (H17), PWL activation deployment.

`ane-dispatch` is the first open library to implement functional SharedEvents and provide a reusable direct dispatch API.

## Research findings

This library is built on reverse engineering work that discovered:

1. **ANE SharedEvents signal-back works** — `aned` firmware honors `_ANESharedSignalEvent`, writing the specified uint64 value to the attached `IOSurfaceSharedEvent` after model execution. Confirmed across 5+ independent tests with different signal values.

2. **`IOSurfaceSharedEvent` bridges bidirectionally with `MTLSharedEvent`** — via Mach port extraction from `MTLSharedEventHandle`. GPU can signal ANE and (with caveats) ANE can signal GPU.

3. **`doEvaluateDirectWithModel:` is 37% faster than `evaluateWithModel:`** — and eliminates the CoreML dispatch overhead entirely.

4. **`_ANEChainingRequest` supports firmware-level enqueue delay** (`fwEnqueueDelay`) and output-to-input loopback (`lbInputSymbolId`/`lbOutputSymbolId`). Completely unexplored by any prior work.

## License

MIT — see [LICENSE](LICENSE).
