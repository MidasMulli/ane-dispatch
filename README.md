# ane-dispatch

Direct Neural Engine programming for Apple Silicon via the private `_ANEClient` dispatch API.

A reusable open library for ANE dispatch with cross-accelerator synchronization (SharedEvents).

> **Scope note.** This library uses a faster *private* dispatch entry point into Apple's ANE stack; it does **not** bypass CoreML's underlying daemon. `doEvaluateDirectWithModel:` is a private method on `_ANEClient` (the ANEServices XPC proxy), so dispatch still flows through `aned`, the system ANE daemon. There is no user-process path that talks to the ANE hardware without `aned`. What this library skips is the higher-level CoreML *runtime* wrapper, not the daemon, and not "CoreML entirely."

## What this does

`ane-dispatch` gives you direct access to the Apple Neural Engine via the private `_ANEClient` API, skipping the higher-level CoreML runtime wrapper:

- **Lower-overhead dispatch** via the private `doEvaluateDirectWithModel:` entry point instead of the CoreML runtime's `evaluateWithModel:`. (A ~37% per-dispatch latency reduction was observed in informal testing; this figure is not yet in the measurement registry and should be treated as indicative, not a benchmarked guarantee.)
- **SharedEvents signal-back**: ANE signals an IOSurfaceSharedEvent on completion. (We are not aware of another open implementation of functional ANE SharedEvents; see Prior art for the verification status of this claim.)
- **Metal bridge**: ANE completion events are bridgeable to MTLSharedEvent for GPU↔ANE synchronization
- **Zero-copy IOSurface I/O**: direct buffer sharing between CPU, GPU, and ANE
- **Chaining API** (experimental): pipelined multi-step execution with firmware-level enqueue delay

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

// Evaluate (direct _ANEClient dispatch, skips the CoreML runtime wrapper)
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

// Attach to request: ANE will signal 42 after execution
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
ANE SharedEvents cause a background SIGSEGV in `aned`'s XPC completion handler after evaluation. This does **not** affect execution results or signal values: the eval succeeds and the completion signal is correct. However, subsequent evaluations with SharedEvents through the same connection may hang.

**Workaround:** `signal(SIGSEGV, SIG_IGN)` suppresses the background crash. Use one-shot SharedEvents (one eval with events per process), or fork for multiple tests.

### Wait events
- **Pre-signaled waits work**: if the event value already meets the threshold at eval time, ANE proceeds and signals back.
- **Cross-accelerator waits (GPU → ANE) are blocked**: Metal-bridged events with unmet thresholds cause the eval to block indefinitely. The kernel wait mechanism exists but GPU → ANE signal routing is unresolved.
- **Standalone events with unmet thresholds are skipped**: ANE executes but skips all events (no signal-back either).

### Chaining
`ANEChainRequest` constructs successfully but `validate` crashes with `symbolIndex`: the chaining API requires IOSurfaces created through the model's internal mapper (`_ANEProgramIOSurfacesMapper`), not standalone buffers. Chaining with standalone `ANEBuffer` objects is not currently supported. Investigation needed into how `symbolIndex` maps to model I/O ports.

### MIL compilation
The `_ANEInMemoryModelDescriptor` API exists but the MIL text format for direct compilation is undocumented. Currently, models must be pre-compiled via `coremltools` (Python). The library accepts compiled `.mlmodelc` directories.

### One-shot SharedEvents
After an evaluation with SharedEvents, the `aned` XPC connection is corrupted by the background crash. Subsequent evaluations with events on the same connection may hang. Create fresh `ANERequest` objects for each SharedEvents evaluation, or use one event-bearing eval per process.

## Requirements

- macOS 26+ (Apple Silicon with ANE)
- Xcode Command Line Tools
- A compiled CoreML model (`.mlmodelc`)

**Tested on:** MacBook Pro M5 Pro, 64 GB, macOS 26.0 (build 26A5840e).

## Architecture

```
Your code
    ↓
ANEDispatch (this library)
    ↓ _ANEClient private API (ANEServices XPC proxy)
aned (system ANE daemon, root)
    ↓ IOKit / H11ANE
ANE hardware (16 tiles, ~111 GB/s dedicated DMA)
```

The CoreML *runtime wrapper* is not in the path: the library communicates with `aned` directly via `_ANEClient`'s XPC interface, which is the same private layer CoreML itself dispatches through. `aned` remains in the path; it exclusively owns the IOKit connection to the hardware, and there is no user-process route around it. (DMA figure: `ane.dma_bandwidth` = 111 GB/s, measurement registry, Main 14, status canonical, VERIFIED_IOREPORT.)

## Prior art

- **maderix** ([github.com/maderix/ANE](https://github.com/maderix/ANE)): ANE characterization, in-memory compilation, IOSurface I/O. Listed SharedEvents as "unexplored."
- **Orion** (arXiv 2603.06728): Direct `_ANEClient` dispatch for training. Did not use SharedEvents.
- **ane-toolkit** ([github.com/MidasMulli/ane-toolkit](https://github.com/MidasMulli/ane-toolkit)): ANE binary format (H17), PWL activation deployment.

As far as we are aware, `ane-dispatch` is the first open library to implement *functional* ANE SharedEvents and provide a reusable `_ANEClient`-direct dispatch API. maderix and Orion both used direct `_ANEClient` dispatch but neither implemented SharedEvents. This primacy claim is supported by an internal prior-art audit (status: verified) against maderix and Orion; it has **not** been re-audited against ANEMLL or other ecosystem projects that may have appeared since. Treat the "first" framing as best-effort, not an exhaustively verified primacy assertion.

## Research findings

This library is built on reverse engineering work that discovered:

1. **ANE SharedEvents signal-back works**: `aned` firmware honors `_ANESharedSignalEvent`, writing the specified uint64 value to the attached `IOSurfaceSharedEvent` after model execution. Confirmed across 5+ independent tests with different signal values.

2. **`IOSurfaceSharedEvent` bridges bidirectionally with `MTLSharedEvent`**: via Mach port extraction from `MTLSharedEventHandle`. GPU can signal ANE and (with caveats) ANE can signal GPU.

3. **`doEvaluateDirectWithModel:` is lower-overhead than `evaluateWithModel:`**: it is a private `_ANEClient` entry point that skips the CoreML runtime wrapper's dispatch path while still going through `aned`. A ~37% per-dispatch latency reduction was observed in informal testing (not yet in the measurement registry; indicative, not a benchmarked guarantee).

4. **`_ANEChainingRequest` supports firmware-level enqueue delay** (`fwEnqueueDelay`) and output-to-input loopback (`lbInputSymbolId`/`lbOutputSymbolId`). We are not aware of prior work exploring this, though it has not been exhaustively audited.

## License

MIT: see [LICENSE](LICENSE).
