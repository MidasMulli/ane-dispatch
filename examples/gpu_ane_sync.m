// gpu_ane_sync.m — Minimal GPU→ANE hardware wait gate demonstration.
//
// This example completes the cross-accelerator signaling story:
//   ANE → GPU: completion signal (shared_events.m, existing)
//   GPU → ANE: wait gate      (this file, new)
//
// The ANE holds execution pending a value on an IOSurfaceSharedEvent that
// the CPU (or, via bridgeToMetalDevice:, the GPU) signals. Once the signal
// arrives, the ANE executes and signals its own completion event. Round-
// trip zero-CPU coordination between accelerators is achievable.
//
// Important quirks confirmed Main 58/59:
//   1. evaluate: is ASYNCHRONOUS. It returns in ~0.1 ms; the ANE executes
//      later. Do not read the completion event or unload the model before
//      the ANE has had time to see the gate and complete.
//   2. Poll completion.signaledValue with usleep(1000). The library's
//      waitUntilValue:timeoutMS: busy-waits (99% CPU) on ANE-signaled
//      events on this macOS version.
//   3. One gated eval per process. Mixing baseline and gated evals in a
//      single process hangs the second one. Forks/subprocesses for
//      back-to-back evals.
//   4. signal(SIGSEGV, SIG_IGN) is mandatory — aned's XPC completion
//      handler crashes after SharedEvents evaluation and the SIGSEGV
//      suppression keeps the test process alive.

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <signal.h>
#import <unistd.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN);

    @autoreleasepool {
        const char *path = (argc > 1) ? argv[1] :
            "/Users/midas/Desktop/cowork/ngram-engine/"
            "ane_reverse/mode_sweep_models/base_relu_compiled/"
            "base_relu.mlmodelc";

        NSError *error = nil;
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]]
            error:&error];
        if (![model prepareWithError:&error]) {
            fprintf(stderr, "Prepare failed: %s\n", [[error description] UTF8String]);
            return 1;
        }
        printf("Model loaded. handle=%llu\n\n",
               (unsigned long long)model.programHandle);

        ANEBuffer *input  = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,
                            0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];

        ANERequest *request = [ANERequest requestWithInputs:@[input]
                                                    outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:request error:nil];

        // Wait gate: ANE holds execution until gate.signaledValue reaches 50.
        ANEEvent *gate = [ANEEvent event];
        gate.signaledValue = 0;
        [request setWaitGate:gate value:50];

        // Completion signal: ANE sets completion.signaledValue to 99 when done.
        ANEEvent *completion = [ANEEvent event];
        completion.signaledValue = 0;
        [request setCompletionSignal:completion value:99];

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        __block uint64_t t_gate_signaled = 0;

        // Simulated GPU work: background thread signals gate → 50 after 80 ms.
        // In production, this would be a Metal command-buffer completion
        // signaling a MTLSharedEvent bridged via [event bridgeToMetalDevice:].
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 80 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                gate.signaledValue = 50;
                t_gate_signaled = mach_absolute_time();
            });

        printf("═══ GPU→ANE wait gate (ANE waits until gate≥50) ═══\n");
        printf("Dispatching eval (async — returns before ANE executes)...\n");
        uint64_t t0 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:request error:&error];
        uint64_t t_ret = mach_absolute_time();
        double dispatch_ms =
            (double)(t_ret - t0) * info.numer / info.denom / 1e6;
        printf("  evaluate returned in %.2f ms (ANE still waiting on gate)\n",
               dispatch_ms);
        printf("  immediate completion read: %llu (expect 0)\n",
               (unsigned long long)completion.signaledValue);

        printf("\nPolling completion.signaledValue (1 ms ticks)...\n");
        BOOL ok = NO;
        uint64_t t_completion = 0;
        int tick = 0;
        for (; tick < 500; tick++) {       // up to 500 ms
            usleep(1000);
            if (completion.signaledValue == 99) {
                ok = YES;
                t_completion = mach_absolute_time();
                break;
            }
        }
        uint64_t t1 = mach_absolute_time();
        double total_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        double gate_ms = t_gate_signaled ? (double)(t_gate_signaled - t0) * info.numer / info.denom / 1e6 : -1.0;
        double completion_ms = t_completion ? (double)(t_completion - t0) * info.numer / info.denom / 1e6 : -1.0;
        printf("  t_gate_signaled = %.2f ms after eval-start\n", gate_ms);
        printf("  t_completion_99 = %.2f ms after eval-start\n", completion_ms);
        if (gate_ms > 0 && completion_ms > 0) {
            double delta = completion_ms - gate_ms;
            printf("  completion came %.2f ms after gate signal — %s\n",
                   delta,
                   (delta >= 0) ? "consistent with gate holding ANE"
                                : "⚠ BEFORE gate signal — gate is NOT holding the ANE");
        }

        uint16_t result[8] = {0};
        [output readFloat16:result count:8];

        printf("  completion reached 99 at ~%d ms\n", tick);
        printf("  output[0]=0x%04x (0x3C00 = ANE wrote relu(1.0)=1.0)\n",
               result[0]);
        printf("  end-to-end=%.2f ms (gate delay + ANE execute + settle)\n\n",
               total_ms);

        double gate_to_comp_ms = (gate_ms > 0 && completion_ms > 0) ?
            (completion_ms - gate_ms) : -999.0;
        if (ok && result[0] == 0x3C00 &&
            gate_ms >= 70.0 && gate_to_comp_ms >= -0.5 && gate_to_comp_ms < 20.0) {
            printf("✓ GPU→ANE hardware wait gate WORKS via IOSurfaceSharedEvent.\n");
            printf("  ANE held execution ~%.1f ms waiting for the gate, then\n",
                   gate_ms);
            printf("  executed within %.2f ms of the gate signal and signaled\n",
                   gate_to_comp_ms);
            printf("  completion=99. Round-trip\n");
            printf("  accelerator-to-accelerator sync without CPU polling is\n");
            printf("  achievable (the polling here is just so this demo can\n");
            printf("  observe completion — in production, the GPU's command\n");
            printf("  buffer would observe the bridged MTLSharedEvent directly).\n");
        } else {
            printf("✗ Gate or completion did not behave as expected.\n");
            printf("  ok=%d output=0x%04x tick=%d\n", ok, result[0], tick);
        }

        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];
    }
    return 0;
}
