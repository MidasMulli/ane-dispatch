// probe1_native_wait.m
// Main 58 Probe 1: Native IOSurfaceSharedEvent wait gate (non-Metal-bridged).
//
// Hypothesis: The Metal-bridged wait gate blocks indefinitely because `aned`
// cannot receive signals on a port created in the userspace process context
// via the Metal bridge. A natively created ANEEvent (not bridged from Metal)
// lives in the correct port namespace for `aned` to observe.
//
// Test: Set a native ANEEvent as the wait gate with an unmet threshold.
// Fire eval. Signal the gate from a background thread 100ms later.
// If ANE proceeds after the delay, port namespace was the issue.
//
// Expected outcome if hypothesis correct:
//   eval returns after ~100ms delay, completion value == 99
//
// Failure modes:
//   Returns immediately (<10ms): gate skipped silently — not port namespace issue
//   Hangs indefinitely (>5s):    kill and record; move to Probe 2

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <signal.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN); // Mandatory: suppress aned XPC crash

    @autoreleasepool {
        NSString *modelPath = @"/Users/midas/Desktop/cowork/ngram-engine/"
                               "ane_reverse/mode_sweep_models/base_relu_compiled/"
                               "base_relu.mlmodelc";
        NSError *error = nil;

        // Fresh model instance — never reuse across probes
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        if (![model prepareWithError:&error]) {
            fprintf(stderr, "Prepare failed: %s\n",
                    [[error description] UTF8String]);
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

        // Native wait gate — threshold NOT met at dispatch time
        ANEEvent *gate = [ANEEvent event];
        gate.signaledValue = 0;
        [request setWaitGate:gate value:50]; // ANE should wait until gate >= 50

        // Completion signal
        ANEEvent *completion = [ANEEvent event];
        completion.signaledValue = 0;
        [request setCompletionSignal:completion value:99];

        // Signal gate from background thread after 100ms
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                printf("  [+100ms] Signaling gate to 50...\n");
                gate.signaledValue = 50;
            });

        printf("Firing eval — expect ~100ms wait for gate signal...\n");
        uint64_t t0 = mach_absolute_time();
        BOOL ok = [[ANEDispatch shared] evaluate:model request:request
                                           error:&error];
        uint64_t t1 = mach_absolute_time();

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        double elapsed = (double)(t1 - t0) * info.numer / info.denom / 1e6;

        printf("\nEval returned: ok=%d elapsed=%.1fms err=%s\n",
               ok, elapsed,
               error ? [[error description] UTF8String] : "none");
        printf("Completion value: %llu (expect 99)\n\n",
               (unsigned long long)completion.signaledValue);

        if (completion.signaledValue == 99 && elapsed > 80.0) {
            printf("PROBE 1: PASS\n");
            printf("  ANE waited %.1fms for native gate signal.\n", elapsed);
            printf("  Port namespace hypothesis confirmed.\n");
        } else if (elapsed < 10.0) {
            printf("PROBE 1: FAIL — gate skipped silently (elapsed=%.1fms)\n",
                   elapsed);
            printf("  Not a port namespace issue. Move to Probe 2.\n");
        } else {
            printf("PROBE 1: INCONCLUSIVE — elapsed=%.1fms comp=%llu\n",
                   elapsed, (unsigned long long)completion.signaledValue);
        }

        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];
        printf("\nModel unloaded. Program slot released.\n");
    }
    return 0;
}
