// probe4_fresh_event.m
// Main 58/59 follow-up: isolates whether the Probe 3 "eval silent-skip after
// CPU wait" anomaly is event-reuse contamination or interaction-with-wait.
//
// Hypothesis: signaling a non-dispatch ANEEvent before an evaluate leaves the
// aned state with a stale reference to that event, which then taints the
// subsequent evaluate even though no wait gate is set on the request.
//
// Test: Same sequence as Probe 3, but after the CPU wait clears, DROP the
// gpuDoneEvent, build a FRESH ANERequest + completion ANEEvent + fresh mapped
// buffers, and evaluate that. If the eval now completes normally (completion
// reaches 77, elapsed >> 0.1ms), the Probe 3 anomaly is event-reuse.
// If it still silent-skips, the anomaly is ANE state interaction with the
// act of having waited, not event-reuse.

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <signal.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN);

    @autoreleasepool {
        NSString *modelPath = @"/Users/midas/Desktop/cowork/ngram-engine/"
                               "ane_reverse/mode_sweep_models/base_relu_compiled/"
                               "base_relu.mlmodelc";
        NSError *error = nil;
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        [model prepareWithError:&error];

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        // -- Stage 1: CPU-mediated wait using a throwaway event --
        ANEEvent *gpuDoneEvent = [ANEEvent event];
        gpuDoneEvent.signaledValue = 0;
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                gpuDoneEvent.signaledValue = 42;
            });

        printf("CPU waiting for simulated GPU signal (expect ~50ms)...\n");
        uint64_t t0 = mach_absolute_time();
        BOOL signaled = [gpuDoneEvent waitUntilValue:42 timeoutMS:5000];
        uint64_t t1 = mach_absolute_time();
        double wait_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        printf("CPU wait: signaled=%d elapsed=%.2fms\n", signaled, wait_ms);
        if (!signaled) {
            printf("PROBE 4: ABORT — CPU wait failed.\n");
            [model unloadWithError:nil];
            return 1;
        }

        // -- Drop gpuDoneEvent reference and build a FRESH request --
        gpuDoneEvent = nil;

        ANEBuffer *input  = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,
                            0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];

        ANERequest *freshRequest = [ANERequest requestWithInputs:@[input]
                                                         outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:freshRequest error:nil];

        ANEEvent *freshCompletion = [ANEEvent event];
        freshCompletion.signaledValue = 0;
        [freshRequest setCompletionSignal:freshCompletion value:77];

        // -- Stage 2: Evaluate with fresh request/event, after the wait --
        printf("\nDispatching ANE with FRESH request + completion event...\n");
        uint64_t t2 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:freshRequest error:&error];
        uint64_t t3 = mach_absolute_time();
        double eval_ms = (double)(t3 - t2) * info.numer / info.denom / 1e6;
        printf("ANE eval: %.2fms completion=%llu (expect 77)\n",
               eval_ms, (unsigned long long)freshCompletion.signaledValue);

        if (freshCompletion.signaledValue == 77 && eval_ms > 0.5) {
            printf("\nPROBE 4: PASS\n");
            printf("  Fresh request + fresh event after CPU wait evaluates normally.\n");
            printf("  Probe 3 silent-skip is CAUSED BY EVENT REUSE of gpuDoneEvent.\n");
            printf("  Root cause: signaling a non-dispatch ANEEvent before an evaluate\n");
            printf("  taints subsequent evals that share no wait gate with that event.\n");
        } else if (eval_ms < 0.5 && freshCompletion.signaledValue == 0) {
            printf("\nPROBE 4: FAIL — still silent-skipping.\n");
            printf("  Fresh request + fresh event does NOT fix the anomaly.\n");
            printf("  Root cause: ANE state is tainted by the CPU wait itself,\n");
            printf("  independent of event identity. Deeper investigation needed.\n");
        } else {
            printf("\nPROBE 4: INCONCLUSIVE\n");
            printf("  eval_ms=%.2f comp=%llu — unexpected combination.\n",
                   eval_ms, (unsigned long long)freshCompletion.signaledValue);
        }

        [[ANEDispatch shared] unmapBuffers:model request:freshRequest];
        [model unloadWithError:nil];
        printf("\nModel unloaded.\n");
    }
    return 0;
}
