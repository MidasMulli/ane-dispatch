// probe5_4_pool_drain.m
// Main 59 Probe 5 variant 4: does wrapping the CPU wait in a nested
// @autoreleasepool (so its temporary ObjC references drain before the
// subsequent evaluate) clear the Main 58 Probe 4 poisoning?
//
// Hypothesis: waitUntilValue:timeoutMS: creates autoreleased objects
// (kevent wrappers, internal port references) that don't get released
// until the enclosing pool drains. If the next evaluate runs in the
// same pool, those lingering refs could taint ANE state.
//
// Test: scope the CPU wait inside a nested @autoreleasepool {}.
// After the block exits, run evaluate in the outer pool.

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

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        [model prepareWithError:&error];
        printf("Model prepared. handle=%llu\n\n",
               (unsigned long long)model.programHandle);

        // -- The poisoning wait, scoped to a nested pool --
        BOOL waitSignaled = NO;
        double wait_ms = 0.0;
        @autoreleasepool {
            ANEEvent *gpuDoneEvent = [ANEEvent event];
            gpuDoneEvent.signaledValue = 0;
            dispatch_after(
                dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC),
                dispatch_get_global_queue(0, 0), ^{
                    gpuDoneEvent.signaledValue = 42;
                });

            printf("[inside nested pool] CPU wait starting...\n");
            uint64_t t0 = mach_absolute_time();
            waitSignaled = [gpuDoneEvent waitUntilValue:42 timeoutMS:5000];
            uint64_t t1 = mach_absolute_time();
            wait_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
            printf("[inside nested pool] wait signaled=%d elapsed=%.2fms\n",
                   waitSignaled, wait_ms);
            // gpuDoneEvent falls out of scope here; pool drains on block exit.
        }
        printf("Nested pool drained. ObjC autoreleased refs released.\n\n");

        if (!waitSignaled) { printf("ABORT\n"); [model unloadWithError:nil]; return 1; }

        // -- Evaluate AFTER the drained wait --
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

        printf("Evaluating after nested pool drain...\n");
        uint64_t t2 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:freshRequest error:&error];
        uint64_t t3 = mach_absolute_time();
        double eval_ms = (double)(t3 - t2) * info.numer / info.denom / 1e6;
        printf("  eval elapsed=%.2fms completion=%llu (expect 77)\n\n",
               eval_ms, (unsigned long long)freshCompletion.signaledValue);

        if (freshCompletion.signaledValue == 77 && eval_ms > 0.5) {
            printf("PROBE 5 VARIANT 4: PASS\n");
            printf("  Autorelease pool drain clears the poisoning.\n");
            printf("  Root cause is an autoreleased ObjC reference from waitUntilValue.\n");
        } else if (eval_ms < 0.5 && freshCompletion.signaledValue == 0) {
            printf("PROBE 5 VARIANT 4: FAIL — still silent-skipping after pool drain.\n");
            printf("  Not an autoreleased-ref issue. Try variant 5 (dispatch barrier).\n");
        } else {
            printf("PROBE 5 VARIANT 4: INCONCLUSIVE — eval_ms=%.2f comp=%llu\n",
                   eval_ms, (unsigned long long)freshCompletion.signaledValue);
        }

        [[ANEDispatch shared] unmapBuffers:model request:freshRequest];
        [model unloadWithError:nil];
    }
    return 0;
}
