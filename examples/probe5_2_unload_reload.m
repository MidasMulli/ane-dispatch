// probe5_2_unload_reload.m
// Main 59 Probe 5 variant 2: does [model unloadWithError:] + re-prepare
// after a successful CPU wait clear the process-state poisoning identified
// in Main 58 Probe 4?
//
// Sequence:
//   1. Load + prepare model
//   2. CPU wait on a throwaway ANEEvent (known to poison)
//   3. Unload the model
//   4. Re-prepare the SAME model (fresh compile + load)
//   5. Build fresh request + fresh completion event
//   6. Evaluate
//
// Expected if variant 2 is the fix:
//   completion == 77, eval_ms > 0.5 ms
// Expected if still broken:
//   completion == 0,  eval_ms < 0.5 ms (silent skip matches Probe 4)

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

        // -- Stage 1: Load model + do the poisoning CPU wait --
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        if (![model prepareWithError:&error]) {
            fprintf(stderr, "initial prepare failed: %s\n",
                    [[error description] UTF8String]);
            return 1;
        }
        printf("Stage 1: model prepared, handle=%llu\n",
               (unsigned long long)model.programHandle);

        ANEEvent *gpuDoneEvent = [ANEEvent event];
        gpuDoneEvent.signaledValue = 0;
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                gpuDoneEvent.signaledValue = 42;
            });

        printf("Stage 2: CPU wait (poisoning act)...\n");
        uint64_t t0 = mach_absolute_time();
        BOOL signaled = [gpuDoneEvent waitUntilValue:42 timeoutMS:5000];
        uint64_t t1 = mach_absolute_time();
        double wait_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        printf("  wait signaled=%d elapsed=%.2fms\n", signaled, wait_ms);
        if (!signaled) {
            printf("ABORT — wait failed.\n");
            [model unloadWithError:nil];
            return 1;
        }
        gpuDoneEvent = nil;

        // -- Stage 3: Unload the model --
        printf("Stage 3: unloading model...\n");
        BOOL unloadOK = [model unloadWithError:&error];
        printf("  unload ok=%d err=%s\n", unloadOK,
               error ? [[error description] UTF8String] : "none");
        error = nil;

        // -- Stage 4: Re-prepare the SAME model (fresh compile + load) --
        printf("Stage 4: re-preparing model...\n");
        uint64_t tp0 = mach_absolute_time();
        BOOL reprepOK = [model prepareWithError:&error];
        uint64_t tp1 = mach_absolute_time();
        double prep_ms = (double)(tp1 - tp0) * info.numer / info.denom / 1e6;
        printf("  re-prepare ok=%d elapsed=%.2fms handle=%llu err=%s\n",
               reprepOK, prep_ms,
               (unsigned long long)model.programHandle,
               error ? [[error description] UTF8String] : "none");
        if (!reprepOK) {
            printf("ABORT — re-prepare failed.\n");
            return 1;
        }

        // -- Stage 5: Build fresh request + fresh completion event --
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

        // -- Stage 6: Evaluate --
        printf("Stage 6: evaluating with fresh request...\n");
        uint64_t t2 = mach_absolute_time();
        BOOL evalOK = [[ANEDispatch shared] evaluate:model
                                              request:freshRequest
                                                error:&error];
        uint64_t t3 = mach_absolute_time();
        double eval_ms = (double)(t3 - t2) * info.numer / info.denom / 1e6;
        printf("  eval ok=%d elapsed=%.2fms completion=%llu (expect 77) err=%s\n",
               evalOK, eval_ms,
               (unsigned long long)freshCompletion.signaledValue,
               error ? [[error description] UTF8String] : "none");

        printf("\n");
        if (freshCompletion.signaledValue == 77 && eval_ms > 0.5) {
            printf("PROBE 5 VARIANT 2: PASS\n");
            printf("  Unload + reload clears the CPU-wait poisoning.\n");
            printf("  Cleanup step = [model unloadWithError:] followed by re-prepare.\n");
        } else if (eval_ms < 0.5 && freshCompletion.signaledValue == 0) {
            printf("PROBE 5 VARIANT 2: FAIL\n");
            printf("  Unload + reload did NOT clear the poisoning.\n");
            printf("  The poisoning is below model lifecycle — proceed to variant 3 (connection teardown).\n");
        } else {
            printf("PROBE 5 VARIANT 2: INCONCLUSIVE\n");
            printf("  eval_ms=%.2f completion=%llu — unexpected combination.\n",
                   eval_ms, (unsigned long long)freshCompletion.signaledValue);
        }

        [[ANEDispatch shared] unmapBuffers:model request:freshRequest];
        [model unloadWithError:nil];
    }
    return 0;
}
