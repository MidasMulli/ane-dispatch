// probe1_v2_wait_with_settle.m
// Main 59 re-test of M58 Probe 1 with correct async-aware timing.
//
// Original Probe 1 read completion immediately after evaluate returned (0.1 ms)
// and then unloaded the model. If ANE eval is async (Probe 7 confirmed), the
// original Probe 1's "silent skip" reading may have been premature: the ANE
// could have been sitting waiting for the gate signal, which arrived at
// t=100 ms — AFTER the model had been unloaded.
//
// This v2 holds the model alive through the gate signal and adds settle time,
// then reads the completion. If completion reaches 99, the GPU→ANE hardware
// wait gate is NOT a dead path; it works and the original diagnosis was wrong.

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <signal.h>
#import <unistd.h>
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

        ANEEvent *gate = [ANEEvent event];
        gate.signaledValue = 0;
        [request setWaitGate:gate value:50];

        ANEEvent *completion = [ANEEvent event];
        completion.signaledValue = 0;
        [request setCompletionSignal:completion value:99];

        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                printf("  [+100ms] Signaling gate to 50...\n");
                gate.signaledValue = 50;
            });

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        printf("Dispatching ANE eval with unmet wait gate...\n");
        uint64_t t0 = mach_absolute_time();
        BOOL ok = [[ANEDispatch shared] evaluate:model request:request
                                           error:&error];
        uint64_t t1 = mach_absolute_time();
        double eval_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        printf("  evaluate returned: ok=%d elapsed=%.2fms (async dispatch)\n",
               ok, eval_ms);
        printf("  completion immediately after return: %llu (expect 0 because ANE is waiting)\n",
               (unsigned long long)completion.signaledValue);

        // Hold the model alive, let the background signal fire AND let ANE
        // complete after the gate clears. 300 ms is generous: gate fires at
        // t=100 ms, ANE should complete within a few ms after that.
        printf("\nSleeping 300 ms to let gate signal fire and ANE complete...\n");
        useconds_t checkpoints[] = {50000, 100000, 150000, 200000, 300000};
        useconds_t slept = 0;
        for (int i = 0; i < 5; i++) {
            useconds_t d = checkpoints[i] - slept;
            usleep(d);
            slept = checkpoints[i];
            printf("  t=%u us: gate=%llu completion=%llu\n", slept,
                   (unsigned long long)gate.signaledValue,
                   (unsigned long long)completion.signaledValue);
            if (completion.signaledValue == 99) break;
        }

        // Final read + verdict
        uint16_t result[8] = {0};
        [output readFloat16:result count:8];
        printf("\nFinal output[0]: 0x%04x (0x3C00 = ANE wrote relu(1.0)=1.0)\n",
               result[0]);
        printf("Final completion: %llu (expect 99)\n",
               (unsigned long long)completion.signaledValue);

        if (completion.signaledValue == 99 && result[0] == 0x3C00) {
            printf("\nPROBE 1 v2: PASS\n");
            printf("  GPU→ANE hardware wait gate via SharedEvents WORKS.\n");
            printf("  M58 DEAD_PATH verdict was based on an async-timing artifact —\n");
            printf("  the original Probe 1 unloaded the model before the gate signal\n");
            printf("  arrived. With the model held alive and proper settle time,\n");
            printf("  the ANE correctly waits for the gate and then executes.\n");
        } else if (result[0] == 0x3C00 && completion.signaledValue != 99) {
            printf("\nPROBE 1 v2: PARTIAL\n");
            printf("  ANE executed (output written) but completion never fired.\n");
        } else {
            printf("\nPROBE 1 v2: FAIL\n");
            printf("  ANE did not execute even with 300 ms settle. Original M58 DEAD\n");
            printf("  PATH verdict stands for the hardware wait gate, but the\n");
            printf("  evidence is now stronger (model held alive, full settle).\n");
        }

        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];
    }
    return 0;
}
