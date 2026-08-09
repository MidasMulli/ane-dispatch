// probe7_async_check.m
// Main 59 follow-up: baseline shared_events.m uses usleep(5000) after evaluate
// before reading the completion value ("Brief settle for background propagation").
// Our M58 Probe 3/4 and M59 Probe 5.2/5.4 do NOT usleep before reading.
//
// Hypothesis: evaluate: is asynchronous. The eval returns ~0.1 ms after
// dispatching; the ANE actually executes in the background and signals the
// completion event some time later. Reading completion.signaledValue
// immediately gives 0 because the ANE hasn't finished. The "silent skip"
// pattern (eval_ms < 0.5, completion=0) may be a timing artifact, not a real
// firmware-level drop.
//
// Test: reproduce the Probe 5.2 sequence (CPU wait, unload+reload, evaluate)
// with three increasing settle times after the final evaluate.
// If a longer settle lets completion reach 77, the M58 "poisoning" collapses
// to an async-timing issue that doesn't actually exist — just undermeasured.

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

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        // --- Same CPU-wait "poisoning" setup as Probe 3 ---
        ANEEvent *gpuDoneEvent = [ANEEvent event];
        gpuDoneEvent.signaledValue = 0;
        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                gpuDoneEvent.signaledValue = 42;
            });

        printf("CPU wait starting...\n");
        uint64_t w0 = mach_absolute_time();
        BOOL signaled = [gpuDoneEvent waitUntilValue:42 timeoutMS:5000];
        uint64_t w1 = mach_absolute_time();
        double wait_ms = (double)(w1 - w0) * info.numer / info.denom / 1e6;
        printf("  wait signaled=%d elapsed=%.2fms\n", signaled, wait_ms);
        gpuDoneEvent = nil;

        // --- Fresh request + completion ---
        ANEBuffer *input  = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,
                            0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];
        ANERequest *req = [ANERequest requestWithInputs:@[input]
                                                outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:req error:nil];
        ANEEvent *comp = [ANEEvent event];
        comp.signaledValue = 0;
        [req setCompletionSignal:comp value:77];

        printf("Dispatching ANE (async)...\n");
        uint64_t t0 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:req error:&error];
        uint64_t t1 = mach_absolute_time();
        double eval_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        printf("  evaluate returned in %.2fms, completion=%llu (immediate read)\n",
               eval_ms, (unsigned long long)comp.signaledValue);

        // --- Check completion after three progressively longer settles ---
        useconds_t settles[3] = {1000, 5000, 50000};
        for (int i = 0; i < 3; i++) {
            usleep(settles[i]);
            printf("  after usleep(%u) total-settle=%u: completion=%llu\n",
                   settles[i],
                   (i == 0 ? 1000 : (i == 1 ? 6000 : 56000)),
                   (unsigned long long)comp.signaledValue);
            if (comp.signaledValue == 77) {
                printf("\nPROBE 7: PASS — async-timing artifact confirmed.\n");
                printf("  Eval IS running normally. The 'silent skip' in Probes 3/4/5.2/5.4/6\n");
                printf("  was reading completion before ANE finished executing.\n");
                printf("  baseline shared_events.m:53 usleep(5000) was load-bearing.\n");
                [[ANEDispatch shared] unmapBuffers:model request:req];
                [model unloadWithError:nil];
                return 0;
            }
        }

        // Read output buffer too — in baseline shared_events, the output is written
        // by ANE even when completion.signaledValue never reaches target.
        uint16_t result[8] = {0};
        [output readFloat16:result count:8];
        printf("  output[0]=0x%04x (0x3C00 means ANE wrote relu(1.0)=1.0)\n",
               result[0]);

        if (result[0] == 0x3C00 && comp.signaledValue != 77) {
            printf("\nPROBE 7: PARTIAL — ANE executed (output buffer written) but\n");
            printf("  completion event never fired, even after 56ms settle.\n");
            printf("  The CPU wait specifically poisons the completion-signal path,\n");
            printf("  not the execution path.\n");
        } else if (result[0] != 0x3C00) {
            printf("\nPROBE 7: FAIL — ANE did not execute (output buffer empty).\n");
            printf("  Full poisoning confirmed. Not an async-timing artifact.\n");
        }

        [[ANEDispatch shared] unmapBuffers:model request:req];
        [model unloadWithError:nil];
    }
    return 0;
}
