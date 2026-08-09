// probe3_cpu_wait.m
// Main 58 Probe 3: CPU-mediated wait as the reliable fallback when hardware
// GPU→ANE signaling is unavailable (Probes 1 & 2 confirmed firmware silently
// skips unmet wait thresholds).
//
// Measures: overhead of waitUntilValue:timeoutMS: vs expected delay.
// Expected: signaled=YES, elapsed ≈ 50ms +/- 5ms. Overhead = elapsed - 50ms.

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

        // Completion signal (standard, not wait gate)
        ANEEvent *completion = [ANEEvent event];
        completion.signaledValue = 0;
        [request setCompletionSignal:completion value:77];

        // Simulate GPU work: create a gate event and signal it from
        // background thread after 50ms
        ANEEvent *gpuDoneEvent = [ANEEvent event];
        gpuDoneEvent.signaledValue = 0;

        dispatch_after(
            dispatch_time(DISPATCH_TIME_NOW, 50 * NSEC_PER_MSEC),
            dispatch_get_global_queue(0, 0), ^{
                gpuDoneEvent.signaledValue = 42; // simulates GPU completion
            });

        // CPU-mediated wait: block until GPU signals
        printf("CPU waiting for GPU signal (expect ~50ms)...\n");
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        uint64_t t0 = mach_absolute_time();
        BOOL signaled = [gpuDoneEvent waitUntilValue:42 timeoutMS:5000];
        uint64_t t1 = mach_absolute_time();

        double wait_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;
        double overhead = wait_ms - 50.0;
        printf("CPU wait: signaled=%d elapsed=%.2fms overhead=%.2fms\n",
               signaled, wait_ms, overhead);

        if (!signaled) {
            printf("PROBE 3: FAIL — waitUntilValue timed out. Method broken.\n");
            [model unloadWithError:nil];
            return 1;
        }

        // Now dispatch ANE (CPU-mediated gate cleared)
        printf("Gate cleared. Dispatching ANE...\n");
        uint64_t t2 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:request error:&error];
        uint64_t t3 = mach_absolute_time();

        double eval_ms = (double)(t3 - t2) * info.numer / info.denom / 1e6;
        printf("ANE eval: %.2fms completion=%llu (expect 77)\n",
               eval_ms, (unsigned long long)completion.signaledValue);

        if (signaled && completion.signaledValue == 77) {
            printf("\nPROBE 3: PASS\n");
            printf("  CPU-mediated wait works. Overhead: %.2fms\n", overhead);
            printf("  Total pipeline: wait(%.2fms) + eval(%.2fms) = %.2fms\n",
                   wait_ms, eval_ms, wait_ms + eval_ms);
            if (overhead < 2.0) {
                printf("  Overhead <2ms: usable in production as fallback.\n");
            } else {
                printf("  Overhead %.2fms: above 2ms threshold, note in report.\n",
                       overhead);
            }
        } else {
            printf("\nPROBE 3: FAIL — completion not received.\n");
        }

        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];
    }
    return 0;
}
