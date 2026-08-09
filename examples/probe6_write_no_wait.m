// probe6_write_no_wait.m
// Main 59 Probe 6: isolate the poisoning act — is it the waitUntilValue call,
// or is it ANY user-side write to signaledValue on a non-dispatch ANEEvent?
//
// Baseline shared_events never writes to an event from user code; ANE signals it.
// Our poisoning probes all do user-side signaledValue=0 (init) then later =42.
//
// Three cases probed in sequence (fresh ANEEvent per case, no wait):
//   A. Create ANEEvent, DO NOT touch signaledValue, then evaluate.
//   B. Create ANEEvent, set signaledValue=0 only (a no-op-ish user write), evaluate.
//   C. Create ANEEvent, set signaledValue=42 (user-side write to nonzero), evaluate.
//
// Each case uses a FRESH request + fresh completion event.
// Expected:
//   A passes → the side-event existing alone doesn't poison
//   B passes → the zero-write doesn't poison
//   B fails → any user-side setSignaledValue poisons (so shared_events works only
//             because it never touches signaledValue from user code)
//   C fails → user-side write to non-zero value poisons

#import <Foundation/Foundation.h>
#import <mach/mach_time.h>
#import <signal.h>
#import "ANEDispatch.h"

static void run_case(const char *label, ANEModel *model, int variant) {
    @autoreleasepool {
        NSError *error = nil;
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);

        // Side event per the variant
        ANEEvent *sideEvent = [ANEEvent event];
        if (variant == 1) {
            sideEvent.signaledValue = 0;
        } else if (variant == 2) {
            sideEvent.signaledValue = 42;
        } // variant 0: no user write at all

        // Fresh request + completion for the evaluate
        ANEBuffer *input  = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1]
                                                dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,
                            0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];
        ANERequest *req = [ANERequest requestWithInputs:@[input] outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:req error:nil];
        ANEEvent *comp = [ANEEvent event];
        // Note: setCompletionSignal calls setSignaledValue is irrelevant —
        // the completion is written BY ANE. We don't touch comp.signaledValue
        // from user code here.
        [req setCompletionSignal:comp value:77];

        uint64_t t0 = mach_absolute_time();
        [[ANEDispatch shared] evaluate:model request:req error:&error];
        uint64_t t1 = mach_absolute_time();
        double eval_ms = (double)(t1 - t0) * info.numer / info.denom / 1e6;

        printf("%s eval_ms=%.2f completion=%llu %s\n",
               label, eval_ms, (unsigned long long)comp.signaledValue,
               (comp.signaledValue == 77 && eval_ms > 0.5) ? "PASS" : "FAIL (silent-skip)");

        [[ANEDispatch shared] unmapBuffers:model request:req];
    }
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN);

    @autoreleasepool {
        NSString *modelPath = @"/Users/midas/Desktop/cowork/ngram-engine/"
                               "ane_reverse/mode_sweep_models/base_relu_compiled/"
                               "base_relu.mlmodelc";
        NSError *error = nil;

        // NOTE: each case runs in the SAME process after the side-event write.
        // If poisoning is cumulative, only case A gives us the clean measurement.
        // So we need a fresh process per case to isolate. This single binary
        // runs case A; the user will invoke probe6_b and probe6_c variants
        // (separate binaries below would be preferred, but a single-pass here
        // is the cheapest first cut).

        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        [model prepareWithError:&error];
        printf("Model prepared. handle=%llu\n\n",
               (unsigned long long)model.programHandle);

        // The argument selects which case runs in this process.
        int variant = (argc > 1) ? atoi(argv[1]) : 0;
        const char *labels[] = {
            "[A no-user-write]",
            "[B user-write zero]",
            "[C user-write 42] ",
        };
        if (variant < 0 || variant > 2) {
            fprintf(stderr, "usage: %s [0|1|2]\n", argv[0]);
            return 2;
        }
        printf("Variant %d running: %s\n", variant, labels[variant]);

        run_case(labels[variant], model, variant);

        [model unloadWithError:nil];
    }
    return 0;
}
