// shared_events.m — ANE SharedEvents completion signal example
// First external implementation of functional ANE SharedEvents.
//
// Known limitation: SharedEvents corrupt the aned XPC connection after eval.
// Subsequent evals with events may hang. Use one-shot per process, or fork.
//
// Build: make -C .. examples
// Run:   ./shared_events /path/to/model.mlmodelc

#import <Foundation/Foundation.h>
#import <signal.h>
#import <mach/mach_time.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN); // Suppress aned XPC completion handler crash

    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <path/to/model.mlmodelc>\n", argv[0]);
            return 1;
        }

        NSError *error = nil;
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
            error:&error];
        if (![model prepareWithError:&error]) {
            fprintf(stderr, "Prepare: %s\n", [[error description] UTF8String]);
            return 1;
        }
        printf("Model loaded (handle: %llu)\n\n", (unsigned long long)model.programHandle);

        // ── Signal-back demo ──
        printf("═══ ANE SharedEvents: Completion Signal ═══\n\n");

        ANEBuffer *input = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];

        ANERequest *request = [ANERequest requestWithInputs:@[input] outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:request error:nil];

        // Create completion event with target value 42
        ANEEvent *completion = [ANEEvent event];
        completion.signaledValue = 0;
        [request setCompletionSignal:completion value:42];

        // Evaluate — ANE firmware will signal 42 to the event after execution
        [[ANEDispatch shared] evaluate:model request:request error:nil];
        usleep(5000); // Brief settle for background propagation

        uint16_t result[8];
        [output readFloat16:result count:8];

        printf("  Model output[0]: 0x%04x (relu(1.0) → 1.0 → 0x3C00)\n", result[0]);
        printf("  Completion event: %llu (target: 42)\n\n",
               (unsigned long long)completion.signaledValue);

        if (completion.signaledValue == 42) {
            printf("  ✓ ANE firmware signaled completion to IOSurfaceSharedEvent.\n");
            printf("    This event is bridgeable to MTLSharedEvent for GPU↔ANE sync.\n");
            printf("    First external implementation of functional ANE SharedEvents.\n");
        } else {
            printf("  ✗ Signal-back failed. Value: %llu\n",
                   (unsigned long long)completion.signaledValue);
        }

        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];
        printf("\nDone.\n");
    }
    return 0;
}
