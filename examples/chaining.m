// chaining.m — ANE chaining request example (EXPERIMENTAL)
// Tests _ANEChainingRequest construction and dispatch via prepareChainingWithModel:.
//
// Build: make -C .. examples
// Run:   ./chaining /path/to/model.mlmodelc

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <signal.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    signal(SIGSEGV, SIG_IGN);

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
        printf("[OK] Model prepared\n");

        // Create buffers for chaining: input → model → output_set_1 → (loopback) → output_set_2
        ANEBuffer *input = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
        ANEBuffer *out1 = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
        ANEBuffer *out2 = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];

        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];

        // Completion event
        ANEEvent *completion = [ANEEvent event];

        // Create chain request
        ANEChainRequest *chain = [ANEChainRequest
            chainWithInputs:@[input]
                 outputSets:@[@[out1], @[out2]]
       loopbackInputIndices:@[@0]
      loopbackOutputIndices:@[@0]
                signalEvent:completion
             fwEnqueueDelay:nil
               memoryPoolId:nil];

        printf("[%s] Chain request: %p\n", chain ? "OK" : "FAIL",
               chain ? chain.aneChainingRequest : nil);

        if (!chain) {
            printf("[FAIL] Could not construct chain request\n");
            return 1;
        }

        // Validate
        id internalChain = chain.aneChainingRequest;
        if ([internalChain respondsToSelector:@selector(validate)]) {
            BOOL valid = ((BOOL (*)(id, SEL))objc_msgSend)(internalChain, @selector(validate));
            printf("[%s] Validate\n", valid ? "OK" : "FAIL");
        }

        // Try prepareChain
        printf("Preparing chain...\n");
        error = nil;
        BOOL ok = [[ANEDispatch shared] prepareChain:model chainRequest:chain error:&error];
        printf("[%s] PrepareChain: %s\n", ok ? "OK" : "FAIL",
               error ? [[error description] UTF8String] : "none");

        if (ok) {
            usleep(10000);
            printf("Completion value: %llu\n", (unsigned long long)completion.signaledValue);
        }

        [model unloadWithError:nil];
        printf("Done.\n");
    }
    return 0;
}
