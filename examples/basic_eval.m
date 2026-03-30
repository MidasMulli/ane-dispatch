// basic_eval.m — Minimal ane-dispatch example
// Loads a compiled CoreML model, evaluates on ANE, reads output.
//
// Build: make -C .. examples
// Run:   ./basic_eval /path/to/model.mlmodelc

#import <Foundation/Foundation.h>
#import "ANEDispatch.h"

int main(int argc, char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <path/to/model.mlmodelc>\n", argv[0]);
            return 1;
        }

        NSString *modelPath = [NSString stringWithUTF8String:argv[1]];
        NSError *error = nil;

        // 1. Load model
        ANEModel *model = [ANEModel modelWithCompiledURL:[NSURL fileURLWithPath:modelPath]
                                                   error:&error];
        if (!model) {
            fprintf(stderr, "Failed to load model: %s\n", [[error description] UTF8String]);
            return 1;
        }

        // 2. Compile + load in one step
        if (![model prepareWithError:&error]) {
            fprintf(stderr, "Failed to prepare model: %s\n", [[error description] UTF8String]);
            return 1;
        }
        printf("Model loaded. programHandle: %llu\n", (unsigned long long)model.programHandle);

        // 3. Create I/O buffers (adjust shape to match your model)
        ANEBuffer *input = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1, @8, @1, @1] dtype:ANEDtypeFloat16];

        // 4. Fill input with 1.0 (FP16 = 0x3C00)
        uint16_t ones[8] = {0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00};
        [input fillFloat16:ones count:8];

        // 5. Create request + map buffers
        ANERequest *request = [ANERequest requestWithInputs:@[input] outputs:@[output]];
        [[ANEDispatch shared] mapBuffers:model request:request error:&error];

        // 6. Evaluate (direct dispatch, no CoreML)
        if (![[ANEDispatch shared] evaluate:model request:request error:&error]) {
            fprintf(stderr, "Evaluation failed: %s\n", [[error description] UTF8String]);
            return 1;
        }

        // 7. Read output
        uint16_t result[8];
        [output readFloat16:result count:8];
        printf("Output: ");
        for (int i = 0; i < 8; i++) printf("0x%04x ", result[i]);
        printf("\n");

        // 8. Cleanup
        [[ANEDispatch shared] unmapBuffers:model request:request];
        [model unloadWithError:nil];

        printf("Done.\n");
    }
    return 0;
}
