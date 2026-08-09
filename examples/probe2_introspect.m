// probe2_introspect.m
// Main 58 Probe 2: Runtime introspection of the internal request state after
// setWaitGate:value: is called. Replaces the interactive LLDB approach in the
// directive with a scripted introspection that prints the same answers:
//   - waitEvents count on _ANESharedEvents
//   - Mach port value on the bridged/native event
//   - Whether the wait gate actually reaches the internal request object
//
// Invoked after Probe 1 FAIL_SILENT to determine whether the silent skip is:
//   (a) our code not populating waitEvents (waitEvents count == 0)
//   (b) null Mach port (bridging broken)
//   (c) everything populated correctly but firmware silently skips

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <signal.h>
#import "ANEDispatch.h"

static void dump_array(const char *label, id arr) {
    if (!arr) { printf("  %s: NIL\n", label); return; }
    if (![arr isKindOfClass:[NSArray class]]) {
        printf("  %s: not an NSArray, class=%s\n", label, class_getName([arr class]));
        return;
    }
    printf("  %s: count=%lu\n", label, (unsigned long)[arr count]);
    NSUInteger i = 0;
    for (id x in arr) {
        printf("    [%lu] class=%s  repr=%s\n",
               (unsigned long)i++,
               class_getName([x class]),
               [[x description] UTF8String]);
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
        ANEModel *model = [ANEModel modelWithCompiledURL:
            [NSURL fileURLWithPath:modelPath] error:&error];
        [model prepareWithError:&error];

        ANEBuffer *input  = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1] dtype:ANEDtypeFloat16];
        ANEBuffer *output = [ANEBuffer bufferWithShape:@[@1,@8,@1,@1] dtype:ANEDtypeFloat16];
        uint16_t ones[8] = {0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00,0x3C00};
        [input fillFloat16:ones count:8];

        ANERequest *request = [ANERequest requestWithInputs:@[input] outputs:@[output]];

        ANEEvent *gate = [ANEEvent event];
        gate.signaledValue = 0;

        printf("=== BEFORE setWaitGate ===\n");
        id beforeReq = request.aneRequest;
        printf("  aneRequest class: %s\n", class_getName([beforeReq class]));
        id beforeShared = ((id (*)(id, SEL))objc_msgSend)(
            beforeReq, NSSelectorFromString(@"sharedEvents"));
        printf("  sharedEvents: %s\n", beforeShared ? "present" : "NIL (expected)");

        // Does setWaitGate:value: exist on the wrapper?
        SEL setWaitGateSel = @selector(setWaitGate:value:);
        printf("  wrapper respondsToSelector setWaitGate:value: = %d\n",
               [request respondsToSelector:setWaitGateSel]);

        // What selectors does the internal request respond to?
        NSArray *candidateSels = @[
            @"setWaitGate:value:",
            @"setWaitEvent:value:",
            @"addWaitEvent:value:",
            @"setInputEvent:value:",
            @"setPrerequisiteEvent:value:",
            @"setSharedEvents:",
            @"sharedEvents",
            @"setWaitEvents:",
        ];
        printf("  Internal request selector probe:\n");
        for (NSString *s in candidateSels) {
            SEL sel = NSSelectorFromString(s);
            printf("    %-30s = %d\n", [s UTF8String],
                   [beforeReq respondsToSelector:sel]);
        }

        printf("\n=== Calling setWaitGate:value:50 ===\n");
        [request setWaitGate:gate value:50];

        printf("\n=== AFTER setWaitGate ===\n");
        id afterReq = request.aneRequest;
        id sharedEvents = ((id (*)(id, SEL))objc_msgSend)(
            afterReq, NSSelectorFromString(@"sharedEvents"));
        printf("  sharedEvents: %s\n", sharedEvents ? "present" : "NIL (BUG if NIL)");

        if (sharedEvents) {
            printf("  sharedEvents class: %s\n", class_getName([sharedEvents class]));

            // waitEvents array
            id waitEvents = ((id (*)(id, SEL))objc_msgSend)(
                sharedEvents, NSSelectorFromString(@"waitEvents"));
            dump_array("waitEvents", waitEvents);

            // signalEvents (should be empty on this request — only wait gate set)
            id signalEvents = ((id (*)(id, SEL))objc_msgSend)(
                sharedEvents, NSSelectorFromString(@"signalEvents"));
            dump_array("signalEvents", signalEvents);

            // Inspect the single wait event's embedded sharedEvent + threshold value
            if ([waitEvents isKindOfClass:[NSArray class]] && [waitEvents count] > 0) {
                id aneWait = [waitEvents objectAtIndex:0];
                printf("\n  First waitEvent:\n");
                printf("    class: %s\n", class_getName([aneWait class]));

                NSArray *waitProps = @[@"value", @"sharedEvent", @"eventType"];
                for (NSString *p in waitProps) {
                    SEL sel = NSSelectorFromString(p);
                    if ([aneWait respondsToSelector:sel]) {
                        if ([p isEqualToString:@"value"]) {
                            uint64_t v = ((uint64_t (*)(id, SEL))objc_msgSend)(aneWait, sel);
                            printf("    value: %llu\n", (unsigned long long)v);
                        } else {
                            id r = ((id (*)(id, SEL))objc_msgSend)(aneWait, sel);
                            printf("    %s: %s\n", [p UTF8String],
                                   r ? [[r description] UTF8String] : "NIL");
                            if ([p isEqualToString:@"sharedEvent"] && r) {
                                printf("    shared class: %s\n", class_getName([r class]));
                                // Mach port on IOSurfaceSharedEvent
                                SEL portSel = NSSelectorFromString(@"eventPort");
                                if ([r respondsToSelector:portSel]) {
                                    uint32_t port = ((uint32_t (*)(id, SEL))objc_msgSend)(r, portSel);
                                    printf("    eventPort: 0x%x (%s)\n", port,
                                           port ? "NON-ZERO (valid)" : "ZERO (INVALID)");
                                }
                                // Current signaledValue
                                SEL sigVal = NSSelectorFromString(@"signaledValue");
                                if ([r respondsToSelector:sigVal]) {
                                    uint64_t sv = ((uint64_t (*)(id, SEL))objc_msgSend)(r, sigVal);
                                    printf("    signaledValue: %llu\n",
                                           (unsigned long long)sv);
                                }
                            }
                        }
                    } else {
                        printf("    %s: (selector not supported)\n", [p UTF8String]);
                    }
                }
            }
        }

        printf("\n=== Summary ===\n");
        id se = ((id (*)(id, SEL))objc_msgSend)(afterReq, NSSelectorFromString(@"sharedEvents"));
        if (!se) {
            printf("  ROOT CAUSE: setWaitGate did not reach internal request (sharedEvents NIL)\n");
            [model unloadWithError:nil]; return 1;
        }
        id we = ((id (*)(id, SEL))objc_msgSend)(se, NSSelectorFromString(@"waitEvents"));
        NSUInteger weCount = (we && [we isKindOfClass:[NSArray class]]) ? [we count] : 0;
        printf("  waitEvents count: %lu\n", (unsigned long)weCount);
        if (weCount == 0) {
            printf("  ROOT CAUSE: waitEvents not populated by setWaitGate\n");
        } else {
            printf("  waitEvents IS populated — Probe 1 silent-skip is NOT caused by our code missing the array\n");
            printf("  The failure is firmware-level: ANE sees an unmet wait threshold and silently drops the request\n");
        }

        [model unloadWithError:nil];
        printf("\nModel unloaded.\n");
    }
    return 0;
}
