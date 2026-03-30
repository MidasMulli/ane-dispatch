// ANEDispatch.m — Direct Neural Engine Programming for Apple Silicon
// Implementation wrapping proven _ANEClient private APIs.
//
// Copyright 2026 Nick Lo. MIT License.

#import "ANEDispatch.h"
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// ════════════════════════════════════════════════════════════════
#pragma mark - Framework Loading
// ════════════════════════════════════════════════════════════════

static BOOL g_frameworksLoaded = NO;

static void ensureFrameworks(void) {
    if (g_frameworksLoaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler", RTLD_NOW);
    g_frameworksLoaded = YES;
}

// ════════════════════════════════════════════════════════════════
#pragma mark - ANEBuffer
// ════════════════════════════════════════════════════════════════

@implementation ANEBuffer {
    IOSurfaceRef _surface;
    id _aneObject;
}

+ (nullable instancetype)bufferWithShape:(NSArray<NSNumber *> *)shape dtype:(ANEDtype)dtype {
    ANEBuffer *buf = [[ANEBuffer alloc] init];
    if (!buf) return nil;

    // Calculate size from shape
    NSUInteger elements = 1;
    for (NSNumber *dim in shape) elements *= [dim unsignedIntegerValue];

    NSUInteger bytesPerElement;
    switch (dtype) {
        case ANEDtypeFloat16: bytesPerElement = 2; break;
        case ANEDtypeFloat32: bytesPerElement = 4; break;
        case ANEDtypeInt8:    bytesPerElement = 1; break;
        case ANEDtypeInt16:   bytesPerElement = 2; break;
    }

    size_t totalBytes = elements * bytesPerElement;
    // ANE constraint: minimum 49KB allocation (Orion constraint #4)
    if (totalBytes < 49152) totalBytes = 49152;

    buf->_surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(totalBytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(totalBytes),
        (id)kIOSurfaceAllocSize: @(totalBytes),
        (id)kIOSurfacePixelFormat: @0,
    });

    if (!buf->_surface) return nil;

    buf->_shape = [shape copy];
    buf->_dtype = dtype;
    buf->_allocSize = totalBytes;

    // Wrap in _ANEIOSurfaceObject
    ensureFrameworks();
    Class cls = NSClassFromString(@"_ANEIOSurfaceObject");
    if (cls) {
        buf->_aneObject = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
            (id)cls, NSSelectorFromString(@"objectWithIOSurface:"), buf->_surface);
    }

    return buf;
}

+ (nullable instancetype)bufferWithIOSurface:(IOSurfaceRef)surface {
    ANEBuffer *buf = [[ANEBuffer alloc] init];
    if (!buf || !surface) return nil;

    CFRetain(surface);
    buf->_surface = surface;
    buf->_allocSize = IOSurfaceGetAllocSize(surface);

    ensureFrameworks();
    Class cls = NSClassFromString(@"_ANEIOSurfaceObject");
    if (cls) {
        buf->_aneObject = ((id (*)(id, SEL, IOSurfaceRef))objc_msgSend)(
            (id)cls, NSSelectorFromString(@"objectWithIOSurface:"), surface);
    }

    return buf;
}

- (IOSurfaceRef)surface { return _surface; }
- (id)aneObject { return _aneObject; }

- (void *)lockAndGetBaseAddress {
    IOSurfaceLock(_surface, 0, NULL);
    return IOSurfaceGetBaseAddress(_surface);
}

- (void)unlock {
    IOSurfaceUnlock(_surface, 0, NULL);
}

- (void)fillFloat16:(const uint16_t *)data count:(NSUInteger)count {
    void *ptr = [self lockAndGetBaseAddress];
    memcpy(ptr, data, count * sizeof(uint16_t));
    [self unlock];
}

- (void)readFloat16:(uint16_t *)dst count:(NSUInteger)count {
    IOSurfaceLock(_surface, kIOSurfaceLockReadOnly, NULL);
    void *ptr = IOSurfaceGetBaseAddress(_surface);
    memcpy(dst, ptr, count * sizeof(uint16_t));
    IOSurfaceUnlock(_surface, kIOSurfaceLockReadOnly, NULL);
}

- (void)dealloc {
    if (_surface) CFRelease(_surface);
}

@end

// ════════════════════════════════════════════════════════════════
#pragma mark - ANEEvent
// ════════════════════════════════════════════════════════════════

@implementation ANEEvent {
    id _ioSurfaceEvent;
}

+ (instancetype)event {
    ANEEvent *evt = [[ANEEvent alloc] init];
    evt->_ioSurfaceEvent = [[NSClassFromString(@"IOSurfaceSharedEvent") alloc] init];
    return evt->_ioSurfaceEvent ? evt : nil;
}

+ (nullable instancetype)eventBridgedFromMetal:(id<MTLSharedEvent>)metalEvent {
    ANEEvent *evt = [[ANEEvent alloc] init];

    // Extract Mach port from MTLSharedEvent via MTLSharedEventHandle
    MTLSharedEventHandle *handle = [metalEvent newSharedEventHandle];
    if (!handle) return nil;

    // Private -eventPort method on MTLSharedEventHandle (proven in Phase 2)
    uint32_t port = ((uint32_t (*)(id, SEL))objc_msgSend)((id)handle,
        NSSelectorFromString(@"eventPort"));
    if (port == 0) return nil;

    // Create IOSurfaceSharedEvent from same Mach port (bidirectional bridge)
    evt->_ioSurfaceEvent = ((id (*)(id, SEL, uint32_t))objc_msgSend)(
        [NSClassFromString(@"IOSurfaceSharedEvent") alloc],
        NSSelectorFromString(@"initWithMachPort:"), port);

    return evt->_ioSurfaceEvent ? evt : nil;
}

- (uint64_t)signaledValue {
    return ((uint64_t (*)(id, SEL))objc_msgSend)(_ioSurfaceEvent,
        NSSelectorFromString(@"signaledValue"));
}

- (void)setSignaledValue:(uint64_t)signaledValue {
    ((void (*)(id, SEL, uint64_t))objc_msgSend)(_ioSurfaceEvent,
        @selector(setSignaledValue:), signaledValue);
}

- (BOOL)waitUntilValue:(uint64_t)value timeoutMS:(uint64_t)timeout {
    return ((BOOL (*)(id, SEL, uint64_t, uint64_t))objc_msgSend)(
        _ioSurfaceEvent,
        NSSelectorFromString(@"waitUntilSignaledValue:timeoutMS:"),
        value, timeout);
}

- (nullable id<MTLSharedEvent>)bridgeToMetalDevice:(id<MTLDevice>)device {
    uint32_t port = ((uint32_t (*)(id, SEL))objc_msgSend)(_ioSurfaceEvent,
        NSSelectorFromString(@"eventPort"));
    if (port == 0) return nil;

    // Construct MTLSharedEventHandle with our port
    MTLSharedEventHandle *handle = [[MTLSharedEventHandle alloc] init];
    Ivar privIvar = class_getInstanceVariable([MTLSharedEventHandle class], "_priv");
    if (privIvar) {
        void **privPtr = (void **)((uint8_t *)(__bridge void *)handle + ivar_getOffset(privIvar));
        void *privStruct = calloc(1, 64);
        *(uint32_t *)privStruct = port;
        *privPtr = privStruct;
    }

    return [device newSharedEventWithHandle:handle];
}

- (id)ioSurfaceEvent { return _ioSurfaceEvent; }

@end

// ════════════════════════════════════════════════════════════════
#pragma mark - ANEModel
// ════════════════════════════════════════════════════════════════

@implementation ANEModel {
    id _aneModel;
    ANEModelState _state;
}

+ (nullable instancetype)modelWithCompiledURL:(NSURL *)url error:(NSError **)error {
    ensureFrameworks();

    ANEModel *model = [[ANEModel alloc] init];
    Class cls = NSClassFromString(@"_ANEModel");
    if (!cls) {
        if (error) *error = [NSError errorWithDomain:@"ANEDispatch" code:1
            userInfo:@{NSLocalizedDescriptionKey: @"_ANEModel class not found. AppleNeuralEngine framework may not be available."}];
        return nil;
    }

    model->_aneModel = ((id (*)(id, SEL, id, id))objc_msgSend)(
        (id)cls, NSSelectorFromString(@"modelAtURL:key:"), url, @"default");

    if (!model->_aneModel) {
        if (error) *error = [NSError errorWithDomain:@"ANEDispatch" code:2
            userInfo:@{NSLocalizedDescriptionKey:
                [NSString stringWithFormat:@"Failed to create _ANEModel from %@", url]}];
        return nil;
    }

    model->_state = ANEModelStateCreated;
    return model;
}

- (BOOL)compileWithError:(NSError **)error {
    id client = [[ANEDispatch shared] aneClient];
    BOOL ok = ((BOOL (*)(id, SEL, id, id, int, NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"compileModel:options:qos:error:"),
        _aneModel, @{}, 21, error);
    if (ok) _state = ANEModelStateCompiled;
    return ok;
}

- (BOOL)loadWithError:(NSError **)error {
    id client = [[ANEDispatch shared] aneClient];
    BOOL ok = ((BOOL (*)(id, SEL, id, id, int, NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"loadModel:options:qos:error:"),
        _aneModel, @{}, 21, error);
    if (ok) _state = ANEModelStateLoaded;
    return ok;
}

- (BOOL)unloadWithError:(NSError **)error {
    id client = [[ANEDispatch shared] aneClient];
    BOOL ok = ((BOOL (*)(id, SEL, id, id, int, NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"unloadModel:options:qos:error:"),
        _aneModel, @{}, 21, error);
    if (ok) _state = ANEModelStateCreated;
    return ok;
}

- (BOOL)prepareWithError:(NSError **)error {
    if (![self compileWithError:error]) return NO;
    return [self loadWithError:error];
}

- (ANEModelState)state { return _state; }

- (uint64_t)programHandle {
    return ((uint64_t (*)(id, SEL))objc_msgSend)(_aneModel,
        NSSelectorFromString(@"programHandle"));
}

- (id)aneModel { return _aneModel; }

@end

// ════════════════════════════════════════════════════════════════
#pragma mark - ANERequest
// ════════════════════════════════════════════════════════════════

@implementation ANERequest {
    id _aneRequest;
    id _sharedEvents;
}

+ (nullable instancetype)requestWithInputs:(NSArray<ANEBuffer *> *)inputs
                                   outputs:(NSArray<ANEBuffer *> *)outputs {
    ensureFrameworks();

    ANERequest *req = [[ANERequest alloc] init];

    // Build arrays of _ANEIOSurfaceObject and indices
    NSMutableArray *inObjects = [NSMutableArray arrayWithCapacity:inputs.count];
    NSMutableArray *inIndices = [NSMutableArray arrayWithCapacity:inputs.count];
    NSMutableArray *outObjects = [NSMutableArray arrayWithCapacity:outputs.count];
    NSMutableArray *outIndices = [NSMutableArray arrayWithCapacity:outputs.count];

    for (NSUInteger i = 0; i < inputs.count; i++) {
        [inObjects addObject:[inputs[i] aneObject]];
        [inIndices addObject:@(i)];
    }
    for (NSUInteger i = 0; i < outputs.count; i++) {
        [outObjects addObject:[outputs[i] aneObject]];
        [outIndices addObject:@(i)];
    }

    Class cls = NSClassFromString(@"_ANERequest");
    req->_aneRequest = ((id (*)(id, SEL, id, id, id, id, uint32_t))objc_msgSend)(
        (id)cls,
        NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
        inObjects, inIndices, outObjects, outIndices, (uint32_t)0);

    return req->_aneRequest ? req : nil;
}

- (void)setCompletionSignal:(ANEEvent *)event value:(uint64_t)value {
    ensureFrameworks();

    Class signalCls = NSClassFromString(@"_ANESharedSignalEvent");
    Class eventsCls = NSClassFromString(@"_ANESharedEvents");

    id aneSignal = ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
        signalCls,
        NSSelectorFromString(@"signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
        value, (uint32_t)0, (int64_t)0, [event ioSurfaceEvent]);

    if (!aneSignal) return;

    // Get existing wait events or empty array
    NSArray *waits = @[];
    if (_sharedEvents) {
        id existing = ((id (*)(id, SEL))objc_msgSend)(_sharedEvents,
            NSSelectorFromString(@"waitEvents"));
        if (existing) waits = existing;
    }

    // Use init directly (proven pattern from debug test: comp=42)
    _sharedEvents = ((id (*)(id, SEL, id, id))objc_msgSend)(
        [eventsCls alloc],
        NSSelectorFromString(@"initWithSignalEvents:waitEvents:"),
        @[aneSignal], waits);

    if (_sharedEvents) {
        ((void (*)(id, SEL, id))objc_msgSend)(_aneRequest,
            NSSelectorFromString(@"setSharedEvents:"), _sharedEvents);
    }
}

- (void)setWaitGate:(ANEEvent *)event value:(uint64_t)value {
    ensureFrameworks();

    Class waitCls = NSClassFromString(@"_ANESharedWaitEvent");
    Class eventsCls = NSClassFromString(@"_ANESharedEvents");

    id aneWait = ((id (*)(Class, SEL, uint64_t, id, uint64_t))objc_msgSend)(
        waitCls,
        NSSelectorFromString(@"waitEventWithValue:sharedEvent:eventType:"),
        value, [event ioSurfaceEvent], (uint64_t)0);

    if (!aneWait) return;

    // Get existing signal events or empty array
    NSArray *signals = @[];
    if (_sharedEvents) {
        id existing = ((id (*)(id, SEL))objc_msgSend)(_sharedEvents,
            NSSelectorFromString(@"signalEvents"));
        if (existing) signals = existing;
    }

    _sharedEvents = ((id (*)(id, SEL, id, id))objc_msgSend)(
        [eventsCls alloc],
        NSSelectorFromString(@"initWithSignalEvents:waitEvents:"),
        signals, @[aneWait]);

    ((void (*)(id, SEL, id))objc_msgSend)(_aneRequest,
        NSSelectorFromString(@"setSharedEvents:"), _sharedEvents);
}

- (id)aneRequest { return _aneRequest; }

@end

// ════════════════════════════════════════════════════════════════
#pragma mark - ANEChainRequest (Experimental)
// ════════════════════════════════════════════════════════════════

@implementation ANEChainRequest {
    id _aneChainingRequest;
}

+ (nullable instancetype)chainWithInputs:(NSArray<ANEBuffer *> *)inputs
                              outputSets:(NSArray<NSArray<ANEBuffer *> *> *)outputSets
                    loopbackInputIndices:(NSArray<NSNumber *> *)loopbackInputIndices
                   loopbackOutputIndices:(NSArray<NSNumber *> *)loopbackOutputIndices
                             signalEvent:(nullable ANEEvent *)signalEvent
                          fwEnqueueDelay:(nullable NSNumber *)fwEnqueueDelay
                            memoryPoolId:(nullable NSNumber *)memoryPoolId {
    ensureFrameworks();

    ANEChainRequest *chain = [[ANEChainRequest alloc] init];
    Class cls = NSClassFromString(@"_ANEChainingRequest");
    if (!cls) return nil;

    // Build input IOSurface objects
    NSMutableArray *inObjects = [NSMutableArray array];
    for (ANEBuffer *buf in inputs)
        [inObjects addObject:[buf aneObject]];

    // Build output set arrays
    NSMutableArray *outSets = [NSMutableArray array];
    for (NSArray<ANEBuffer *> *set in outputSets) {
        NSMutableArray *setObjects = [NSMutableArray array];
        for (ANEBuffer *buf in set)
            [setObjects addObject:[buf aneObject]];
        [outSets addObject:setObjects];
    }

    // Build signal events array
    NSArray *signalEvents = @[];
    if (signalEvent) {
        Class signalCls = NSClassFromString(@"_ANESharedSignalEvent");
        id aneSig = ((id (*)(Class, SEL, uint64_t, uint32_t, int64_t, id))objc_msgSend)(
            signalCls,
            NSSelectorFromString(@"signalEventWithValue:symbolIndex:eventType:sharedEvent:"),
            (uint64_t)1, (uint32_t)0, (int64_t)0, [signalEvent ioSurfaceEvent]);
        if (aneSig) signalEvents = @[aneSig];
    }

    // All params are NSObjects (confirmed via runtime introspection)
    SEL factorySel = NSSelectorFromString(
        @"chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:"
        @"procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");

    if ([cls respondsToSelector:factorySel]) {
        chain->_aneChainingRequest = ((id (*)(id, SEL, id, id, id, id,
            id, id, id, id, id))objc_msgSend)(
            (id)cls, factorySel,
            inObjects,                  // inputs
            outSets,                    // outputSets
            loopbackInputIndices,       // lbInputSymbolId (NSArray<NSNumber>)
            loopbackOutputIndices,      // lbOutputSymbolId (NSArray<NSNumber>)
            @(0),                       // procedureIndex (NSNumber)
            signalEvents,               // signalEvents
            nil,                        // transactionHandle
            fwEnqueueDelay,             // fwEnqueueDelay (NSNumber or nil)
            memoryPoolId                // memoryPoolId (NSNumber or nil)
        );
    }

    return chain->_aneChainingRequest ? chain : nil;
}

- (id)aneChainingRequest { return _aneChainingRequest; }

@end

// ════════════════════════════════════════════════════════════════
#pragma mark - ANEDispatch
// ════════════════════════════════════════════════════════════════

@implementation ANEDispatch {
    id _aneClient;
}

+ (instancetype)shared {
    static ANEDispatch *instance;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[ANEDispatch alloc] init];
        ensureFrameworks();
        instance->_aneClient = ((id (*)(id, SEL))objc_msgSend)(
            (id)NSClassFromString(@"_ANEClient"),
            NSSelectorFromString(@"sharedConnection"));
    });
    return instance;
}

- (BOOL)evaluate:(ANEModel *)model
         request:(ANERequest *)request
           error:(NSError **)error {
    // Use doEvaluateDirectWithModel: — 37% faster than evaluateWithModel:,
    // and avoids the XPC completion handler crash when SharedEvents are attached
    return ((BOOL (*)(id, SEL, id, id, id, int, NSError **))objc_msgSend)(
        _aneClient,
        NSSelectorFromString(@"doEvaluateDirectWithModel:options:request:qos:error:"),
        [(ANEModel *)model aneModel], @{}, [request aneRequest], 21, error);
}

- (BOOL)mapBuffers:(ANEModel *)model
           request:(ANERequest *)request
             error:(NSError **)error {
    return ((BOOL (*)(id, SEL, id, id, BOOL, NSError **))objc_msgSend)(
        _aneClient,
        NSSelectorFromString(@"mapIOSurfacesWithModel:request:cacheInference:error:"),
        [(ANEModel *)model aneModel], [request aneRequest], NO, error);
}

- (void)unmapBuffers:(ANEModel *)model request:(ANERequest *)request {
    ((void (*)(id, SEL, id, id))objc_msgSend)(
        _aneClient,
        NSSelectorFromString(@"unmapIOSurfacesWithModel:request:"),
        [(ANEModel *)model aneModel], [request aneRequest]);
}

- (BOOL)prepareChain:(ANEModel *)model
        chainRequest:(ANEChainRequest *)chain
               error:(NSError **)error {
    return ((BOOL (*)(id, SEL, id, id, id, int, NSError **))objc_msgSend)(
        _aneClient,
        NSSelectorFromString(@"doPrepareChainingWithModel:options:chainingReq:qos:error:"),
        [(ANEModel *)model aneModel], @{}, [chain aneChainingRequest], 21, error);
}

- (BOOL)compiledModelExistsFor:(ANEModel *)model {
    return ((BOOL (*)(id, SEL, id))objc_msgSend)(
        _aneClient,
        NSSelectorFromString(@"compiledModelExistsFor:"),
        [(ANEModel *)model aneModel]);
}

- (id)aneClient { return _aneClient; }

@end
