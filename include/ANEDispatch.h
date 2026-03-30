// ANEDispatch.h — Direct Neural Engine Programming for Apple Silicon
//
// First open library for ANE dispatch without CoreML.
// Supports: model compile/load, IOSurface I/O, SharedEvents cross-accelerator sync,
// chaining requests with firmware-level enqueue delay.
//
// Copyright 2026 Nick Lo. MIT License.
// https://github.com/MidasMulli/ane-dispatch

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>

NS_ASSUME_NONNULL_BEGIN

// ════════════════════════════════════════════════════════════════
// MARK: - ANEBuffer
// Zero-copy IOSurface wrapper for ANE I/O
// ════════════════════════════════════════════════════════════════

typedef NS_ENUM(NSInteger, ANEDtype) {
    ANEDtypeFloat16 = 0,
    ANEDtypeFloat32 = 1,
    ANEDtypeInt8    = 2,
    ANEDtypeInt16   = 3,
};

@interface ANEBuffer : NSObject

/// Create a buffer with given shape and data type.
/// Automatically applies ANE constraints (min 49KB allocation, packed layout).
/// @param shape Array of NSNumbers describing tensor dimensions [N, C, H, W]
/// @param dtype Data type for buffer elements
+ (nullable instancetype)bufferWithShape:(NSArray<NSNumber *> *)shape
                                   dtype:(ANEDtype)dtype;

/// Create from existing IOSurface (zero-copy wrap)
+ (nullable instancetype)bufferWithIOSurface:(IOSurfaceRef)surface;

/// Raw IOSurface for interop
@property (nonatomic, readonly) IOSurfaceRef surface;

/// Internal wrapped object for ANE dispatch
@property (nonatomic, readonly) id aneObject;

/// Shape and dtype
@property (nonatomic, readonly) NSArray<NSNumber *> *shape;
@property (nonatomic, readonly) ANEDtype dtype;
@property (nonatomic, readonly) size_t allocSize;

/// Lock/unlock for CPU access (required before reading/writing data)
- (void *)lockAndGetBaseAddress;
- (void)unlock;

/// Convenience: write FP16 data
- (void)fillFloat16:(const uint16_t *)data count:(NSUInteger)count;

/// Convenience: read FP16 data
- (void)readFloat16:(uint16_t *)dst count:(NSUInteger)count;

@end

// ════════════════════════════════════════════════════════════════
// MARK: - ANEEvent
// Cross-accelerator synchronization via IOSurfaceSharedEvent bridge
// ════════════════════════════════════════════════════════════════

@interface ANEEvent : NSObject

/// Create a standalone event (CPU-signalable)
+ (instancetype)event;

/// Create from an existing MTLSharedEvent (bridges Metal → ANE)
+ (nullable instancetype)eventBridgedFromMetal:(id<MTLSharedEvent>)metalEvent;

/// Current signaled value
@property (nonatomic) uint64_t signaledValue;

/// Block until value reaches target (with timeout in ms, 0 = no timeout)
- (BOOL)waitUntilValue:(uint64_t)value timeoutMS:(uint64_t)timeout;

/// Bridge this event TO a Metal shared event (for GPU-side observation)
- (nullable id<MTLSharedEvent>)bridgeToMetalDevice:(id<MTLDevice>)device;

/// Internal IOSurfaceSharedEvent
@property (nonatomic, readonly) id ioSurfaceEvent;

@end

// ════════════════════════════════════════════════════════════════
// MARK: - ANEModel
// Direct ANE model: compile → load → evaluate
// ════════════════════════════════════════════════════════════════

typedef NS_ENUM(NSInteger, ANEModelState) {
    ANEModelStateCreated   = 1,
    ANEModelStateCompiled  = 2,
    ANEModelStateLoaded    = 3,
};

@interface ANEModel : NSObject

/// Load a compiled CoreML model (.mlmodelc directory)
+ (nullable instancetype)modelWithCompiledURL:(NSURL *)url
                                        error:(NSError **)error;

/// Compile the model for ANE execution
- (BOOL)compileWithError:(NSError **)error;

/// Load onto ANE hardware
- (BOOL)loadWithError:(NSError **)error;

/// Unload from ANE hardware
- (BOOL)unloadWithError:(NSError **)error;

/// Compile + load in one call
- (BOOL)prepareWithError:(NSError **)error;

/// Current state
@property (nonatomic, readonly) ANEModelState state;

/// Internal program handle (valid after load)
@property (nonatomic, readonly) uint64_t programHandle;

@end

// ════════════════════════════════════════════════════════════════
// MARK: - ANERequest
// Evaluation request with optional SharedEvents
// ════════════════════════════════════════════════════════════════

@interface ANERequest : NSObject

/// Create a request with input/output buffers
+ (nullable instancetype)requestWithInputs:(NSArray<ANEBuffer *> *)inputs
                                   outputs:(NSArray<ANEBuffer *> *)outputs;

/// Optional: signal this event to `value` after ANE execution completes
- (void)setCompletionSignal:(ANEEvent *)event value:(uint64_t)value;

/// Optional: wait for this event to reach `value` before executing
/// NOTE: Pre-signaled waits work. Cross-accelerator waits (GPU→ANE) are experimental.
- (void)setWaitGate:(ANEEvent *)event value:(uint64_t)value;

/// Internal request object
@property (nonatomic, readonly) id aneRequest;

@end

// ════════════════════════════════════════════════════════════════
// MARK: - ANEChainRequest (Experimental)
// Pipelined multi-step execution with firmware-level timing
// ════════════════════════════════════════════════════════════════

@interface ANEChainRequest : NSObject

/// Create a chaining request for pipelined execution.
/// Chaining allows output-to-input loopback: the output of step N feeds the input of step N+1.
/// fwEnqueueDelay is firmware-level timing control (units TBD, likely microseconds).
///
/// @param inputs Input buffers
/// @param outputSets Array of output buffer arrays (one set per pipeline step)
/// @param loopbackInputIndices Indices of inputs that receive loopback (NSArray of NSNumber)
/// @param loopbackOutputIndices Indices of outputs that feed loopback (NSArray of NSNumber)
/// @param signalEvent Optional completion signal event
/// @param fwEnqueueDelay Optional firmware delay before dispatch (NSNumber, units TBD)
/// @param memoryPoolId Optional memory pool identifier (NSNumber)
///
/// @note EXPERIMENTAL. Chaining dispatch via _ANEChainingRequest. The loopback
/// mechanism and fwEnqueueDelay are undocumented. Signal events work (confirmed).
/// Wait semantics are partially blocked (see SharedEvents documentation).
+ (nullable instancetype)chainWithInputs:(NSArray<ANEBuffer *> *)inputs
                              outputSets:(NSArray<NSArray<ANEBuffer *> *> *)outputSets
                    loopbackInputIndices:(NSArray<NSNumber *> *)loopbackInputIndices
                   loopbackOutputIndices:(NSArray<NSNumber *> *)loopbackOutputIndices
                             signalEvent:(nullable ANEEvent *)signalEvent
                          fwEnqueueDelay:(nullable NSNumber *)fwEnqueueDelay
                            memoryPoolId:(nullable NSNumber *)memoryPoolId;

/// Internal chaining request object
@property (nonatomic, readonly) id aneChainingRequest;

@end

// ════════════════════════════════════════════════════════════════
// MARK: - ANEDispatch
// Main dispatch interface — singleton, thread-safe
// ════════════════════════════════════════════════════════════════

@interface ANEDispatch : NSObject

/// Shared dispatch instance (wraps _ANEClient.sharedConnection)
+ (instancetype)shared;

/// Evaluate a model with a request (direct dispatch, no CoreML)
/// This uses doEvaluateDirectWithModel: internally (37% faster than CoreML path)
- (BOOL)evaluate:(ANEModel *)model
         request:(ANERequest *)request
           error:(NSError **)error;

/// Map IOSurfaces for a model+request pair (call before first evaluate)
- (BOOL)mapBuffers:(ANEModel *)model
           request:(ANERequest *)request
             error:(NSError **)error;

/// Unmap IOSurfaces
- (void)unmapBuffers:(ANEModel *)model request:(ANERequest *)request;

/// Prepare a chaining request (experimental)
- (BOOL)prepareChain:(ANEModel *)model
        chainRequest:(ANEChainRequest *)chain
               error:(NSError **)error;

/// Check if a compiled model exists in ANE cache
- (BOOL)compiledModelExistsFor:(ANEModel *)model;

/// Internal _ANEClient
@property (nonatomic, readonly) id aneClient;

@end

NS_ASSUME_NONNULL_END
