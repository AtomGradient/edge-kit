// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine

struct NativeCmlxCommandBufferLimitState {
    var commandBufferLimits: (maxOps: Int, maxMB: Int)?
    var memoryLimitBytes: Int?

    mutating func reset() {
        commandBufferLimits = nil
        memoryLimitBytes = nil
    }
}

enum NativeCmlxCommandBufferLimitApplier {
    static func apply(
        contextLengthHint: Int,
        state: inout NativeCmlxCommandBufferLimitState,
        commandBufferDiagnosticName: String,
        memoryLimitDiagnosticName: String,
        includeRequestedContext: Bool = false,
        emitDiagnostic: ((String) -> Void)?
    ) throws {
        var configuration = NativeRuntimeBridge.currentMetalConfiguration
        let effectiveContextLengthHint = max(0, contextLengthHint)
        configuration.contextLengthHint = effectiveContextLengthHint
        let limits = (
            maxOps: configuration.effectiveMaxOpsPerCommandBuffer,
            maxMB: configuration.maxMBPerCommandBuffer
        )

        if state.commandBufferLimits?.maxOps != limits.maxOps ||
            state.commandBufferLimits?.maxMB != limits.maxMB {
            try QwenCmlxLazyDecodeSession.configureCommandBufferLimits(
                maxOps: limits.maxOps,
                maxMB: limits.maxMB
            )
            state.commandBufferLimits = limits
            var message = "\(commandBufferDiagnosticName) maxOps=\(limits.maxOps) maxMB=\(limits.maxMB) ctx=\(effectiveContextLengthHint)"
            if includeRequestedContext {
                message += " requestedCtx=\(max(0, contextLengthHint))"
            }
            emitDiagnostic?(message)
        }

        if let memoryLimitBytes = configuration.memoryLimitBytes,
           state.memoryLimitBytes != memoryLimitBytes {
            try QwenCmlxLazyDecodeSession.configureMemoryLimit(bytes: memoryLimitBytes)
            state.memoryLimitBytes = memoryLimitBytes
            emitDiagnostic?(
                "\(memoryLimitDiagnosticName) memoryLimitMB=\(memoryLimitBytes / 1_048_576)"
            )
        }
    }
}
