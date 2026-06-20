// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine

public extension NativeRuntimeBridge {
    static func metalConfiguration(
        for plan: MemoryBudgetPlanner.Plan,
        contextLengthHint: Int? = nil,
        quantizedBufferCacheLimitBytes: Int? = nil,
        releaseQuantizedHostStorageAfterUpload: Bool = false
    ) -> MetalRuntimeConfiguration {
        let cacheLimitBytes = quantizedBufferCacheLimitBytes ?? plan.cacheLimitBytes
        var configuration = metalConfiguration(
            maxOpsPerCommandBuffer: plan.maxOpsPerBuffer,
            maxMBPerCommandBuffer: plan.maxMBPerBuffer,
            contextLengthHint: contextLengthHint ?? plan.dsrMaxCritical,
            dynamicOpsSchedule: plan.dynamicOpsEnabled
                ? DynamicOpsSchedule(
                    floor: plan.dynamicOpsFloor,
                    contextLow: plan.dynamicOpsCtxLow,
                    contextHigh: plan.dynamicOpsCtxHigh
                )
                : nil,
            memoryLimitBytes: plan.memoryLimitBytes,
            quantizedBufferCacheLimitBytes: cacheLimitBytes,
            releaseQuantizedHostStorageAfterUpload: releaseQuantizedHostStorageAfterUpload || cacheLimitBytes > 0
        )
        configuration.commandBufferBatchingEnabled = true
        configuration.useVendoredCommandBufferPrefillQMM = plan.vendoredCommandBufferPrefillQMMEnabled
        configuration.useFusedGDNDecode = plan.fusedGDNDecodeEnabled
        if plan.syncEval {
            configuration.maxInFlightCommandBuffers = 1
        }
        return configuration.applyingEnvironmentOverrides()
    }
}
