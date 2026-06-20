// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Runtime policy layer for inference parameter resolution.
///
/// Runs before each `generate()` call and resolves memory, KV cache,
/// prompt cache, and thermal throttling parameters from current device state.
///
public struct InferencePolicy: Sendable {

    /// Real-time device state sampled before generation.
    public struct DeviceSnapshot: Sendable {
        /// Memory remaining before Jetsam termination, in MB.
        public let availableMemoryMB: Int
        /// Actual Jetsam limit in MB, equal to footprint plus available memory.
        public let jetsamLimitMB: Int
        /// Total device RAM in GB, used for logging.
        public let totalRAMGB: Int
        /// Current thermal state.
        public let thermalLevel: ThermalManager.ThermalLevel
        /// Measured memory bandwidth in GB/s, when available.
        public let measuredBandwidthGBs: Double?

        /// Samples current device state. Bandwidth is supplied by the caller when known.
        public static func capture(bandwidthGBs: Double? = nil) -> DeviceSnapshot {
            let memSnapshot = DeviceProfile.captureMemorySnapshot()
            let thermal = ThermalManager().level
            return DeviceSnapshot(
                availableMemoryMB: memSnapshot.availableMB,
                jetsamLimitMB: memSnapshot.jetsamLimitMB,
                totalRAMGB: memSnapshot.totalPhysicalMB / 1024,
                thermalLevel: thermal,
                measuredBandwidthGBs: bandwidthGBs
            )
        }
    }

    /// Context for the current conversation turn.
    public struct TurnContext: Sendable {
        /// One-based conversation turn index.
        public let turn: Int
        /// Number of tokens currently cached in prompt cache.
        public let cachedTokenCount: Int
        /// Model architecture metadata for KV budget calculation.
        public let archInfo: ModelArchInfo
        /// Scene type used by DSR budget allocation.
        public let scene: DSRSceneType
        /// Requested maximum output tokens.
        public let requestedMaxTokens: Int
        /// Baseline prefill step size computed by `MemoryBudgetPlanner`.
        public let planPrefillStepSize: Int?
        /// Product-level memory intent from the load-time planner or caller.
        public let memoryIntent: EdgeMemoryIntent

        public init(
            turn: Int = 1,
            cachedTokenCount: Int = 0,
            archInfo: ModelArchInfo,
            scene: DSRSceneType = .chat,
            requestedMaxTokens: Int = 2048,
            planPrefillStepSize: Int? = nil,
            memoryIntent: EdgeMemoryIntent = .balanced
        ) {
            self.turn = turn
            self.cachedTokenCount = cachedTokenCount
            self.archInfo = archInfo
            self.scene = scene
            self.requestedMaxTokens = requestedMaxTokens
            self.planPrefillStepSize = planPrefillStepSize
            self.memoryIntent = memoryIntent
        }
    }

    /// Resolved policy that maps directly to `EdgeGenerateParameters`.
    public struct Resolved: Sendable {
        public let useDSR: Bool
        public let dsrMaxCritical: Int?
        public let dsrScene: DSRSceneType
        public let dsrEvictionInterval: Int
        public let maxKVSize: Int?

        public let kvBits: Int?
        public let kvGroupSize: Int
        public let quantizedKVStart: Int

        public let syncEval: Bool
        public let prefillStepSize: Int

        public let maxTokensCap: Int?
        public let shouldPause: Bool

        public let reusePromptCache: Bool

        public let reasoning: String

        /// Applies the resolved policy to generation parameters.
        public func apply(to params: inout EdgeGenerateParameters) {
            if let bits = kvBits {
                params.kvBits = bits
                params.kvGroupSize = kvGroupSize
                params.quantizedKVStart = quantizedKVStart
            }

            params.prefillStepSize = prefillStepSize
            params.syncEval = syncEval
            params.useDSR = useDSR
            params.dsrMaxCritical = dsrMaxCritical
            if useDSR {
                params.maxKVSize = nil
            } else if let maxKVSize {
                params.maxKVSize = params.maxKVSize.map { min($0, maxKVSize) } ?? maxKVSize
            }
            params.dsrScene = dsrScene
            params.dsrEvictionInterval = dsrEvictionInterval

            if let cap = maxTokensCap {
                params.maxTokens = min(params.maxTokens, cap)
            }
        }
    }

    /// Resolves inference parameters from device state, turn context, and static policy.
    ///
    /// Dynamic policy may tighten the static load-time policy, but must not relax it.
    public static func resolve(
        snapshot: DeviceSnapshot,
        context: TurnContext,
        staticPolicy: KVCacheMemoryPolicy? = nil,
        dsrMaxCriticalOverride: Int? = nil
    ) -> Resolved {
        let avail = snapshot.availableMemoryMB
        let arch = context.archInfo
        var reasons: [String] = []
        let explicitDSRMaxCritical = dsrMaxCriticalOverride.map { max(512, $0) }
        let memoryIntent = staticPolicy?.memoryIntent ?? context.memoryIntent
        reasons.append("intent=\(memoryIntent.rawValue)")

        let useDSR: Bool
        let dsrMaxCritical: Int?

        if let sp = staticPolicy {
            if sp.useDSR {
                useDSR = true
                if let explicitDSRMaxCritical {
                    dsrMaxCritical = explicitDSRMaxCritical
                    reasons.append("DSR on (override \(explicitDSRMaxCritical), static \(sp.dsrMaxCritical ?? sp.maxContextLength))")
                } else {
                    let staticMax = sp.dsrMaxCritical ?? sp.maxContextLength
                    let safeAvailMB = max(0, avail - 200)
                    let dynamicMax = arch.maxContextLength(
                        availableMemoryMB: safeAvailMB, quantized: false
                    )
                    dsrMaxCritical = max(512, min(staticMax, dynamicMax))
                    reasons.append("DSR on (static \(staticMax), dynamic \(dynamicMax), final \(dsrMaxCritical!))")
                }
            } else {
                if let explicitDSRMaxCritical {
                    useDSR = true
                    dsrMaxCritical = explicitDSRMaxCritical
                    reasons.append("DSR on (override \(explicitDSRMaxCritical), static policy disabled)")
                } else {
                    useDSR = false
                    dsrMaxCritical = nil
                    reasons.append("DSR unavailable (static policy)")
                }
            }
        } else {
            let kvPerTokenMB = Double(arch.kvBytesPerTokenFP16) / (1024 * 1024)
            let projectedKVForDefault = kvPerTokenMB * 4096
            let memoryIsTight = avail < 2048 || projectedKVForDefault > Double(avail) * 0.4

            if let explicitDSRMaxCritical {
                useDSR = true
                dsrMaxCritical = explicitDSRMaxCritical
                reasons.append("DSR on (override \(explicitDSRMaxCritical)tok, avail \(avail)MB)")
            } else if memoryIsTight || context.turn >= 3 {
                useDSR = true
                let safeAvailMB = max(0, avail - 200)
                let maxTokensFromMemory = arch.maxContextLength(
                    availableMemoryMB: safeAvailMB, quantized: false
                )
                let baseMaxCritical = min(8192, max(512, maxTokensFromMemory))
                dsrMaxCritical = MemoryBudgetPlanner.intentAdjustedDSRMaxCritical(
                    base: baseMaxCritical,
                    budgetCeiling: maxTokensFromMemory,
                    intent: memoryIntent
                )
                reasons.append("DSR on (budget \(dsrMaxCritical!)tok, avail \(avail)MB)")
            } else {
                useDSR = false
                dsrMaxCritical = nil
                reasons.append("DSR off (avail \(avail)MB, turn \(context.turn))")
            }
        }
        let maxKVSize: Int? = {
            guard !useDSR, let sp = staticPolicy else { return nil }
            return sp.maxKVSize ?? sp.maxContextLength
        }()

        let kvBits: Int?
        let quantizedKVStart: Int

        if let sp = staticPolicy {
            let staticBits = sp.kvBits
            let staticStart = sp.quantizeAtTokens

            if avail < 768 {
                kvBits = 4
                quantizedKVStart = min(staticStart, 0)
                reasons.append("KV INT4@0 (static INT\(staticBits)@\(staticStart), avail \(avail)MB < 768)")
            } else if avail < 1024 && staticBits > 4 {
                kvBits = 4
                quantizedKVStart = min(staticStart, 256)
                reasons.append("KV INT4@\(min(staticStart, 256)) (tightened from INT\(staticBits), avail \(avail)MB)")
            } else {
                kvBits = staticBits
                quantizedKVStart = staticStart
                reasons.append("KV INT\(staticBits)@\(staticStart) (static, avail \(avail)MB)")
            }
        } else {
            if avail < 768 {
                kvBits = 4
                quantizedKVStart = 0
                reasons.append("KV INT4 immediate (avail \(avail)MB < 768)")
            } else if avail < 1024 {
                kvBits = 4
                quantizedKVStart = 256
                reasons.append("KV INT4 @256tok (avail \(avail)MB < 1024)")
            } else if avail < 1536 {
                kvBits = 8
                quantizedKVStart = 512
                reasons.append("KV INT8 @512tok (avail \(avail)MB < 1536)")
            } else {
                kvBits = nil
                quantizedKVStart = 0
                reasons.append("KV FP16 (avail \(avail)MB)")
            }
        }

        let syncEval: Bool
        let ctxSyncThreshold = 8192
        if let sp = staticPolicy {
            if sp.syncEval {
                syncEval = true
                reasons.append("syncEval on (static policy)")
            } else if avail < 512 {
                syncEval = true
                reasons.append("syncEval on (critical avail \(avail)MB)")
            } else {
                syncEval = false
                reasons.append("syncEval off (static policy)")
            }
        } else if context.cachedTokenCount >= ctxSyncThreshold {
            syncEval = true
            reasons.append("syncEval on (ctx \(context.cachedTokenCount) >= \(ctxSyncThreshold))")
        } else {
            syncEval = avail < 1024 || snapshot.jetsamLimitMB <= 6144
        }
        if syncEval && !reasons.contains(where: { $0.contains("syncEval") }) {
            reasons.append("syncEval on")
        }

        let basePrefill = context.planPrefillStepSize ?? 256
        let prefillStepSize: Int

        if avail < 600 {
            prefillStepSize = 64
            reasons.append("prefill 64 (avail \(avail)MB critical)")
        } else if avail < 800 {
            prefillStepSize = min(basePrefill, 128)
            reasons.append("prefill \(min(basePrefill, 128)) (avail \(avail)MB)")
        } else if avail < 1536 {
            prefillStepSize = min(basePrefill, 256)
        } else {
            prefillStepSize = basePrefill
        }

        let maxTokensCap: Int?
        let shouldPause: Bool

        switch snapshot.thermalLevel {
        case .nominal, .fair:
            maxTokensCap = nil
            shouldPause = false
        case .serious:
            maxTokensCap = min(context.requestedMaxTokens, 512)
            shouldPause = false
            reasons.append("thermal throttle (serious)")
        case .critical:
            maxTokensCap = 0
            shouldPause = true
            reasons.append("thermal PAUSE (critical)")
        }

        let reuseCache = context.turn > 1 && context.cachedTokenCount > 0
        if reuseCache {
            reasons.append("cache reuse (\(context.cachedTokenCount)tok)")
        }

        let dsrEvictionInterval: Int
        if avail < 1024 || (staticPolicy?.mode == .aggressive) {
            dsrEvictionInterval = 16
        } else if avail < 2048 {
            dsrEvictionInterval = 32
        } else {
            dsrEvictionInterval = 64
        }

        return Resolved(
            useDSR: useDSR,
            dsrMaxCritical: dsrMaxCritical,
            dsrScene: context.scene,
            dsrEvictionInterval: dsrEvictionInterval,
            maxKVSize: maxKVSize,
            kvBits: kvBits,
            kvGroupSize: 64,
            quantizedKVStart: quantizedKVStart,
            syncEval: syncEval,
            prefillStepSize: prefillStepSize,
            maxTokensCap: maxTokensCap,
            shouldPause: shouldPause,
            reusePromptCache: reuseCache,
            reasoning: reasons.joined(separator: " | ")
        )
    }
}
