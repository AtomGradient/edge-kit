// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeEngine
import Metal

/// Plans memory and runtime settings from device and model signals.
public struct MemoryBudgetPlanner: Sendable {

    struct InferenceCapability: Sendable {
        let metalFamilyTier: Int
        let measuredBandwidthGBs: Double
        let bandwidthSource: DeviceCapabilityProfile.BandwidthSource
        let jetsamBudgetMB: Int
        let physicalMemoryMB: Int
        let recommendedWorkingSetMB: Int?
        let thermalState: DeviceProfile.ThermalState
        let isPhone: Bool
        let isMac: Bool

        init(
            profile: DeviceProfile,
            snapshot: DeviceProfile.MemorySnapshot,
            measuredBandwidthGBs overrideBandwidth: Double?
        ) {
            let hardware = profile.hardwareProfile
            self.metalFamilyTier = hardware.metalFamilyTier
            self.measuredBandwidthGBs = overrideBandwidth ?? hardware.measuredBandwidthGBs ?? 0
            self.bandwidthSource = overrideBandwidth == nil ? hardware.bandwidthSource : .measured
            self.jetsamBudgetMB = snapshot.jetsamLimitMB
            self.physicalMemoryMB = hardware.totalPhysicalMemoryMB
            self.recommendedWorkingSetMB = hardware.recommendedMaxWorkingSetMB
            self.thermalState = profile.thermalState
            self.isPhone = hardware.isPhone
            self.isMac = hardware.isMac || snapshot.jetsamLimitMB > 16_384
        }

        var chipStableAtLongContext: Bool {
            metalFamilyTier >= 10
        }

        var supportsVendoredCommandBufferPrefillQMM: Bool {
            metalFamilyTier >= 9
        }

        var supportsFusedGDNDecode: Bool {
            metalFamilyTier >= 10
        }
    }

    /// Complete memory planning result.
    public struct Plan: Sendable, CustomStringConvertible {
        public let maxOpsPerBuffer: Int
        public let maxMBPerBuffer: Int

        public let memoryLimitBytes: Int
        public let cacheLimitBytes: Int

        public let forwardPassReserveMB: Int
        public let kvBudgetMB: Int
        public let dsrMaxCritical: Int
        public let quantizeAtTokens: Int
        public let kvBits: Int
        public let kvGroupSize: Int

        public let dynamicOpsEnabled: Bool
        public let dynamicOpsFloor: Int
        public let dynamicOpsCtxLow: Int
        public let dynamicOpsCtxHigh: Int

        public let prefillStepSize: Int
        public let syncEval: Bool
        public let vendoredCommandBufferPrefillQMMEnabled: Bool
        public let fusedGDNDecodeEnabled: Bool

        public let mode: KVCacheMemoryPolicy.Mode
        public let memoryIntent: EdgeMemoryIntent
        public let reasoning: String
        public let measuredBandwidthGBs: Double

        public init(
            maxOpsPerBuffer: Int,
            maxMBPerBuffer: Int,
            memoryLimitBytes: Int,
            cacheLimitBytes: Int,
            forwardPassReserveMB: Int,
            kvBudgetMB: Int,
            dsrMaxCritical: Int,
            quantizeAtTokens: Int,
            kvBits: Int,
            kvGroupSize: Int,
            dynamicOpsEnabled: Bool,
            dynamicOpsFloor: Int,
            dynamicOpsCtxLow: Int,
            dynamicOpsCtxHigh: Int,
            prefillStepSize: Int,
            syncEval: Bool,
            vendoredCommandBufferPrefillQMMEnabled: Bool,
            fusedGDNDecodeEnabled: Bool,
            mode: KVCacheMemoryPolicy.Mode,
            reasoning: String,
            measuredBandwidthGBs: Double,
            memoryIntent: EdgeMemoryIntent = .balanced
        ) {
            self.maxOpsPerBuffer = maxOpsPerBuffer
            self.maxMBPerBuffer = maxMBPerBuffer
            self.memoryLimitBytes = memoryLimitBytes
            self.cacheLimitBytes = cacheLimitBytes
            self.forwardPassReserveMB = forwardPassReserveMB
            self.kvBudgetMB = kvBudgetMB
            self.dsrMaxCritical = dsrMaxCritical
            self.quantizeAtTokens = quantizeAtTokens
            self.kvBits = kvBits
            self.kvGroupSize = kvGroupSize
            self.dynamicOpsEnabled = dynamicOpsEnabled
            self.dynamicOpsFloor = dynamicOpsFloor
            self.dynamicOpsCtxLow = dynamicOpsCtxLow
            self.dynamicOpsCtxHigh = dynamicOpsCtxHigh
            self.prefillStepSize = prefillStepSize
            self.syncEval = syncEval
            self.vendoredCommandBufferPrefillQMMEnabled = vendoredCommandBufferPrefillQMMEnabled
            self.fusedGDNDecodeEnabled = fusedGDNDecodeEnabled
            self.mode = mode
            self.memoryIntent = memoryIntent
            self.reasoning = reasoning
            self.measuredBandwidthGBs = measuredBandwidthGBs
        }

        public var description: String {
            """
            MemoryPlan [\(mode.rawValue), intent=\(memoryIntent.rawValue)]:
              maxOps=\(maxOpsPerBuffer) maxMB=\(maxMBPerBuffer)\(dynamicOpsEnabled ? " dynOps=\(maxOpsPerBuffer)→\(dynamicOpsFloor)@ctx\(dynamicOpsCtxLow/1024)K-\(dynamicOpsCtxHigh/1024)K" : "")
              memLimit=\(memoryLimitBytes / 1_048_576)MB cache=\(cacheLimitBytes / 1_048_576)MB
              fwdReserve=\(forwardPassReserveMB)MB kvBudget=\(kvBudgetMB)MB
              dsrMax=\(dsrMaxCritical) INT4@\(quantizeAtTokens)
              prefill=\(prefillStepSize) syncEval=\(syncEval) cbPrefillQMM=\(vendoredCommandBufferPrefillQMMEnabled) fusedGDNDecode=\(fusedGDNDecodeEnabled)
            """
        }

        /// Dictionary representation for device-test JSON reports.
        public var dictionary: [String: Any] {
            [
                "maxOpsPerBuffer": maxOpsPerBuffer,
                "maxMBPerBuffer": maxMBPerBuffer,
                "memoryLimitMB": memoryLimitBytes / 1_048_576,
                "cacheLimitMB": cacheLimitBytes / 1_048_576,
                "forwardPassReserveMB": forwardPassReserveMB,
                "kvBudgetMB": kvBudgetMB,
                "dsrMaxCritical": dsrMaxCritical,
                "quantizeAtTokens": quantizeAtTokens,
                "kvBits": kvBits,
                "dynamicOpsEnabled": dynamicOpsEnabled,
                "dynamicOpsFloor": dynamicOpsFloor,
                "dynamicOpsCtxLow": dynamicOpsCtxLow,
                "dynamicOpsCtxHigh": dynamicOpsCtxHigh,
                "prefillStepSize": prefillStepSize,
                "syncEval": syncEval,
                "vendoredCommandBufferPrefillQMMEnabled": vendoredCommandBufferPrefillQMMEnabled,
                "fusedGDNDecodeEnabled": fusedGDNDecodeEnabled,
                "mode": mode.rawValue,
                "memoryIntent": memoryIntent.rawValue,
                "reasoning": reasoning,
                "measuredBandwidthGBs": measuredBandwidthGBs,
            ]
        }
    }

    /// Computes a complete memory plan.
    ///
    /// - Parameters:
    ///   - profile: Device profile.
    ///   - modelSizeGB: Model size in GB.
    ///   - measuredBandwidthGBs: Optional measured bandwidth in GB/s.
    ///   - rawIntent: Memory intent requested by the caller.
    public static func plan(
        profile: DeviceProfile,
        modelSizeGB: Double,
        measuredBandwidthGBs: Double? = nil,
        intent rawIntent: EdgeMemoryIntent = .balanced
    ) -> Plan {
        let intent = rawIntent
        let snapshot = DeviceProfile.captureMemorySnapshot()
        let footprintMB = snapshot.footprintMB
        let capability = InferenceCapability(
            profile: profile,
            snapshot: snapshot,
            measuredBandwidthGBs: measuredBandwidthGBs
        )
        let effectiveBandwidthGBs = capability.measuredBandwidthGBs

        var reasons: [String] = []

        let estimatedFootprintMB = Int(modelSizeGB * 1024.0 * 0.9) + 200
        let effectiveFootprintMB = max(footprintMB, estimatedFootprintMB)

        let totalBudgetMB = capability.isMac
            ? max(8192, capability.physicalMemoryMB)
            : capability.jetsamBudgetMB

        let rawHeadroomMB = totalBudgetMB - effectiveFootprintMB

        let mode: KVCacheMemoryPolicy.Mode = capability.isMac
            ? .auto
            : (rawHeadroomMB < 1200 ? .aggressive : .auto)

        let safetyMarginMB = mode == .aggressive ? 200 : 400
        let headroomMB = max(0, rawHeadroomMB - safetyMarginMB)

        reasons.append("\(mode == .aggressive ? "Aggressive" : "Auto")")
        reasons.append("jetsam \(totalBudgetMB)MB")
        reasons.append("footprint ~\(effectiveFootprintMB)MB")
        reasons.append("headroom \(headroomMB)MB")
        reasons.append("bw=\(String(format: "%.1f", effectiveBandwidthGBs))GB/s(\(capability.bandwidthSource.rawValue))")
        if capability.metalFamilyTier > 0 {
            reasons.append("metalFamily=\(capability.metalFamilyTier)")
        }
        if let recommendedWorkingSetMB = capability.recommendedWorkingSetMB {
            reasons.append("workingSet=\(recommendedWorkingSetMB)MB")
        }

        let isMac = capability.isMac
        let chipStableAtLongCtx = capability.chipStableAtLongContext
        let highThroughputPhone = capability.isPhone && isHighThroughputPhone(
            effectiveBandwidthGBs: effectiveBandwidthGBs,
            metalFamilyTier: capability.metalFamilyTier
        )

        let maxOps: Int
        let maxMB: Int
        let peakReserveMB: Int

        if isMac {
            maxOps = 700
            maxMB = 256
            peakReserveMB = min(400, headroomMB / 4)
        } else if capability.isPhone {
            let phoneBudget = phoneCommandBufferBudget(highThroughputPhone: highThroughputPhone)
            maxOps = phoneBudget.maxOps
            maxMB = phoneBudget.maxMB
            peakReserveMB = phoneBudget.peakReserveMB
            reasons.append(phoneBudget.reason)
        } else if mode == .aggressive {
            maxOps = 5
            maxMB = 5
            peakReserveMB = 150
        } else {
            maxOps = 15
            maxMB = 15
            peakReserveMB = 300
        }

        let kvBudgetMB = max(0, headroomMB - peakReserveMB)

        reasons.append("maxOps=\(maxOps)")
        reasons.append("peakRsv=\(peakReserveMB)MB")
        reasons.append("kvBudget=\(kvBudgetMB)MB")
        reasons.append("intent=\(intent.rawValue)")

        let dsrMaxCritical = dsrMaxCriticalForPlan(
            kvBudgetMB: kvBudgetMB,
            modelSizeGB: modelSizeGB,
            totalBudgetMB: totalBudgetMB,
            isMac: isMac,
            isPhone: capability.isPhone,
            intent: intent
        )

        let quantizeAt = 0
        reasons.append("INT4@0")

        let prefillStepSize = prefillStepSizeForPlan(
            isMac: isMac,
            isHighThroughputPhone: highThroughputPhone,
            headroomMB: headroomMB
        )

        let cacheLimitMB = quantizedBufferCacheLimitMB(
            totalBudgetMB: totalBudgetMB,
            headroomMB: headroomMB,
            modelSizeGB: modelSizeGB,
            isPhone: capability.isPhone,
            isMac: isMac,
            chipStableAtLongCtx: chipStableAtLongCtx,
            mode: mode
        )
        let cacheLimitBytes = cacheLimitMB * 1024 * 1024
        reasons.append("qcache=\(cacheLimitMB)MB")

        let memoryLimitMB: Int = {
            if isMac {
                let physicalMB = max(8_192, capability.physicalMemoryMB)
                return max(4_096, Int(Double(physicalMB) * 0.80))
            }
            let jetsamMB = capability.jetsamBudgetMB > 0
                ? capability.jetsamBudgetMB
                : max(4_096, capability.physicalMemoryMB / 2)
            let reserveMB = max(512, jetsamMB / 8)
            return max(2_048, jetsamMB - reserveMB)
        }()
        let memoryLimitBytes = memoryLimitMB * 1024 * 1024
        reasons.append("memLimit=\(memoryLimitMB)MB")

        let syncEval = syncEvalForPlan(
            isHighThroughputPhone: highThroughputPhone,
            chipStableAtLongCtx: chipStableAtLongCtx,
            headroomMB: headroomMB
        )

        let dynEnabled = dynamicOpsEnabledForPlan(
            maxOps: maxOps,
            isMac: isMac,
            modelSizeGB: modelSizeGB,
            chipStableAtLongCtx: chipStableAtLongCtx
        )
        let dynFloor = 5
        let dynCtxLow = 4096
        let dynCtxHigh = 12288

        if dynEnabled {
            reasons.append("dynOps=\(maxOps)→\(dynFloor)@ctx\(dynCtxLow/1024)K-\(dynCtxHigh/1024)K")
        }
        reasons.append("chipStable=\(chipStableAtLongCtx ? "yes" : "no")")
        reasons.append("syncEval=\(syncEval ? "on" : "off")")

        let vendoredCBPrefillQMM = vendoredCommandBufferPrefillQMMEnabledForPlan(
            isMac: isMac,
            metalFamilyTier: capability.metalFamilyTier,
            cacheLimitMB: cacheLimitMB
        )
        let fusedGDNDecode = fusedGDNDecodeEnabledForPlan(
            isMac: isMac,
            metalFamilyTier: capability.metalFamilyTier
        )

        reasons.append("vendoredCBPrefillQMM=\(vendoredCBPrefillQMM ? "on" : "off")")
        reasons.append("fusedGDNDecode=\(fusedGDNDecode ? "on" : "off")")
        reasons.append("dsr=\(dsrMaxCritical) prefill=\(prefillStepSize)")

        return Plan(
            maxOpsPerBuffer: maxOps,
            maxMBPerBuffer: maxMB,
            memoryLimitBytes: memoryLimitBytes,
            cacheLimitBytes: cacheLimitBytes,
            forwardPassReserveMB: peakReserveMB,
            kvBudgetMB: kvBudgetMB,
            dsrMaxCritical: dsrMaxCritical,
            quantizeAtTokens: quantizeAt,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: dynEnabled,
            dynamicOpsFloor: dynFloor,
            dynamicOpsCtxLow: dynCtxLow,
            dynamicOpsCtxHigh: dynCtxHigh,
            prefillStepSize: prefillStepSize,
            syncEval: syncEval,
            vendoredCommandBufferPrefillQMMEnabled: vendoredCBPrefillQMM,
            fusedGDNDecodeEnabled: fusedGDNDecode,
            mode: mode,
            reasoning: reasons.joined(separator: " | "),
            measuredBandwidthGBs: effectiveBandwidthGBs,
            memoryIntent: intent
        )
    }

    /// Applies native GPU settings from a memory plan.
    public static func applyGPUSettings(_ plan: Plan) {
        guard MTLCreateSystemDefaultDevice() != nil else { return }

        let nativeConfig = NativeRuntimeBridge.applyMetalConfiguration(
            NativeRuntimeBridge.metalConfiguration(for: plan, contextLengthHint: plan.dsrMaxCritical)
        )

        debugPrint("[MemoryBudgetPlanner] Applied: \(plan.reasoning)")
        debugPrint("[MemoryBudgetPlanner] nativeCache=\(plan.cacheLimitBytes / 1_048_576)MB nativeMaxOps=\(nativeConfig.maxOpsPerCommandBuffer) nativeEffectiveMaxOps=\(nativeConfig.effectiveMaxOpsPerCommandBuffer) nativeMaxMB=\(nativeConfig.maxMBPerCommandBuffer)")
    }

    /// Converts a memory plan into a native KV cache policy.
    public static func toKVPolicy(_ plan: Plan) -> KVCacheMemoryPolicy {
        KVCacheMemoryPolicy(
            mode: plan.mode,
            quantizeAtTokens: plan.quantizeAtTokens,
            kvBits: plan.kvBits,
            kvGroupSize: plan.kvGroupSize,
            maxKVSize: nil,
            maxContextLength: plan.dsrMaxCritical,
            reasoning: plan.reasoning + " | native-dsr=enabled",
            syncEval: plan.syncEval,
            useDSR: true,
            dsrMaxCritical: plan.dsrMaxCritical,
            dsrHeavyBudget: nil,
            dsrRecentBudget: nil,
            dsrScene: .chat,
            memoryIntent: plan.memoryIntent
        )
    }

    static func dynamicOpsEnabledForPlan(
        maxOps: Int,
        isMac: Bool,
        modelSizeGB: Double,
        chipStableAtLongCtx: Bool
    ) -> Bool {
        guard maxOps > 5, !isMac else { return false }
        return true
    }

    static func isHighThroughputPhone(
        effectiveBandwidthGBs _: Double,
        metalFamilyTier: Int
    ) -> Bool {
        metalFamilyTier >= 10
    }

    static func phoneCommandBufferBudget(
        highThroughputPhone: Bool
    ) -> (maxOps: Int, maxMB: Int, peakReserveMB: Int, reason: String) {
        if highThroughputPhone {
            return (20, 10, 50, "phone-planner-v2")
        }
        return (5, 5, 150, "phone-conservative")
    }

    static func prefillStepSizeForPlan(
        isMac: Bool,
        isHighThroughputPhone: Bool,
        headroomMB: Int
    ) -> Int {
        if isMac {
            return 512
        }
        if isHighThroughputPhone {
            return 256
        }
        if headroomMB >= 2000 {
            return 512
        }
        if headroomMB >= 800 {
            return 256
        }
        return 128
    }

    static func syncEvalForPlan(
        isHighThroughputPhone: Bool,
        chipStableAtLongCtx: Bool,
        headroomMB: Int
    ) -> Bool {
        isHighThroughputPhone || (!chipStableAtLongCtx && headroomMB < 512)
    }

    static func dsrMaxCriticalForBudget(
        kvBudgetMB: Int,
        modelSizeGB: Double
    ) -> Int {
        let kvBytesPerTokenINT4 = max(1.0, modelSizeGB * 8.0 * 1024.0)
        let maxAffordableINT4 = Int(Double(kvBudgetMB) * 1024.0 * 1024.0 / kvBytesPerTokenINT4)
        return max(maxAffordableINT4, 2048)
    }

    static func dsrMaxCriticalForPlan(
        kvBudgetMB: Int,
        modelSizeGB: Double,
        totalBudgetMB: Int,
        isMac: Bool,
        isPhone: Bool,
        intent: EdgeMemoryIntent = .balanced
    ) -> Int {
        let budgetDerived = dsrMaxCriticalForBudget(
            kvBudgetMB: kvBudgetMB,
            modelSizeGB: modelSizeGB
        )
        let base: Int
        if isMac || isPhone {
            base = budgetDerived
        } else {
            base = min(budgetDerived, nonPhoneMobileDSRCap(totalBudgetMB: totalBudgetMB))
        }
        return intentAdjustedDSRMaxCritical(
            base: base,
            budgetCeiling: budgetDerived,
            intent: intent
        )
    }

    static func intentAdjustedDSRMaxCritical(
        base: Int,
        budgetCeiling: Int,
        intent: EdgeMemoryIntent
    ) -> Int {
        let floor = 2_048
        switch intent {
        case .balanced:
            return base
        case .longSession, .exactRecall:
            let expanded = min(max(base, base * 2), max(base, budgetCeiling))
            return roundedDSRWindow(expanded, minimum: floor)
        case .batteryFriendly:
            return roundedDSRWindow(max(floor, base / 2), minimum: floor)
        }
    }

    static func roundedDSRWindow(_ value: Int, minimum: Int = 512) -> Int {
        max(minimum, (max(value, minimum) / 512) * 512)
    }

    static func nonPhoneMobileDSRCap(totalBudgetMB: Int) -> Int {
        let baselineJetsamMB = 6_144.0
        let baselineWindow = 13_536.0
        let scaled = baselineWindow * max(1.0, Double(totalBudgetMB) / baselineJetsamMB)
        return max(8_192, Int(scaled.rounded(.down)))
    }

    static func vendoredCommandBufferPrefillQMMEnabledForPlan(
        isMac: Bool,
        metalFamilyTier: Int,
        cacheLimitMB: Int
    ) -> Bool {
        guard cacheLimitMB > 0 else { return false }
        return isMac || metalFamilyTier >= 9
    }

    static func fusedGDNDecodeEnabledForPlan(
        isMac: Bool,
        metalFamilyTier: Int
    ) -> Bool {
        isMac || metalFamilyTier >= 10
    }

    static func quantizedBufferCacheLimitMB(
        totalBudgetMB: Int,
        headroomMB: Int,
        modelSizeGB: Double,
        isPhone: Bool,
        isMac: Bool,
        chipStableAtLongCtx: Bool,
        mode: KVCacheMemoryPolicy.Mode
    ) -> Int {
        func roundUpTo256(_ value: Int) -> Int {
            max(256, ((value + 255) / 256) * 256)
        }

        func roundDownTo256(_ value: Int) -> Int {
            max(256, (value / 256) * 256)
        }

        let estimatedHotQuantizedMB = max(
            256,
            Int((modelSizeGB * 1024.0 * 0.94).rounded(.up)) + 128
        )

        if isMac {
            let targetMB = roundUpTo256(max(estimatedHotQuantizedMB, 4_096))
            let ceilingMB = max(4_096, roundDownTo256(totalBudgetMB / 2))
            return min(targetMB, ceilingMB)
        }

        if isPhone {
            return 64
        }

        if chipStableAtLongCtx {
            let targetMB = roundUpTo256(max(estimatedHotQuantizedMB, 4_096))
            let ceilingMB = max(4_096, roundDownTo256(totalBudgetMB - 256))
            return min(targetMB, ceilingMB)
        }

        if mode == .aggressive {
            let conservativeMB = modelSizeGB <= 4.5 ? 1_024 : 1_536
            let ceilingMB = max(256, roundDownTo256(headroomMB / 2))
            return min(conservativeMB, ceilingMB)
        }

        let targetMB = roundUpTo256(max(estimatedHotQuantizedMB, 2_048))
        let ceilingMB = max(2_048, roundDownTo256(totalBudgetMB - 1_024))
        return min(targetMB, ceilingMB)
    }
}

import Metal
