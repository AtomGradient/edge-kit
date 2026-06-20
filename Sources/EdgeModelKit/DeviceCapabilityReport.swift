// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

/// SDK-level device capability summary derived from EdgeInference hardware and memory planning.
public struct DeviceCapabilityReport: Sendable, CustomStringConvertible {
    public let profile: DeviceProfile
    public let memorySnapshot: DeviceProfile.MemorySnapshot
    public let modelSizeGB: Double
    public let memoryPlan: MemoryBudgetPlanner.Plan
    public let recommendedTier: ModelTier
    public let availableTiers: [ModelTier]

    public init(
        profile: DeviceProfile,
        memorySnapshot: DeviceProfile.MemorySnapshot,
        modelSizeGB: Double,
        memoryPlan: MemoryBudgetPlanner.Plan
    ) {
        self.profile = profile
        self.memorySnapshot = memorySnapshot
        self.modelSizeGB = modelSizeGB
        self.memoryPlan = memoryPlan
        self.recommendedTier = ModelTierSelector.recommend(for: profile)
        self.availableTiers = ModelTier.allCases.filter { tier in
            ModelConfig.config(for: tier).minRAMGB <= profile.totalRAMGB
        }
    }

    public static func current(
        modelSizeGB: Double,
        measuredBandwidthGBs: Double? = nil
    ) -> DeviceCapabilityReport {
        let profile = DeviceProfile.current
        let snapshot = DeviceProfile.captureMemorySnapshot()
        let plan = MemoryBudgetPlanner.plan(
            profile: profile,
            modelSizeGB: modelSizeGB,
            measuredBandwidthGBs: measuredBandwidthGBs
        )
        return DeviceCapabilityReport(
            profile: profile,
            memorySnapshot: snapshot,
            modelSizeGB: modelSizeGB,
            memoryPlan: plan
        )
    }

    public var summary: String {
        let deviceName = profile.metalDeviceName ?? profile.machineIdentifier
        let tierNames = availableTiers.map(\.rawValue).joined(separator: ", ")
        return [
            "Device: \(deviceName)",
            "RAM: \(memorySnapshot.totalPhysicalMB)MB physical, \(memorySnapshot.jetsamLimitMB)MB jetsam",
            "Model: \(String(format: "%.1f", modelSizeGB))GB",
            "Recommended tier: \(recommendedTier.rawValue)",
            "Available tiers: \(tierNames)",
            "Memory plan: \(memoryPlan.mode.rawValue), KV \(memoryPlan.kvBudgetMB)MB, DSR \(memoryPlan.dsrMaxCritical), prefill \(memoryPlan.prefillStepSize)",
            "Reasoning: \(memoryPlan.reasoning)"
        ].joined(separator: "\n")
    }

    public var description: String {
        summary
    }
}
