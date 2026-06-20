// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeEngine

/// Aggregate benchmark snapshot for device-side AI capability.
public struct DeviceBenchmark: Sendable {
    public let profile: DeviceProfile
    public let measuredBandwidthGBs: Double?
    public let timestamp: Date

    init(
        profile: DeviceProfile,
        measuredBandwidthGBs: Double?,
        timestamp: Date
    ) {
        self.profile = profile
        self.measuredBandwidthGBs = measuredBandwidthGBs
        self.timestamp = timestamp
    }

    /// Runs the full device benchmark.
    public static func run() async -> DeviceBenchmark {
        let hardware = await DeviceCapabilityProbe.refreshBenchmark()
        let profile = DeviceProfile(hardwareProfile: hardware)

        return DeviceBenchmark(
            profile: profile,
            measuredBandwidthGBs: hardware.measuredBandwidthGBs,
            timestamp: Date()
        )
    }

    /// Return the current profile plus any cached benchmark from the app sandbox.
    public static func cachedOrCurrent() -> DeviceBenchmark {
        let profile = DeviceProfile.current
        return DeviceBenchmark(
            profile: profile,
            measuredBandwidthGBs: profile.measuredBandwidthGBs,
            timestamp: profile.hardwareProfile.measuredAt ?? Date()
        )
    }

    /// Effective memory bandwidth in GB/s, using measured values only.
    public var effectiveBandwidthGBs: Double {
        measuredBandwidthGBs ?? profile.measuredBandwidthGBs ?? 0
    }
}
