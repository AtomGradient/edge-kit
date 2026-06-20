// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeEngine

/// Device capability profile used by inference policy decisions.
///
/// Measured observations from an idle app with the increased-memory-limit entitlement:
///
/// | Device                    | hw.memsize | Jetsam limit | Ratio |
/// |--------------------------|------------|-------------|-------|
/// | iPhone 15 PM (A17, 8GB)  | 7.47 GB    | 6.00 GB     | 80.3% |
/// | iPad Air M3 (8GB)        | 7.46 GB    | 8.00 GB     | 107%  |
/// | iPhone Air (A19, 12GB)   | 11.46 GB   | 6.00 GB     | 52.4% |
///
/// Jetsam limits can be lower than physical RAM and can differ across device classes.
/// Runtime policy must use the measured Jetsam limit instead of relying only on `totalRAMGB`.
public struct DeviceProfile: Sendable {

    public let totalRAMGB: Int
    public let availableRAMGB: Int
    public let thermalState: ThermalState
    public let hardwareProfile: DeviceCapabilityProfile

    public enum ThermalState: Sendable {
        case nominal, fair, serious, critical
    }

    public init(
        totalRAMGB: Int,
        availableRAMGB: Int,
        thermalState: ThermalState,
        hardwareProfile: DeviceCapabilityProfile = DeviceCapabilityProfile.current()
    ) {
        self.totalRAMGB = totalRAMGB
        self.availableRAMGB = availableRAMGB
        self.thermalState = thermalState
        self.hardwareProfile = hardwareProfile
    }

    public init(hardwareProfile: DeviceCapabilityProfile) {
        self.init(
            totalRAMGB: max(1, hardwareProfile.totalPhysicalMemoryMB / 1024),
            availableRAMGB: max(0, hardwareProfile.availableMemoryMB / 1024),
            thermalState: Self.thermalState(from: hardwareProfile.thermalState),
            hardwareProfile: hardwareProfile
        )
    }

    public var measuredBandwidthGBs: Double? {
        hardwareProfile.measuredBandwidthGBs
    }

    public var measuredBandwidthSource: DeviceCapabilityProfile.BandwidthSource {
        hardwareProfile.bandwidthSource
    }

    public var metalFamilyTier: Int {
        hardwareProfile.metalFamilyTier
    }

    public var recommendedMaxWorkingSetMB: Int? {
        hardwareProfile.recommendedMaxWorkingSetMB
    }

    public var machineIdentifier: String {
        hardwareProfile.machineIdentifier
    }

    public var metalDeviceName: String? {
        hardwareProfile.metalDeviceName
    }

    public static var current: DeviceProfile {
        let hardware = DeviceCapabilityProfile.current()
        return DeviceProfile(hardwareProfile: hardware)
    }

    /// Approximate memory remaining before Jetsam terminates the process (iOS)
    /// or memory pressure becomes critical (macOS).
    /// Accounts for entitlements like Increased Memory Limit.
    public static func availableBeforeJetsamMB() -> Int {
        DeviceCapabilityProbe.availableBeforeJetsamMB()
    }

    /// Current process physical memory footprint in MB
    public static func physFootprintMB() -> Int {
        DeviceCapabilityProbe.physFootprintMB()
    }

    /// Actual Jetsam limit in MB, derived from `phys_footprint + os_proc_available_memory`.
    ///
    /// This is the total memory budget for the current process before Jetsam termination.
    /// In idle app measurements, footprint plus available memory approximates the system-assigned limit.
    ///
    /// Different devices can share the same Jetsam limit even with different physical RAM:
    /// - iPhone 15 PM (8GB): 6144 MB
    /// - iPhone Air (12GB): 6144 MB
    /// - iPad Air M3 (8GB): 8192 MB
    public static func jetsamLimitMB() -> Int {
        DeviceCapabilityProbe.jetsamLimitMB()
    }

    /// Physical memory reported by `hw.memsize`, in bytes.
    /// The reported value is usually lower than the nominal hardware capacity.
    public static func hwMemsizeBytes() -> UInt64 {
        UInt64(DeviceCapabilityProbe.totalPhysicalMemoryMB()) * 1_048_576
    }

    /// Memory diagnostic snapshot for logging, debugging, and policy decisions.
    public struct MemorySnapshot: Sendable, CustomStringConvertible {
        /// Physical RAM reported by `hw.memsize`, in MB.
        public let totalPhysicalMB: Int
        /// os_proc_available_memory() (MB)
        public let availableMB: Int
        /// phys_footprint (MB)
        public let footprintMB: Int
        /// Derived Jetsam limit, equal to footprint plus available memory, in MB.
        public let jetsamLimitMB: Int
        /// Ratio of Jetsam limit to physical RAM.
        public let jetsamToPhysicalRatio: Double

        public var description: String {
            let lines = [
                "[MemorySnapshot]",
                "  hw.memsize:       \(totalPhysicalMB) MB (\(String(format: "%.2f", Double(totalPhysicalMB) / 1024)) GB)",
                "  available:        \(availableMB) MB (\(String(format: "%.2f", Double(availableMB) / 1024)) GB)",
                "  footprint:        \(footprintMB) MB",
                "  jetsam limit:     \(jetsamLimitMB) MB (\(String(format: "%.2f", Double(jetsamLimitMB) / 1024)) GB)",
                "  jetsam/physical:  \(String(format: "%.1f%%", jetsamToPhysicalRatio * 100))",
            ]
            return lines.joined(separator: "\n")
        }
    }

    /// Captures the current process memory snapshot.
    public static func captureMemorySnapshot() -> MemorySnapshot {
        let totalMB = DeviceCapabilityProbe.totalPhysicalMemoryMB()
        let availMB = availableBeforeJetsamMB()
        let footMB = physFootprintMB()
        let jetsamMB = availMB + footMB
        let ratio = totalMB > 0 ? Double(jetsamMB) / Double(totalMB) : 0

        return MemorySnapshot(
            totalPhysicalMB: totalMB,
            availableMB: availMB,
            footprintMB: footMB,
            jetsamLimitMB: jetsamMB,
            jetsamToPhysicalRatio: ratio
        )
    }

    static func sysctlString(_ key: String) -> String? {
        DeviceCapabilityProbe.sysctlString(key)
    }

    private static func thermalState(from state: DeviceCapabilityProfile.ThermalState) -> ThermalState {
        switch state {
        case .nominal, .unknown: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        }
    }
}
