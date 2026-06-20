// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
import EdgeEngine
@testable import EdgeInference

final class DeviceProfileTests: XCTestCase {
    func testCurrentProfileCarriesEngineHardwareFacts() {
        let profile = DeviceProfile.current

        XCTAssertFalse(profile.machineIdentifier.isEmpty)
        XCTAssertGreaterThanOrEqual(profile.metalFamilyTier, 0)
        XCTAssertEqual(profile.hardwareProfile.machineIdentifier, profile.machineIdentifier)
    }

    func testChipStableDerivesFromApple10MetalFamily() {
        XCTAssertTrue(makeCapability(metalFamilyTier: 10).chipStableAtLongContext)
        XCTAssertFalse(makeCapability(metalFamilyTier: 9).chipStableAtLongContext)
    }

    func testDeviceProfileExposesMeasuredHardwareSignals() {
        let profile = makeProfile(metalFamilyTier: 10)

        XCTAssertEqual(profile.metalFamilyTier, 10)
        XCTAssertEqual(profile.metalDeviceName, "test-metal")
        XCTAssertEqual(profile.machineIdentifier, "test-device")
    }

    private func makeCapability(metalFamilyTier: Int) -> MemoryBudgetPlanner.InferenceCapability {
        MemoryBudgetPlanner.InferenceCapability(
            profile: makeProfile(metalFamilyTier: metalFamilyTier),
            snapshot: DeviceProfile.MemorySnapshot(
                totalPhysicalMB: 8_192,
                availableMB: 4_096,
                footprintMB: 512,
                jetsamLimitMB: 6_144,
                jetsamToPhysicalRatio: 0.75
            ),
            measuredBandwidthGBs: nil
        )
    }

    private func makeProfile(metalFamilyTier: Int) -> DeviceProfile {
        let hardware = DeviceCapabilityProfile(
            machineIdentifier: "test-device",
            cpuBrandString: nil,
            osVersion: "test-os",
            totalPhysicalMemoryMB: 8_192,
            availableMemoryMB: 4_096,
            footprintMB: 512,
            jetsamLimitMB: 6_144,
            thermalState: .nominal,
            metalDeviceName: "test-metal",
            metalFamilyTier: metalFamilyTier,
            recommendedMaxWorkingSetMB: nil,
            measuredBandwidthGBs: nil,
            measuredBandwidthMedianGBs: nil,
            bandwidthSource: .none,
            confidence: .provisional,
            measuredAt: nil
        )
        return DeviceProfile(
            totalRAMGB: 8,
            availableRAMGB: 4,
            thermalState: .nominal,
            hardwareProfile: hardware
        )
    }
}
