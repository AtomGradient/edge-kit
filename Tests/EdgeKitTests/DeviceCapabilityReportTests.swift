// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import XCTest
@testable import EdgeInference
@testable import EdgeModelKit

final class DeviceCapabilityReportTests: XCTestCase {
    func testReportIncludesTierAndMemoryPlanSummary() {
        let report = DeviceCapabilityReport(
            profile: makeProfile(totalRAMGB: 8, availableRAMGB: 6),
            memorySnapshot: DeviceProfile.MemorySnapshot(
                totalPhysicalMB: 8_192,
                availableMB: 4_096,
                footprintMB: 512,
                jetsamLimitMB: 6_144,
                jetsamToPhysicalRatio: 0.75
            ),
            modelSizeGB: 3.5,
            memoryPlan: makePlan()
        )

        XCTAssertEqual(report.recommendedTier, .pro)
        XCTAssertTrue(report.availableTiers.contains(.standard))
        XCTAssertTrue(report.summary.contains("Recommended tier: pro"))
        XCTAssertTrue(report.summary.contains("Memory plan: auto"))
        XCTAssertTrue(report.summary.contains("DSR 3072"))
    }

    private func makeProfile(totalRAMGB: Int, availableRAMGB: Int) -> DeviceProfile {
        let hardware = DeviceCapabilityProfile(
            machineIdentifier: "test-device",
            cpuBrandString: nil,
            osVersion: "test-os",
            totalPhysicalMemoryMB: totalRAMGB * 1024,
            availableMemoryMB: availableRAMGB * 1024,
            footprintMB: 512,
            jetsamLimitMB: 6_144,
            thermalState: .nominal,
            metalDeviceName: "test-metal",
            metalFamilyTier: 10,
            recommendedMaxWorkingSetMB: nil,
            measuredBandwidthGBs: 100,
            measuredBandwidthMedianGBs: nil,
            bandwidthSource: .measured,
            confidence: .measured,
            measuredAt: nil
        )
        return DeviceProfile(
            totalRAMGB: totalRAMGB,
            availableRAMGB: availableRAMGB,
            thermalState: .nominal,
            hardwareProfile: hardware
        )
    }

    private func makePlan() -> MemoryBudgetPlanner.Plan {
        MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 15,
            maxMBPerBuffer: 15,
            memoryLimitBytes: 6 * 1_073_741_824,
            cacheLimitBytes: 256 * 1_048_576,
            forwardPassReserveMB: 300,
            kvBudgetMB: 1200,
            dsrMaxCritical: 3072,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 256,
            syncEval: false,
            vendoredCommandBufferPrefillQMMEnabled: true,
            fusedGDNDecodeEnabled: true,
            mode: .auto,
            reasoning: "fixture",
            measuredBandwidthGBs: 100
        )
    }
}
