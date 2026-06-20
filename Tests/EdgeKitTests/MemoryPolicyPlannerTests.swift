// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class MemoryPolicyPlannerTests: XCTestCase {
    func testExactRecallRequiresAuditableFactRecallWhenAvailable() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 12_000,
            modelContextLimitTokens: 16_384,
            hasAuditableFactRequirement: true,
            factStoreAvailable: true,
            toolRecallAvailable: true
        ))

        XCTAssertEqual(plan.recall.mode, .required)
        XCTAssertEqual(plan.recall.route, .factStoreAndTool)
        XCTAssertEqual(plan.recall.status, .ready)
        XCTAssertTrue(plan.audit.usesExplicitAuditableFactSignal)
        XCTAssertFalse(plan.audit.usesRegexOrKeywordFactDetection)
        XCTAssertFalse(plan.audit.qualityImprovementClaimAllowed)
    }

    func testExactRecallBlocksWhenNoAuditableRecallBackendExists() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 8_000,
            modelContextLimitTokens: 16_384,
            hasAuditableFactRequirement: true,
            factStoreAvailable: false,
            toolRecallAvailable: false
        ))

        XCTAssertEqual(plan.recall.mode, .required)
        XCTAssertEqual(plan.recall.route, .none)
        XCTAssertEqual(plan.recall.status, .blocked)
        XCTAssertTrue(plan.recall.reason.contains("raw context and summary are not sufficient"))
    }

    func testBatteryFriendlyCompactsEarlierWithSmallerRecentWindow() {
        let balanced = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384
        ))
        let longSession = MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384
        ))
        let battery = MemoryPolicyPlanner.plan(signals: .init(
            intent: .batteryFriendly,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384
        ))

        XCTAssertLessThan(battery.compaction.triggerTokenCount, balanced.compaction.triggerTokenCount)
        XCTAssertLessThan(battery.compaction.recentWindowTokens, balanced.compaction.recentWindowTokens)
        XCTAssertLessThan(balanced.compaction.triggerTokenCount, longSession.compaction.triggerTokenCount)
        XCTAssertEqual(battery.compaction.density, .compact)
        XCTAssertTrue(battery.compaction.shouldCompactNow)
    }

    func testPairedLongContextNodeIsBlockedWithoutExplicitMeshPolicy() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 14_000,
            modelContextLimitTokens: 16_384,
            pairedLongContextNodeAvailable: true,
            meshPolicyAllowsPairedNode: false
        ))

        XCTAssertEqual(plan.mesh.status, .blockedByPolicy)
        XCTAssertTrue(plan.mesh.reason.contains("explicit mesh policy is disabled"))
        XCTAssertFalse(plan.audit.broadMeshRoutingEnabled)
    }

    func testLowMemoryHeadroomTightensCompactionTrigger() {
        let normal = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 7_000,
            modelContextLimitTokens: 16_384,
            availableMemoryMB: 2_048
        ))
        let tight = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 7_000,
            modelContextLimitTokens: 16_384,
            availableMemoryMB: 768
        ))

        XCTAssertLessThan(tight.compaction.triggerTokenCount, normal.compaction.triggerTokenCount)
    }

    func testQualityLoopRecordsSignalsWithoutQualityClaim() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 12_000,
            modelContextLimitTokens: 16_384,
            compactionCount: 1,
            qualitySignalsEnabled: true
        ))

        XCTAssertTrue(plan.qualityLoop.recordOutcomeSignals)
        XCTAssertTrue(plan.qualityLoop.requiresPostCompactionProbe)
        XCTAssertFalse(plan.qualityLoop.allowsQualityClaim)
        XCTAssertTrue(plan.audit.sideEffectFree)
        XCTAssertTrue(plan.audit.complementsMemoryBudgetPlanner)
    }
}
