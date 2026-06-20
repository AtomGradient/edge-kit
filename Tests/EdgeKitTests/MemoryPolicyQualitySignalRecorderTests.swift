// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class MemoryPolicyQualitySignalRecorderTests: XCTestCase {
    func testDisabledLoopFailsClosedWithoutRecordingObservation() {
        let plan = memoryPlan(qualitySignalsEnabled: false)
        let observation = MemoryPolicyQualitySignalRecorder.Observation(
            sessionFingerprint: " session-fp ",
            turnFingerprint: "turn-fp",
            compactionApplied: true,
            recallAttempted: true,
            recallSucceeded: false,
            toolRecallAttempted: true,
            toolRecallSucceeded: false,
            fallbackOccurred: true,
            userCorrectionObserved: true
        )

        let result = MemoryPolicyQualitySignalRecorder().record(
            memoryPlan: plan,
            observation: observation
        )

        XCTAssertEqual(result.status, .disabled)
        XCTAssertFalse(result.shouldRecordReceipt)
        XCTAssertFalse(result.requiresProbe)
        XCTAssertFalse(result.audit.observationRecorded)
        XCTAssertNil(result.audit.sessionFingerprint)
        XCTAssertNil(result.audit.turnFingerprint)
        XCTAssertFalse(result.audit.compactionApplied)
        XCTAssertFalse(result.audit.recallAttempted)
        XCTAssertNil(result.audit.recallSucceeded)
        XCTAssertFalse(result.audit.toolRecallAttempted)
        XCTAssertNil(result.audit.toolRecallSucceeded)
        XCTAssertFalse(result.audit.fallbackOccurred)
        XCTAssertFalse(result.audit.userCorrectionObserved)
        XCTAssertFalse(result.audit.qualityImprovementClaimAllowed)
        XCTAssertFalse(result.audit.toolsExecuted)
        XCTAssertFalse(result.audit.toolCallsCreated)
        XCTAssertFalse(result.audit.factStoreQueried)
        XCTAssertFalse(result.audit.meshRoutingExecuted)
        XCTAssertFalse(result.audit.usesExpectedAnswerOrExpectedTool)
        XCTAssertFalse(result.audit.usesRegexOrKeywordFactDetection)
    }

    func testEnabledLoopRecordsGenericCompactionAndRecallSignals() {
        let plan = memoryPlan(
            qualitySignalsEnabled: true,
            hasAuditableFactRequirement: true,
            toolRecallAvailable: true
        )
        let observation = MemoryPolicyQualitySignalRecorder.Observation(
            sessionFingerprint: "session-fp",
            turnFingerprint: "turn-fp",
            compactionApplied: false,
            recallAttempted: true,
            recallSucceeded: true,
            toolRecallAttempted: true,
            toolRecallSucceeded: true,
            postCompactionProbeCompleted: true
        )

        let result = MemoryPolicyQualitySignalRecorder().record(
            memoryPlan: plan,
            observation: observation
        )

        XCTAssertEqual(result.status, .recorded)
        XCTAssertTrue(result.shouldRecordReceipt)
        XCTAssertFalse(result.requiresProbe)
        XCTAssertTrue(result.audit.observationRecorded)
        XCTAssertEqual(result.audit.sessionFingerprint, "session-fp")
        XCTAssertEqual(result.audit.turnFingerprint, "turn-fp")
        XCTAssertEqual(result.audit.recallRoutePlanned, MemoryPolicyPlanner.RecallRoute.tool.rawValue)
        XCTAssertEqual(result.audit.recallStatusPlanned, MemoryPolicyPlanner.RecallStatus.ready.rawValue)
        XCTAssertTrue(result.audit.recallAttempted)
        XCTAssertEqual(result.audit.recallSucceeded, true)
        XCTAssertTrue(result.audit.toolRecallAttempted)
        XCTAssertEqual(result.audit.toolRecallSucceeded, true)
        XCTAssertFalse(result.audit.qualityImprovementClaimAllowed)
        XCTAssertFalse(result.audit.toolsExecuted)
        XCTAssertFalse(result.audit.factStoreQueried)
        XCTAssertFalse(result.audit.meshRoutingExecuted)
    }

    func testCorrectionFallbackAndFailedRecallRequireProbe() {
        let plan = memoryPlan(qualitySignalsEnabled: true)
        let observation = MemoryPolicyQualitySignalRecorder.Observation(
            recallAttempted: true,
            recallSucceeded: false,
            fallbackOccurred: true,
            userCorrectionObserved: true
        )

        let result = MemoryPolicyQualitySignalRecorder().record(
            memoryPlan: plan,
            observation: observation
        )

        XCTAssertTrue(result.requiresProbe)
        XCTAssertTrue(result.audit.requiresProbe)
        XCTAssertTrue(result.audit.fallbackOccurred)
        XCTAssertTrue(result.audit.userCorrectionObserved)
        XCTAssertTrue(result.audit.reason.contains("follow-up probe required"))
    }

    func testCompletedPostCompactionProbeSatisfiesPlannedProbe() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 12_000,
            modelContextLimitTokens: 16_384,
            compactionCount: 1,
            qualitySignalsEnabled: true
        ))
        let observation = MemoryPolicyQualitySignalRecorder.Observation(
            compactionApplied: true,
            postCompactionProbeCompleted: true
        )

        let result = MemoryPolicyQualitySignalRecorder().record(
            memoryPlan: plan,
            observation: observation
        )

        XCTAssertTrue(plan.qualityLoop.requiresPostCompactionProbe)
        XCTAssertFalse(result.requiresProbe)
        XCTAssertTrue(result.audit.plannedRequiresPostCompactionProbe)
        XCTAssertTrue(result.audit.compactionApplied)
        XCTAssertTrue(result.audit.postCompactionProbeCompleted)
    }

    func testAuditRoundTripsAsReceipt() throws {
        let result = MemoryPolicyQualitySignalRecorder().record(
            memoryPlan: memoryPlan(qualitySignalsEnabled: true),
            observation: .init(
                sessionFingerprint: "session-fp",
                turnFingerprint: "turn-fp",
                fallbackOccurred: true
            )
        )

        let data = try JSONEncoder().encode(result.audit)
        let decoded = try JSONDecoder().decode(
            MemoryPolicyQualitySignalRecorder.Audit.self,
            from: data
        )

        XCTAssertEqual(decoded, result.audit)
        XCTAssertEqual(decoded.schemaVersion, MemoryPolicyQualitySignalRecorder.schemaVersion)
        XCTAssertEqual(decoded.plannerSchemaVersion, MemoryPolicyPlanner.schemaVersion)
        XCTAssertFalse(decoded.qualityImprovementClaimAllowed)
        XCTAssertFalse(decoded.usesExpectedAnswerOrExpectedTool)
    }

    private func memoryPlan(
        qualitySignalsEnabled: Bool,
        hasAuditableFactRequirement: Bool = false,
        toolRecallAvailable: Bool = false
    ) -> MemoryPolicyPlanner.Plan {
        MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 2_000,
            modelContextLimitTokens: 4_096,
            hasAuditableFactRequirement: hasAuditableFactRequirement,
            factStoreAvailable: false,
            toolRecallAvailable: toolRecallAvailable,
            qualitySignalsEnabled: qualitySignalsEnabled
        ))
    }
}
