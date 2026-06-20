// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class MemoryPolicyMeshExecutionPolicyTests: XCTestCase {
    func testPlannerPolicyNotAvailableFailsClosedWithoutRouting() {
        let memoryPlan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384,
            pairedLongContextNodeAvailable: false,
            meshPolicyAllowsPairedNode: false
        ))

        let decision = MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: memoryPlan,
            candidates: [candidate("node-a")]
        )

        XCTAssertEqual(decision.status, .notApplicable)
        XCTAssertTrue(decision.allowedNodeFingerprints.isEmpty)
        XCTAssertEqual(
            decision.rejectedCandidates,
            [.init(nodeFingerprint: "node-a", reason: .plannerPolicyNotAllowed)]
        )
        XCTAssertFalse(decision.audit.broadMeshRoutingEnabled)
        XCTAssertFalse(decision.audit.meshRoutingExecuted)
        XCTAssertFalse(decision.audit.jointInferenceRequestCreated)
        XCTAssertFalse(decision.audit.modelRequestSent)
        XCTAssertFalse(decision.audit.usesBusinessOrDogfoodFields)
    }

    func testPlannerPolicyDisallowedBlocksTrustedCandidates() {
        let memoryPlan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384,
            pairedLongContextNodeAvailable: true,
            meshPolicyAllowsPairedNode: false
        ))

        let decision = MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: memoryPlan,
            candidates: [candidate("node-a")]
        )

        XCTAssertEqual(decision.status, .blocked)
        XCTAssertEqual(
            decision.rejectedCandidates,
            [.init(nodeFingerprint: "node-a", reason: .plannerPolicyNotAllowed)]
        )
        XCTAssertEqual(decision.audit.plannerMeshStatus, MemoryPolicyPlanner.MeshStatus.blockedByPolicy.rawValue)
    }

    func testAllowedPolicyFiltersGenericCandidates() {
        let decision = MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: allowedMemoryPlan(),
            candidates: [
                candidate("node-b", maxContextTokens: 32_768),
                candidate("node-a", maxContextTokens: 16_384),
                candidate("untrusted", trusted: false, maxContextTokens: 32_768),
                candidate("unsupported", supportsLongContext: false, maxContextTokens: 32_768),
                candidate("thermal", maxContextTokens: 32_768, thermalConstrained: true),
                candidate("small-window", maxContextTokens: 8_192),
                candidate(nil, maxContextTokens: 32_768),
            ],
            requiredContextTokens: 12_000
        )

        XCTAssertEqual(decision.status, MemoryPolicyMeshExecutionPolicy.Status.allowed)
        XCTAssertEqual(decision.allowedNodeFingerprints, ["node-a", "node-b"])
        let expectedRejected: Set<MemoryPolicyMeshExecutionPolicy.RejectedCandidate> = Set([
            .init(nodeFingerprint: nil, reason: .missingFingerprint),
            .init(nodeFingerprint: "small-window", reason: .insufficientContextWindow),
            .init(nodeFingerprint: "thermal", reason: .thermalConstrained),
            .init(nodeFingerprint: "unsupported", reason: .longContextUnsupported),
            .init(nodeFingerprint: "untrusted", reason: .untrusted),
        ])
        XCTAssertEqual(
            Set(decision.rejectedCandidates),
            expectedRejected
        )
        XCTAssertEqual(decision.audit.candidateCount, 7)
        XCTAssertFalse(decision.audit.meshRoutingExecuted)
        XCTAssertFalse(decision.audit.jointInferenceRequestCreated)
        XCTAssertFalse(decision.audit.modelRequestSent)
        XCTAssertFalse(decision.audit.toolsExecuted)
        XCTAssertFalse(decision.audit.factStoreQueried)
    }

    func testAllowedPolicyBlocksWhenNoCandidatePassesPreflight() {
        let decision = MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: allowedMemoryPlan(),
            candidates: [
                candidate("untrusted", trusted: false),
                candidate("unsupported", supportsLongContext: false),
            ]
        )

        XCTAssertEqual(decision.status, .blocked)
        XCTAssertTrue(decision.allowedNodeFingerprints.isEmpty)
        XCTAssertTrue(decision.audit.reason.contains("no paired long-context candidate"))
    }

    func testAuditRoundTripsWithoutExecutionSideEffects() throws {
        let decision = MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: allowedMemoryPlan(),
            candidates: [candidate("node-a")]
        )

        let data = try JSONEncoder().encode(decision.audit)
        let decoded = try JSONDecoder().decode(
            MemoryPolicyMeshExecutionPolicy.Audit.self,
            from: data
        )

        XCTAssertEqual(decoded, decision.audit)
        XCTAssertEqual(decoded.schemaVersion, MemoryPolicyMeshExecutionPolicy.schemaVersion)
        XCTAssertEqual(decoded.plannerSchemaVersion, MemoryPolicyPlanner.schemaVersion)
        XCTAssertFalse(decoded.broadMeshRoutingEnabled)
        XCTAssertFalse(decoded.meshRoutingExecuted)
        XCTAssertFalse(decoded.jointInferenceRequestCreated)
        XCTAssertFalse(decoded.modelRequestSent)
    }

    private func allowedMemoryPlan() -> MemoryPolicyPlanner.Plan {
        MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384,
            pairedLongContextNodeAvailable: true,
            meshPolicyAllowsPairedNode: true
        ))
    }

    private func candidate(
        _ fingerprint: String?,
        trusted: Bool = true,
        supportsLongContext: Bool = true,
        maxContextTokens: Int? = nil,
        thermalConstrained: Bool = false
    ) -> MemoryPolicyMeshExecutionPolicy.Candidate {
        MemoryPolicyMeshExecutionPolicy.Candidate(
            nodeFingerprint: fingerprint,
            trusted: trusted,
            supportsLongContext: supportsLongContext,
            maxContextTokens: maxContextTokens,
            thermalConstrained: thermalConstrained
        )
    }
}
