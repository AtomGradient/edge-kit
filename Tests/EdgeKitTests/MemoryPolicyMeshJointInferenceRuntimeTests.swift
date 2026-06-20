// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
@testable import EdgeMesh
import XCTest

final class MemoryPolicyMeshJointInferenceRuntimeTests: XCTestCase {
    @available(iOS 17.0, macOS 14.0, *)
    final class ClientHarness: @unchecked Sendable {
        private let lock = NSLock()
        private var _handler: JointInferenceClient.EventHandler?
        private var _sent: [JointInferenceRequestPayload] = []
        private var _events: [JointInferenceEventType] = []

        var sent: [JointInferenceRequestPayload] {
            lock.lock(); defer { lock.unlock() }
            return _sent
        }

        var events: [JointInferenceEventType] {
            lock.lock(); defer { lock.unlock() }
            return _events
        }

        func register(_ handler: @escaping JointInferenceClient.EventHandler) {
            lock.lock(); defer { lock.unlock() }
            _handler = handler
        }

        func send(_ payload: JointInferenceRequestPayload) {
            lock.lock(); defer { lock.unlock() }
            _sent.append(payload)
        }

        func recordEvent(_ event: JointInferenceEventPayload) {
            lock.lock(); defer { lock.unlock() }
            _events.append(event.type)
        }

        func emit(_ event: JointInferenceEventPayload) {
            lock.lock()
            let handler = _handler
            lock.unlock()
            handler?(event)
        }
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testBlockedPolicyFailsClosedWithoutSendingRequest() async throws {
        let harness = ClientHarness()
        let runtime = runtime(harness: harness)
        let result = try await runtime.generate(
            decision: blockedDecision(),
            selectedNodeFingerprint: "node-a",
            payload: payload(requestID: "blocked-policy")
        )

        XCTAssertNil(result.output)
        XCTAssertTrue(harness.sent.isEmpty)
        XCTAssertEqual(result.audit.failClosedReason, .policyNotAllowed)
        XCTAssertFalse(result.audit.jointInferenceRequestCreated)
        XCTAssertFalse(result.audit.modelRequestSent)
        XCTAssertFalse(result.audit.broadMeshRoutingEnabled)
        XCTAssertFalse(result.audit.meshRoutingExecuted)
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testNotApplicablePolicyFailsClosedWithoutSendingRequest() async throws {
        let harness = ClientHarness()
        let runtime = runtime(harness: harness)
        let result = try await runtime.generate(
            decision: notApplicableDecision(),
            selectedNodeFingerprint: "node-a",
            payload: payload(requestID: "not-applicable-policy")
        )

        XCTAssertNil(result.output)
        XCTAssertTrue(harness.sent.isEmpty)
        XCTAssertEqual(result.audit.policyStatus, MemoryPolicyMeshExecutionPolicy.Status.notApplicable.rawValue)
        XCTAssertEqual(result.audit.failClosedReason, .policyNotAllowed)
        XCTAssertFalse(result.audit.modelRequestSent)
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testWrongSelectedFingerprintFailsClosedWithoutSendingRequest() async throws {
        let harness = ClientHarness()
        let runtime = runtime(harness: harness)
        let result = try await runtime.generate(
            decision: allowedDecision(),
            selectedNodeFingerprint: "node-b",
            payload: payload(requestID: "wrong-fingerprint")
        )

        XCTAssertNil(result.output)
        XCTAssertTrue(harness.sent.isEmpty)
        XCTAssertEqual(result.audit.failClosedReason, .selectedNodeFingerprintNotAllowed)
        XCTAssertEqual(result.audit.selectedNodeFingerprint, "node-b")
        XCTAssertEqual(result.audit.allowedNodeFingerprints, ["node-a"])
        XCTAssertFalse(result.audit.jointInferenceRequestCreated)
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testAllowedPolicySendsExactlyOneJointInferencePayload() async throws {
        let harness = ClientHarness()
        let runtime = runtime(harness: harness)
        let request = payload(requestID: "allowed-runtime", peerID: "paired-peer")

        let task = Task {
            try await runtime.generate(
                decision: allowedDecision(),
                selectedNodeFingerprint: " node-a ",
                payload: request,
                timeoutSeconds: 2
            ) { event in
                harness.recordEvent(event)
            }
        }
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(harness.sent, [request])
        harness.emit(.init(requestID: "allowed-runtime", type: .accepted, sequence: 0))
        harness.emit(.init(requestID: "allowed-runtime", type: .complete, sequence: 1, fullText: "mesh output"))

        let result = try await task.value
        XCTAssertEqual(result.output, "mesh output")
        XCTAssertEqual(harness.sent.count, 1)
        XCTAssertEqual(harness.events, [.accepted, .complete])
        XCTAssertNil(result.audit.failClosedReason)
        XCTAssertEqual(result.audit.selectedNodeFingerprint, "node-a")
        XCTAssertEqual(result.audit.requestID, "allowed-runtime")
        XCTAssertEqual(result.audit.peerID, "paired-peer")
        XCTAssertTrue(result.audit.explicitNodeSelectionRequired)
        XCTAssertFalse(result.audit.broadMeshRoutingEnabled)
        XCTAssertFalse(result.audit.meshRoutingExecuted)
        XCTAssertTrue(result.audit.jointInferenceRequestCreated)
        XCTAssertTrue(result.audit.modelRequestSent)
        XCTAssertTrue(result.audit.outputReceived)
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testAuditCodableRoundTrips() async throws {
        let harness = ClientHarness()
        let runtime = runtime(harness: harness)
        let result = try await runtime.generate(
            decision: allowedDecision(),
            selectedNodeFingerprint: nil,
            payload: payload(requestID: "audit-roundtrip")
        )

        let data = try JSONEncoder().encode(result.audit)
        let decoded = try JSONDecoder().decode(
            MemoryPolicyMeshJointInferenceRuntime.Audit.self,
            from: data
        )

        XCTAssertEqual(decoded, result.audit)
        XCTAssertEqual(decoded.schemaVersion, MemoryPolicyMeshJointInferenceRuntime.schemaVersion)
        XCTAssertEqual(decoded.policySchemaVersion, MemoryPolicyMeshExecutionPolicy.schemaVersion)
        XCTAssertEqual(decoded.failClosedReason, .missingSelectedNodeFingerprint)
        XCTAssertTrue(harness.sent.isEmpty)
    }

    @available(iOS 17.0, macOS 14.0, *)
    private func runtime(harness: ClientHarness) -> MemoryPolicyMeshJointInferenceRuntime {
        MemoryPolicyMeshJointInferenceRuntime(client: JointInferenceClient(
            configuration: .init(timeoutSeconds: 2),
            sendRequest: { payload in harness.send(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        ))
    }

    private func allowedDecision() -> MemoryPolicyMeshExecutionPolicy.Decision {
        MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: meshMemoryPlan(pairedNodeAvailable: true, policyAllowed: true),
            candidates: [candidate("node-a")],
            requiredContextTokens: 12_000
        )
    }

    private func blockedDecision() -> MemoryPolicyMeshExecutionPolicy.Decision {
        MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: meshMemoryPlan(pairedNodeAvailable: true, policyAllowed: false),
            candidates: [candidate("node-a")]
        )
    }

    private func notApplicableDecision() -> MemoryPolicyMeshExecutionPolicy.Decision {
        MemoryPolicyMeshExecutionPolicy().evaluate(
            memoryPlan: meshMemoryPlan(pairedNodeAvailable: false, policyAllowed: false),
            candidates: [candidate("node-a")]
        )
    }

    private func meshMemoryPlan(
        pairedNodeAvailable: Bool,
        policyAllowed: Bool
    ) -> MemoryPolicyPlanner.Plan {
        MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 10_000,
            modelContextLimitTokens: 16_384,
            pairedLongContextNodeAvailable: pairedNodeAvailable,
            meshPolicyAllowsPairedNode: policyAllowed
        ))
    }

    private func candidate(_ fingerprint: String) -> MemoryPolicyMeshExecutionPolicy.Candidate {
        MemoryPolicyMeshExecutionPolicy.Candidate(
            nodeFingerprint: fingerprint,
            trusted: true,
            supportsLongContext: true,
            maxContextTokens: 32_768
        )
    }

    private func payload(
        requestID: String,
        peerID: String? = nil
    ) -> JointInferenceRequestPayload {
        JointInferenceRequestPayload(
            requestID: requestID,
            peerID: peerID,
            modelID: "neutral-long-context-model",
            messages: [JointInferenceMessage(role: "user", content: "hello")],
            maxTokens: 32,
            temperature: 0.2,
            enableThinking: false,
            useNeuralImprint: true,
            routeReason: "memory_policy_mesh_runtime_test"
        )
    }
}
