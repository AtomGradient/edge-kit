// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

@available(iOS 17.0, macOS 14.0, *)
public struct MemoryPolicyMeshJointInferenceRuntime: Sendable {
    public enum FailClosedReason: String, Sendable, Codable, Equatable {
        case policyNotAllowed = "policy_not_allowed"
        case missingSelectedNodeFingerprint = "missing_selected_node_fingerprint"
        case selectedNodeFingerprintNotAllowed = "selected_node_fingerprint_not_allowed"
    }

    public struct Result: Sendable, Equatable {
        public let output: String?
        public let audit: Audit

        public init(output: String?, audit: Audit) {
            self.output = output
            self.audit = audit
        }
    }

    public struct Audit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let policySchemaVersion: String
        public let policyStatus: String
        public let selectedNodeFingerprint: String?
        public let allowedNodeFingerprints: [String]
        public let requestID: String
        public let peerID: String?
        public let explicitNodeSelectionRequired: Bool
        public let broadMeshRoutingEnabled: Bool
        public let meshRoutingExecuted: Bool
        public let jointInferenceRequestCreated: Bool
        public let modelRequestSent: Bool
        public let outputReceived: Bool
        public let failClosedReason: FailClosedReason?
        public let reason: String
    }

    public static let schemaVersion = "edge.memory_policy_mesh_joint_inference_runtime.v1"

    private let client: JointInferenceClient

    public init(client: JointInferenceClient) {
        self.client = client
    }

    public func generate(
        decision: MemoryPolicyMeshExecutionPolicy.Decision,
        selectedNodeFingerprint: String?,
        payload: JointInferenceRequestPayload,
        timeoutSeconds: TimeInterval? = nil,
        onEvent: JointInferenceClient.EventHandler? = nil
    ) async throws -> Result {
        let selected = Self.normalizedFingerprint(selectedNodeFingerprint)
        guard decision.status == .allowed else {
            return failClosedResult(
                decision: decision,
                selectedNodeFingerprint: selected,
                payload: payload,
                failClosedReason: .policyNotAllowed,
                reason: "memory policy mesh execution status is \(decision.status.rawValue)"
            )
        }
        guard let selected else {
            return failClosedResult(
                decision: decision,
                selectedNodeFingerprint: nil,
                payload: payload,
                failClosedReason: .missingSelectedNodeFingerprint,
                reason: "selected paired long-context node fingerprint is required"
            )
        }
        guard Set(decision.allowedNodeFingerprints).contains(selected) else {
            return failClosedResult(
                decision: decision,
                selectedNodeFingerprint: selected,
                payload: payload,
                failClosedReason: .selectedNodeFingerprintNotAllowed,
                reason: "selected paired long-context node did not pass memory policy preflight"
            )
        }

        let output = try await client.generate(
            payload,
            timeoutSeconds: timeoutSeconds,
            onEvent: onEvent
        )
        return Result(
            output: output,
            audit: audit(
                decision: decision,
                selectedNodeFingerprint: selected,
                payload: payload,
                jointInferenceRequestCreated: true,
                modelRequestSent: true,
                outputReceived: true,
                failClosedReason: nil,
                reason: "joint inference request sent through explicitly selected paired long-context node"
            )
        )
    }

    private func failClosedResult(
        decision: MemoryPolicyMeshExecutionPolicy.Decision,
        selectedNodeFingerprint: String?,
        payload: JointInferenceRequestPayload,
        failClosedReason: FailClosedReason,
        reason: String
    ) -> Result {
        Result(
            output: nil,
            audit: audit(
                decision: decision,
                selectedNodeFingerprint: selectedNodeFingerprint,
                payload: payload,
                jointInferenceRequestCreated: false,
                modelRequestSent: false,
                outputReceived: false,
                failClosedReason: failClosedReason,
                reason: reason
            )
        )
    }

    private func audit(
        decision: MemoryPolicyMeshExecutionPolicy.Decision,
        selectedNodeFingerprint: String?,
        payload: JointInferenceRequestPayload,
        jointInferenceRequestCreated: Bool,
        modelRequestSent: Bool,
        outputReceived: Bool,
        failClosedReason: FailClosedReason?,
        reason: String
    ) -> Audit {
        Audit(
            schemaVersion: Self.schemaVersion,
            policySchemaVersion: MemoryPolicyMeshExecutionPolicy.schemaVersion,
            policyStatus: decision.status.rawValue,
            selectedNodeFingerprint: selectedNodeFingerprint,
            allowedNodeFingerprints: decision.allowedNodeFingerprints,
            requestID: payload.requestID,
            peerID: payload.peerID,
            explicitNodeSelectionRequired: true,
            broadMeshRoutingEnabled: false,
            meshRoutingExecuted: false,
            jointInferenceRequestCreated: jointInferenceRequestCreated,
            modelRequestSent: modelRequestSent,
            outputReceived: outputReceived,
            failClosedReason: failClosedReason,
            reason: reason
        )
    }

    private static func normalizedFingerprint(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
