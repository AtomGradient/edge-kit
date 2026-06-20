// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Side-effect-free mesh execution preflight for A8 memory policy.
///
/// This policy only decides whether later runtime layers may consider paired
/// long-context candidates. It does not select a route, create joint-inference
/// requests, send model work, or depend on EdgeMesh types.
public struct MemoryPolicyMeshExecutionPolicy: Sendable {
    public enum Status: String, Sendable, Codable, Equatable {
        case notApplicable
        case blocked
        case allowed
    }

    public enum RejectionReason: String, Sendable, Codable, Equatable {
        case plannerPolicyNotAllowed
        case missingFingerprint
        case untrusted
        case longContextUnsupported
        case thermalConstrained
        case insufficientContextWindow
    }

    public struct Candidate: Sendable, Codable, Equatable {
        public let nodeFingerprint: String?
        public let trusted: Bool
        public let supportsLongContext: Bool
        public let maxContextTokens: Int?
        public let thermalConstrained: Bool

        public init(
            nodeFingerprint: String?,
            trusted: Bool,
            supportsLongContext: Bool,
            maxContextTokens: Int? = nil,
            thermalConstrained: Bool = false
        ) {
            self.nodeFingerprint = Self.normalizedFingerprint(nodeFingerprint)
            self.trusted = trusted
            self.supportsLongContext = supportsLongContext
            self.maxContextTokens = maxContextTokens.map { max(0, $0) }
            self.thermalConstrained = thermalConstrained
        }

        private static func normalizedFingerprint(_ value: String?) -> String? {
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
    }

    public struct RejectedCandidate: Sendable, Codable, Equatable, Hashable {
        public let nodeFingerprint: String?
        public let reason: RejectionReason
    }

    public struct Decision: Sendable, Equatable {
        public let status: Status
        public let allowedNodeFingerprints: [String]
        public let rejectedCandidates: [RejectedCandidate]
        public let audit: Audit
    }

    public struct Audit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let plannerSchemaVersion: String
        public let plannerMeshStatus: String
        public let status: String
        public let candidateCount: Int
        public let allowedNodeFingerprints: [String]
        public let rejectedCandidates: [RejectedCandidate]
        public let sideEffectFree: Bool
        public let broadMeshRoutingEnabled: Bool
        public let meshRoutingExecuted: Bool
        public let jointInferenceRequestCreated: Bool
        public let modelRequestSent: Bool
        public let toolsExecuted: Bool
        public let factStoreQueried: Bool
        public let usesBusinessOrDogfoodFields: Bool
        public let reason: String
    }

    public static let schemaVersion = "edge.memory_policy_mesh_execution_policy.v1"

    public init() {}

    public func evaluate(
        memoryPlan: MemoryPolicyPlanner.Plan,
        candidates: [Candidate],
        requiredContextTokens: Int? = nil
    ) -> Decision {
        guard memoryPlan.mesh.status == .allowedByPolicy else {
            let rejected = candidates.map {
                RejectedCandidate(
                    nodeFingerprint: $0.nodeFingerprint,
                    reason: .plannerPolicyNotAllowed
                )
            }
            return makeDecision(
                memoryPlan: memoryPlan,
                candidates: candidates,
                status: memoryPlan.mesh.status == .notAvailable ? .notApplicable : .blocked,
                allowedNodeFingerprints: [],
                rejectedCandidates: rejected,
                reason: memoryPlan.mesh.reason
            )
        }

        let minimumContext = requiredContextTokens.map { max(0, $0) }
        var allowed: [String] = []
        var rejected: [RejectedCandidate] = []

        for candidate in candidates {
            if let reason = rejectionReason(for: candidate, requiredContextTokens: minimumContext) {
                rejected.append(.init(nodeFingerprint: candidate.nodeFingerprint, reason: reason))
            } else if let fingerprint = candidate.nodeFingerprint {
                allowed.append(fingerprint)
            }
        }

        if allowed.isEmpty {
            return makeDecision(
                memoryPlan: memoryPlan,
                candidates: candidates,
                status: .blocked,
                allowedNodeFingerprints: [],
                rejectedCandidates: rejected,
                reason: "mesh execution policy allowed by planner, but no paired long-context candidate passed preflight"
            )
        }

        return makeDecision(
            memoryPlan: memoryPlan,
            candidates: candidates,
            status: .allowed,
            allowedNodeFingerprints: allowed,
            rejectedCandidates: rejected,
            reason: "paired long-context candidates passed side-effect-free mesh execution preflight"
        )
    }

    private func rejectionReason(
        for candidate: Candidate,
        requiredContextTokens: Int?
    ) -> RejectionReason? {
        guard candidate.nodeFingerprint != nil else {
            return .missingFingerprint
        }
        guard candidate.trusted else {
            return .untrusted
        }
        guard candidate.supportsLongContext else {
            return .longContextUnsupported
        }
        guard !candidate.thermalConstrained else {
            return .thermalConstrained
        }
        if let requiredContextTokens,
           let maxContextTokens = candidate.maxContextTokens,
           maxContextTokens < requiredContextTokens {
            return .insufficientContextWindow
        }
        return nil
    }

    private func makeDecision(
        memoryPlan: MemoryPolicyPlanner.Plan,
        candidates: [Candidate],
        status: Status,
        allowedNodeFingerprints: [String],
        rejectedCandidates: [RejectedCandidate],
        reason: String
    ) -> Decision {
        let sortedAllowed = allowedNodeFingerprints.sorted()
        let sortedRejected = rejectedCandidates.sorted {
            let left = $0.nodeFingerprint ?? ""
            let right = $1.nodeFingerprint ?? ""
            if left == right {
                return $0.reason.rawValue < $1.reason.rawValue
            }
            return left < right
        }
        return Decision(
            status: status,
            allowedNodeFingerprints: sortedAllowed,
            rejectedCandidates: sortedRejected,
            audit: Audit(
                schemaVersion: Self.schemaVersion,
                plannerSchemaVersion: MemoryPolicyPlanner.schemaVersion,
                plannerMeshStatus: memoryPlan.mesh.status.rawValue,
                status: status.rawValue,
                candidateCount: candidates.count,
                allowedNodeFingerprints: sortedAllowed,
                rejectedCandidates: sortedRejected,
                sideEffectFree: true,
                broadMeshRoutingEnabled: false,
                meshRoutingExecuted: false,
                jointInferenceRequestCreated: false,
                modelRequestSent: false,
                toolsExecuted: false,
                factStoreQueried: false,
                usesBusinessOrDogfoodFields: false,
                reason: reason
            )
        )
    }
}
