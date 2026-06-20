// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Audit-only recorder for A8 memory-policy outcome signals.
///
/// The recorder does not judge answer quality, compare against expected text,
/// execute tools, query fact stores, or route work to mesh peers. It only turns
/// caller-provided generic runtime observations into a Codable audit payload.
public struct MemoryPolicyQualitySignalRecorder: Sendable {
    public enum Status: String, Sendable, Codable, Equatable {
        case disabled
        case recorded
    }

    public struct Observation: Sendable, Codable, Equatable {
        public let sessionFingerprint: String?
        public let turnFingerprint: String?
        public let compactionApplied: Bool
        public let recallAttempted: Bool
        public let recallSucceeded: Bool?
        public let toolRecallAttempted: Bool
        public let toolRecallSucceeded: Bool?
        public let fallbackOccurred: Bool
        public let userCorrectionObserved: Bool
        public let postCompactionProbeCompleted: Bool

        public init(
            sessionFingerprint: String? = nil,
            turnFingerprint: String? = nil,
            compactionApplied: Bool = false,
            recallAttempted: Bool = false,
            recallSucceeded: Bool? = nil,
            toolRecallAttempted: Bool = false,
            toolRecallSucceeded: Bool? = nil,
            fallbackOccurred: Bool = false,
            userCorrectionObserved: Bool = false,
            postCompactionProbeCompleted: Bool = false
        ) {
            self.sessionFingerprint = Self.normalizedFingerprint(sessionFingerprint)
            self.turnFingerprint = Self.normalizedFingerprint(turnFingerprint)
            self.compactionApplied = compactionApplied
            self.recallAttempted = recallAttempted
            self.recallSucceeded = recallSucceeded
            self.toolRecallAttempted = toolRecallAttempted
            self.toolRecallSucceeded = toolRecallSucceeded
            self.fallbackOccurred = fallbackOccurred
            self.userCorrectionObserved = userCorrectionObserved
            self.postCompactionProbeCompleted = postCompactionProbeCompleted
        }

        private static func normalizedFingerprint(_ value: String?) -> String? {
            guard let value else { return nil }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
    }

    public struct Result: Sendable, Equatable {
        public let status: Status
        public let shouldRecordReceipt: Bool
        public let requiresProbe: Bool
        public let audit: Audit
    }

    public struct Audit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let plannerSchemaVersion: String
        public let status: String
        public let memoryIntent: String
        public let qualityLoopEnabled: Bool
        public let observationRecorded: Bool
        public let plannedRequiresPostCompactionProbe: Bool
        public let qualityImprovementClaimAllowed: Bool
        public let sideEffectFree: Bool
        public let toolsExecuted: Bool
        public let toolCallsCreated: Bool
        public let factStoreQueried: Bool
        public let meshRoutingExecuted: Bool
        public let usesRegexOrKeywordFactDetection: Bool
        public let usesExpectedAnswerOrExpectedTool: Bool
        public let sessionFingerprint: String?
        public let turnFingerprint: String?
        public let compactionPlanned: Bool
        public let compactionApplied: Bool
        public let recallModePlanned: String
        public let recallRoutePlanned: String
        public let recallStatusPlanned: String
        public let recallAttempted: Bool
        public let recallSucceeded: Bool?
        public let toolRecallAttempted: Bool
        public let toolRecallSucceeded: Bool?
        public let fallbackOccurred: Bool
        public let userCorrectionObserved: Bool
        public let postCompactionProbeCompleted: Bool
        public let requiresProbe: Bool
        public let reason: String
    }

    public static let schemaVersion = "edge.memory_policy_quality_signal_recorder.v1"

    public init() {}

    public func record(
        memoryPlan: MemoryPolicyPlanner.Plan,
        observation: Observation
    ) -> Result {
        guard memoryPlan.qualityLoop.recordOutcomeSignals else {
            let audit = makeAudit(
                memoryPlan: memoryPlan,
                observation: nil,
                status: .disabled,
                requiresProbe: false,
                reason: "quality signal loop disabled; no outcome observation recorded"
            )
            return Result(
                status: .disabled,
                shouldRecordReceipt: false,
                requiresProbe: false,
                audit: audit
            )
        }

        let requiresProbe = probeRequired(memoryPlan: memoryPlan, observation: observation)
        let audit = makeAudit(
            memoryPlan: memoryPlan,
            observation: observation,
            status: .recorded,
            requiresProbe: requiresProbe,
            reason: requiresProbe
                ? "recorded generic memory-policy outcome signals; follow-up probe required"
                : "recorded generic memory-policy outcome signals"
        )
        return Result(
            status: .recorded,
            shouldRecordReceipt: true,
            requiresProbe: requiresProbe,
            audit: audit
        )
    }

    private func probeRequired(
        memoryPlan: MemoryPolicyPlanner.Plan,
        observation: Observation
    ) -> Bool {
        let plannedProbePending = memoryPlan.qualityLoop.requiresPostCompactionProbe
            && !observation.postCompactionProbeCompleted
        let runtimeCompactionNeedsProbe = observation.compactionApplied
            && !observation.postCompactionProbeCompleted
        let recallFailed = observation.recallAttempted && observation.recallSucceeded == false
        let toolRecallFailed = observation.toolRecallAttempted && observation.toolRecallSucceeded == false
        return plannedProbePending
            || runtimeCompactionNeedsProbe
            || recallFailed
            || toolRecallFailed
            || observation.fallbackOccurred
            || observation.userCorrectionObserved
    }

    private func makeAudit(
        memoryPlan: MemoryPolicyPlanner.Plan,
        observation: Observation?,
        status: Status,
        requiresProbe: Bool,
        reason: String
    ) -> Audit {
        Audit(
            schemaVersion: Self.schemaVersion,
            plannerSchemaVersion: MemoryPolicyPlanner.schemaVersion,
            status: status.rawValue,
            memoryIntent: memoryPlan.intent.rawValue,
            qualityLoopEnabled: memoryPlan.qualityLoop.recordOutcomeSignals,
            observationRecorded: observation != nil,
            plannedRequiresPostCompactionProbe: memoryPlan.qualityLoop.requiresPostCompactionProbe,
            qualityImprovementClaimAllowed: false,
            sideEffectFree: true,
            toolsExecuted: false,
            toolCallsCreated: false,
            factStoreQueried: false,
            meshRoutingExecuted: false,
            usesRegexOrKeywordFactDetection: false,
            usesExpectedAnswerOrExpectedTool: false,
            sessionFingerprint: observation?.sessionFingerprint,
            turnFingerprint: observation?.turnFingerprint,
            compactionPlanned: memoryPlan.compaction.shouldCompactNow,
            compactionApplied: observation?.compactionApplied ?? false,
            recallModePlanned: memoryPlan.recall.mode.rawValue,
            recallRoutePlanned: memoryPlan.recall.route.rawValue,
            recallStatusPlanned: memoryPlan.recall.status.rawValue,
            recallAttempted: observation?.recallAttempted ?? false,
            recallSucceeded: observation?.recallSucceeded,
            toolRecallAttempted: observation?.toolRecallAttempted ?? false,
            toolRecallSucceeded: observation?.toolRecallSucceeded,
            fallbackOccurred: observation?.fallbackOccurred ?? false,
            userCorrectionObserved: observation?.userCorrectionObserved ?? false,
            postCompactionProbeCompleted: observation?.postCompactionProbeCompleted ?? false,
            requiresProbe: requiresProbe,
            reason: reason
        )
    }
}
