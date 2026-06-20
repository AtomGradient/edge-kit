// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Phase-2 memory policy planner.
///
/// This planner complements `MemoryBudgetPlanner`: `MemoryBudgetPlanner` owns
/// KV/DSR/cache budgets, while `MemoryPolicyPlanner` owns higher-level
/// compaction, fact-store/tool recall, mesh-adaptation, and quality-loop
/// planning. It is intentionally side-effect free.
public struct MemoryPolicyPlanner: Sendable {
    public struct Signals: Sendable {
        public let intent: EdgeMemoryIntent
        public let contextTokenCount: Int
        public let modelContextLimitTokens: Int
        public let availableMemoryMB: Int?
        public let compactionCount: Int
        public let hasAuditableFactRequirement: Bool
        public let factStoreAvailable: Bool
        public let toolRecallAvailable: Bool
        public let pairedLongContextNodeAvailable: Bool
        public let meshPolicyAllowsPairedNode: Bool
        public let qualitySignalsEnabled: Bool

        public init(
            intent: EdgeMemoryIntent = .balanced,
            contextTokenCount: Int,
            modelContextLimitTokens: Int,
            availableMemoryMB: Int? = nil,
            compactionCount: Int = 0,
            hasAuditableFactRequirement: Bool = false,
            factStoreAvailable: Bool = false,
            toolRecallAvailable: Bool = false,
            pairedLongContextNodeAvailable: Bool = false,
            meshPolicyAllowsPairedNode: Bool = false,
            qualitySignalsEnabled: Bool = false
        ) {
            self.intent = intent
            self.contextTokenCount = max(0, contextTokenCount)
            self.modelContextLimitTokens = max(2_048, modelContextLimitTokens)
            self.availableMemoryMB = availableMemoryMB.map { max(0, $0) }
            self.compactionCount = max(0, compactionCount)
            self.hasAuditableFactRequirement = hasAuditableFactRequirement
            self.factStoreAvailable = factStoreAvailable
            self.toolRecallAvailable = toolRecallAvailable
            self.pairedLongContextNodeAvailable = pairedLongContextNodeAvailable
            self.meshPolicyAllowsPairedNode = meshPolicyAllowsPairedNode
            self.qualitySignalsEnabled = qualitySignalsEnabled
        }
    }

    public struct Plan: Sendable {
        public let intent: EdgeMemoryIntent
        public let compaction: CompactionPlan
        public let recall: RecallPlan
        public let mesh: MeshPlan
        public let qualityLoop: QualityLoopPlan
        public let audit: Audit
    }

    public struct CompactionPlan: Sendable {
        public let triggerTokenCount: Int
        public let shouldCompactNow: Bool
        public let recentWindowTokens: Int
        public let summaryBudgetTokens: Int
        public let density: SummaryDensity
        public let reason: String
    }

    public enum SummaryDensity: String, Sendable {
        case compact
        case balanced
        case detailed
    }

    public struct RecallPlan: Sendable {
        public let mode: RecallMode
        public let route: RecallRoute
        public let status: RecallStatus
        public let reason: String
    }

    public enum RecallMode: String, Sendable {
        case none
        case opportunistic
        case required
    }

    public enum RecallRoute: String, Sendable {
        case none
        case factStore
        case tool
        case factStoreAndTool
    }

    public enum RecallStatus: String, Sendable {
        case ready
        case blocked
        case notApplicable
    }

    public struct MeshPlan: Sendable {
        public let status: MeshStatus
        public let reason: String
    }

    public enum MeshStatus: String, Sendable {
        case notAvailable
        case blockedByPolicy
        case allowedByPolicy
    }

    public struct QualityLoopPlan: Sendable {
        public let recordOutcomeSignals: Bool
        public let requiresPostCompactionProbe: Bool
        public let allowsQualityClaim: Bool
        public let reason: String
    }

    public struct Audit: Sendable {
        public let schemaVersion: String
        public let complementsMemoryBudgetPlanner: Bool
        public let sideEffectFree: Bool
        public let usesExplicitAuditableFactSignal: Bool
        public let usesRegexOrKeywordFactDetection: Bool
        public let broadMeshRoutingEnabled: Bool
        public let qualityImprovementClaimAllowed: Bool
    }

    public static let schemaVersion = "edge.memory_policy_planner.v1"

    public static func plan(signals: Signals) -> Plan {
        let compaction = compactionPlan(signals)
        let recall = recallPlan(signals)
        let mesh = meshPlan(signals)
        let qualityLoop = qualityLoopPlan(signals, compaction: compaction)
        return Plan(
            intent: signals.intent,
            compaction: compaction,
            recall: recall,
            mesh: mesh,
            qualityLoop: qualityLoop,
            audit: Audit(
                schemaVersion: schemaVersion,
                complementsMemoryBudgetPlanner: true,
                sideEffectFree: true,
                usesExplicitAuditableFactSignal: signals.hasAuditableFactRequirement,
                usesRegexOrKeywordFactDetection: false,
                broadMeshRoutingEnabled: false,
                qualityImprovementClaimAllowed: false
            )
        )
    }

    private static func compactionPlan(_ signals: Signals) -> CompactionPlan {
        let limit = signals.modelContextLimitTokens
        let triggerRatio: Double
        let recentRatio: Double
        let density: SummaryDensity

        switch signals.intent {
        case .longSession:
            triggerRatio = 0.86
            recentRatio = 0.18
            density = .detailed
        case .exactRecall:
            triggerRatio = 0.78
            recentRatio = 0.16
            density = signals.hasAuditableFactRequirement ? .balanced : .detailed
        case .balanced:
            triggerRatio = 0.72
            recentRatio = 0.12
            density = .balanced
        case .batteryFriendly:
            triggerRatio = 0.55
            recentRatio = 0.08
            density = .compact
        }

        var trigger = roundedWindow(Int(Double(limit) * triggerRatio), floor: 1_024)
        if let availableMemoryMB = signals.availableMemoryMB, availableMemoryMB < 1_024 {
            trigger = max(1_024, roundedWindow(Int(Double(trigger) * 0.85), floor: 1_024))
        }
        let recent = roundedWindow(Int(Double(limit) * recentRatio), floor: 512)
        let summaryBudget = summaryBudgetTokens(
            limit: limit,
            compactionCount: signals.compactionCount,
            density: density
        )
        return CompactionPlan(
            triggerTokenCount: trigger,
            shouldCompactNow: signals.contextTokenCount >= trigger,
            recentWindowTokens: min(recent, max(512, trigger / 2)),
            summaryBudgetTokens: summaryBudget,
            density: density,
            reason: "intent=\(signals.intent.rawValue) context=\(signals.contextTokenCount) trigger=\(trigger)"
        )
    }

    private static func recallPlan(_ signals: Signals) -> RecallPlan {
        guard signals.hasAuditableFactRequirement else {
            return RecallPlan(
                mode: .none,
                route: .none,
                status: .notApplicable,
                reason: "no explicit auditable fact requirement"
            )
        }

        let route = recallRoute(
            factStoreAvailable: signals.factStoreAvailable,
            toolRecallAvailable: signals.toolRecallAvailable
        )
        if route == .none {
            return RecallPlan(
                mode: signals.intent == .exactRecall ? .required : .opportunistic,
                route: .none,
                status: .blocked,
                reason: "auditable fact recall requires fact-store or tool recall; raw context and summary are not sufficient"
            )
        }

        if signals.intent == .exactRecall {
            return RecallPlan(
                mode: .required,
                route: route,
                status: .ready,
                reason: "exactRecall requires auditable facts through \(route.rawValue)"
            )
        }

        return RecallPlan(
            mode: .opportunistic,
            route: route,
            status: .ready,
            reason: "auditable fact recall available through \(route.rawValue)"
        )
    }

    private static func meshPlan(_ signals: Signals) -> MeshPlan {
        guard signals.pairedLongContextNodeAvailable else {
            return MeshPlan(
                status: .notAvailable,
                reason: "no paired long-context node was advertised"
            )
        }
        guard signals.meshPolicyAllowsPairedNode else {
            return MeshPlan(
                status: .blockedByPolicy,
                reason: "paired long-context node is available but explicit mesh policy is disabled"
            )
        }
        return MeshPlan(
            status: .allowedByPolicy,
            reason: "paired long-context node can be considered by a separate execution layer"
        )
    }

    private static func qualityLoopPlan(
        _ signals: Signals,
        compaction: CompactionPlan
    ) -> QualityLoopPlan {
        let requiresProbe = signals.qualitySignalsEnabled && (
            compaction.shouldCompactNow || signals.compactionCount > 0
        )
        return QualityLoopPlan(
            recordOutcomeSignals: signals.qualitySignalsEnabled,
            requiresPostCompactionProbe: requiresProbe,
            allowsQualityClaim: false,
            reason: signals.qualitySignalsEnabled
                ? "record memory-policy outcome signals; eval evidence is required before any quality claim"
                : "quality signal loop disabled"
        )
    }

    private static func recallRoute(
        factStoreAvailable: Bool,
        toolRecallAvailable: Bool
    ) -> RecallRoute {
        switch (factStoreAvailable, toolRecallAvailable) {
        case (true, true):
            return .factStoreAndTool
        case (true, false):
            return .factStore
        case (false, true):
            return .tool
        case (false, false):
            return .none
        }
    }

    private static func summaryBudgetTokens(
        limit: Int,
        compactionCount: Int,
        density: SummaryDensity
    ) -> Int {
        let baseRatio: Double
        switch density {
        case .compact:
            baseRatio = 0.035
        case .balanced:
            baseRatio = 0.055
        case .detailed:
            baseRatio = 0.075
        }
        let decay = max(0.70, 1.0 - Double(compactionCount) * 0.05)
        return roundedWindow(Int(Double(limit) * baseRatio * decay), floor: 256)
    }

    private static func roundedWindow(_ value: Int, floor: Int) -> Int {
        max(floor, (max(value, floor) / 256) * 256)
    }
}
