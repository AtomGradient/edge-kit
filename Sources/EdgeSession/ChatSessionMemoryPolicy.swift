// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

/// EdgeSession bridge for A8 memory-policy compaction.
///
/// This type intentionally consumes only `MemoryPolicyPlanner.Plan.compaction`.
/// Recall, mesh, and quality-loop plans remain non-runtime audit outputs until
/// their dedicated integration slices exist.
public struct ChatSessionMemoryPolicy: Sendable {
    public static let schemaVersion = "edge.chat_session_memory_policy.compaction.v1"

    public struct CompactionAudit: Sendable {
        public let schemaVersion: String
        public let intent: EdgeMemoryIntent
        public let triggerTokenCount: Int
        public let shouldCompactNow: Bool
        public let recentWindowTokens: Int
        public let summaryBudgetTokens: Int
        public let summaryDensity: String
        public let estimatedCharactersPerToken: Int
        public let sourceCharacterCount: Int?
        public let preparedCharacterCount: Int?
        public let estimatedContextTokens: Int
        public let estimatedPreparedContextTokens: Int
        public let estimatedRecentWindowTokens: Int
        public let estimatedSummaryBudgetTokens: Int
        public let recentWindowCharacterBudget: Int
        public let summaryCharacterBudget: Int
        public let minimumCharacterBudget: Int
        public let plannedCharacterBudget: Int
        public let baseCharacterBudget: Int
        public let effectiveCharacterBudget: Int
        public let tightenedCharacterBudget: Bool
        public let compactedHistory: Bool
        public let reason: String
    }

    public let plan: MemoryPolicyPlanner.Plan
    public let estimatedCharactersPerToken: Int
    public let minimumCharacterBudget: Int

    public init(
        plan: MemoryPolicyPlanner.Plan,
        estimatedCharactersPerToken: Int = 4,
        minimumCharacterBudget: Int = 512
    ) {
        self.plan = plan
        self.estimatedCharactersPerToken = max(1, estimatedCharactersPerToken)
        self.minimumCharacterBudget = max(0, minimumCharacterBudget)
    }

    public func compactorConfig(base: HistoryCompactor.Config) -> HistoryCompactor.Config {
        guard plan.compaction.shouldCompactNow else {
            return base
        }
        var config = base
        config.characterBudget = min(base.characterBudget, plannedCharacterBudget)
        return config
    }

    public func compactionAudit(
        base: HistoryCompactor.Config,
        effective: HistoryCompactor.Config,
        compactedHistory: Bool,
        sourceCharacterCount: Int? = nil,
        preparedCharacterCount: Int? = nil
    ) -> CompactionAudit {
        let normalizedSourceCharacterCount = sourceCharacterCount.map { max(0, $0) }
        let normalizedPreparedCharacterCount = preparedCharacterCount.map { max(0, $0) }
        return CompactionAudit(
            schemaVersion: Self.schemaVersion,
            intent: plan.intent,
            triggerTokenCount: plan.compaction.triggerTokenCount,
            shouldCompactNow: plan.compaction.shouldCompactNow,
            recentWindowTokens: plan.compaction.recentWindowTokens,
            summaryBudgetTokens: plan.compaction.summaryBudgetTokens,
            summaryDensity: plan.compaction.density.rawValue,
            estimatedCharactersPerToken: estimatedCharactersPerToken,
            sourceCharacterCount: normalizedSourceCharacterCount,
            preparedCharacterCount: normalizedPreparedCharacterCount,
            estimatedContextTokens: estimatedTokens(
                forCharacters: normalizedSourceCharacterCount ?? base.characterBudget
            ),
            estimatedPreparedContextTokens: estimatedTokens(
                forCharacters: normalizedPreparedCharacterCount ?? effective.characterBudget
            ),
            estimatedRecentWindowTokens: plan.compaction.recentWindowTokens,
            estimatedSummaryBudgetTokens: plan.compaction.summaryBudgetTokens,
            recentWindowCharacterBudget: recentWindowCharacterBudget,
            summaryCharacterBudget: summaryCharacterBudget,
            minimumCharacterBudget: minimumCharacterBudget,
            plannedCharacterBudget: plannedCharacterBudget,
            baseCharacterBudget: base.characterBudget,
            effectiveCharacterBudget: effective.characterBudget,
            tightenedCharacterBudget: effective.characterBudget < base.characterBudget,
            compactedHistory: compactedHistory,
            reason: plan.compaction.reason
        )
    }

    private var plannedCharacterBudget: Int {
        max(
            minimumCharacterBudget,
            plan.compaction.recentWindowTokens * estimatedCharactersPerToken
        )
    }

    private var recentWindowCharacterBudget: Int {
        plan.compaction.recentWindowTokens * estimatedCharactersPerToken
    }

    private var summaryCharacterBudget: Int {
        plan.compaction.summaryBudgetTokens * estimatedCharactersPerToken
    }

    private func estimatedTokens(forCharacters characters: Int) -> Int {
        let normalizedCharacters = max(0, characters)
        guard normalizedCharacters > 0 else { return 0 }
        return Int(ceil(Double(normalizedCharacters) / Double(estimatedCharactersPerToken)))
    }
}
