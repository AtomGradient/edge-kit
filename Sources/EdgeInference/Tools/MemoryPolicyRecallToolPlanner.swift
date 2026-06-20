// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Tokenizers

/// Side-effect-free bridge from A8 memory recall plans to eligible fact tools.
///
/// The planner only inspects `ToolMetadata` declarations. It does not parse
/// user text, create tool calls, build tool specs, query a fact store, or
/// execute tools.
public struct MemoryPolicyRecallToolPlanner: Sendable {
    public enum Status: String, Sendable, Codable, Equatable {
        case notApplicable
        case blocked
        case ready
    }

    public enum RejectionReason: String, Sendable, Codable, Equatable {
        case notReadOnly
        case missingReadFactsPermission
        case missingFactIntentTag
    }

    public struct RejectedTool: Sendable, Codable, Equatable, Hashable {
        public let name: String
        public let reason: RejectionReason
    }

    public struct Decision: Sendable {
        public let status: Status
        public let eligibleTools: [ToolMetadata]
        public let eligibleToolNames: [String]
        public let rejectedTools: [RejectedTool]
        public let audit: Audit
    }

    public struct Audit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let recallMode: String
        public let recallRoute: String
        public let recallStatus: String
        public let eligibleToolNames: [String]
        public let rejectedTools: [RejectedTool]
        public let toolSpecsGenerated: Bool
        public let toolsExecuted: Bool
        public let toolCallsCreated: Bool
        public let usesRegexOrKeywordFactDetection: Bool
        public let reason: String
    }

    /// Model-visible tool prompt contract derived from an already eligible
    /// recall decision. It never creates calls or executes tools.
    public struct ToolPrompt {
        public let toolSpecs: [ToolSpec]
        public let allowedToolNames: [String]
        public let audit: ToolPromptAudit
    }

    public struct ToolPromptAudit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let eligibilitySchemaVersion: String
        public let eligibilityStatus: String
        public let eligibilityReason: String
        public let eligibleToolNames: [String]
        public let allowedToolNames: [String]
        public let toolSpecCount: Int
        public let toolSpecsGenerated: Bool
        public let toolsExecuted: Bool
        public let toolCallsCreated: Bool
        public let usesRegexOrKeywordFactDetection: Bool
        public let reason: String
    }

    public static let schemaVersion = "edge.memory_policy_recall_tool_planner.v1"
    public static let toolPromptSchemaVersion = "edge.memory_policy_recall_tool_prompt.v1"

    public init() {}

    public func plan(
        memoryPlan: MemoryPolicyPlanner.Plan,
        registry: ToolRegistry = .shared
    ) -> Decision {
        let allTools = registry.allSchemas()
        let candidateTools = allTools.filter(Self.isEligibleFactRecallTool)
        let candidateRejectedTools = allTools.compactMap(Self.rejectedTool)

        let status: Status
        let reason: String
        let shouldExposeTools: Bool
        switch memoryPlan.recall.status {
        case .notApplicable:
            status = .notApplicable
            reason = memoryPlan.recall.reason
            shouldExposeTools = false
        case .blocked:
            status = .blocked
            reason = memoryPlan.recall.reason
            shouldExposeTools = false
        case .ready:
            if !Self.routeIncludesTool(memoryPlan.recall.route) {
                status = .notApplicable
                reason = "memory recall route \(memoryPlan.recall.route.rawValue) does not require tool recall"
                shouldExposeTools = false
            } else if candidateTools.isEmpty {
                status = .blocked
                reason = "tool recall requested but no read-only fact tools are eligible"
                shouldExposeTools = true
            } else {
                status = .ready
                reason = "eligible read-only fact tools: \(candidateTools.map(\.name).joined(separator: ","))"
                shouldExposeTools = true
            }
        }
        let eligibleTools = shouldExposeTools ? candidateTools : []
        let rejectedTools = shouldExposeTools ? candidateRejectedTools : []
        let eligibleNames = eligibleTools.map(\.name)

        return Decision(
            status: status,
            eligibleTools: eligibleTools,
            eligibleToolNames: eligibleNames,
            rejectedTools: rejectedTools,
            audit: Audit(
                schemaVersion: Self.schemaVersion,
                recallMode: memoryPlan.recall.mode.rawValue,
                recallRoute: memoryPlan.recall.route.rawValue,
                recallStatus: memoryPlan.recall.status.rawValue,
                eligibleToolNames: eligibleNames,
                rejectedTools: rejectedTools,
                toolSpecsGenerated: false,
                toolsExecuted: false,
                toolCallsCreated: false,
                usesRegexOrKeywordFactDetection: false,
                reason: reason
            )
        )
    }

    private static func isEligibleFactRecallTool(_ metadata: ToolMetadata) -> Bool {
        rejectedTool(metadata) == nil
    }

    private static func rejectedTool(_ metadata: ToolMetadata) -> RejectedTool? {
        guard metadata.isReadOnly else {
            return RejectedTool(name: metadata.name, reason: .notReadOnly)
        }
        guard metadata.permissions.contains(.readFacts) else {
            return RejectedTool(name: metadata.name, reason: .missingReadFactsPermission)
        }
        guard metadata.supports(intentTag: .exactFact)
            || metadata.supports(intentTag: .aggregateFact) else {
            return RejectedTool(name: metadata.name, reason: .missingFactIntentTag)
        }
        return nil
    }

    private static func routeIncludesTool(_ route: MemoryPolicyPlanner.RecallRoute) -> Bool {
        switch route {
        case .tool, .factStoreAndTool:
            return true
        case .none, .factStore:
            return false
        }
    }
}

public extension MemoryPolicyRecallToolPlanner.Decision {
    /// Convert an eligibility decision into ToolSpec prompt material.
    ///
    /// Blocked, not-applicable, or empty decisions fail closed with no tools.
    func toolPrompt() throws -> MemoryPolicyRecallToolPlanner.ToolPrompt {
        guard status == .ready, !eligibleTools.isEmpty else {
            return MemoryPolicyRecallToolPlanner.ToolPrompt(
                toolSpecs: [],
                allowedToolNames: [],
                audit: toolPromptAudit(
                    allowedToolNames: [],
                    toolSpecCount: 0,
                    toolSpecsGenerated: false,
                    reason: "memory recall tool prompt not generated: \(audit.reason)"
                )
            )
        }

        let specs = try eligibleTools.map { try $0.toolSpec() }
        return MemoryPolicyRecallToolPlanner.ToolPrompt(
            toolSpecs: specs,
            allowedToolNames: eligibleToolNames,
            audit: toolPromptAudit(
                allowedToolNames: eligibleToolNames,
                toolSpecCount: specs.count,
                toolSpecsGenerated: true,
                reason: "generated memory recall tool prompt for eligible tools: \(eligibleToolNames.joined(separator: ","))"
            )
        )
    }

    private func toolPromptAudit(
        allowedToolNames: [String],
        toolSpecCount: Int,
        toolSpecsGenerated: Bool,
        reason: String
    ) -> MemoryPolicyRecallToolPlanner.ToolPromptAudit {
        MemoryPolicyRecallToolPlanner.ToolPromptAudit(
            schemaVersion: MemoryPolicyRecallToolPlanner.toolPromptSchemaVersion,
            eligibilitySchemaVersion: audit.schemaVersion,
            eligibilityStatus: status.rawValue,
            eligibilityReason: audit.reason,
            eligibleToolNames: eligibleToolNames,
            allowedToolNames: allowedToolNames,
            toolSpecCount: toolSpecCount,
            toolSpecsGenerated: toolSpecsGenerated,
            toolsExecuted: false,
            toolCallsCreated: false,
            usesRegexOrKeywordFactDetection: false,
            reason: reason
        )
    }
}
