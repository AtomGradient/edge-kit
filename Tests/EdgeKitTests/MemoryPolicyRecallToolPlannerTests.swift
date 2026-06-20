// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class MemoryPolicyRecallToolPlannerTests: XCTestCase {
    func testReadyToolRouteExposesOnlyReadOnlyFactTools() {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        registry.register(tool(name: "aggregate_facts", permissions: [.readFacts], intentTags: [.aggregateFact]))
        registry.register(tool(name: "write_facts", permissions: [.readFacts, .writeFacts], intentTags: [.exactFact]))
        registry.register(tool(name: "profile_read", permissions: [.readProfile], intentTags: [.userProfile]))
        registry.register(tool(name: "fact_without_intent", permissions: [.readFacts], intentTags: [.userProfile]))
        registry.register(tool(name: "app_action", permissions: [.appAction], intentTags: [.appAction]))

        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )

        XCTAssertEqual(decision.status, .ready)
        XCTAssertEqual(decision.eligibleToolNames, ["aggregate_facts", "query_facts"])
        XCTAssertEqual(decision.eligibleTools.map(\.name), decision.eligibleToolNames)
        XCTAssertEqual(
            Set(decision.rejectedTools),
            Set([
                .init(name: "app_action", reason: .notReadOnly),
                .init(name: "fact_without_intent", reason: .missingFactIntentTag),
                .init(name: "profile_read", reason: .missingReadFactsPermission),
                .init(name: "write_facts", reason: .notReadOnly),
            ])
        )
        XCTAssertFalse(decision.audit.toolSpecsGenerated)
        XCTAssertFalse(decision.audit.toolsExecuted)
        XCTAssertFalse(decision.audit.toolCallsCreated)
        XCTAssertFalse(decision.audit.usesRegexOrKeywordFactDetection)
    }

    func testToolRouteWithNoEligibleReadFactToolsBlocks() {
        let registry = ToolRegistry()
        registry.register(tool(name: "profile_read", permissions: [.readProfile], intentTags: [.userProfile]))
        registry.register(tool(name: "write_facts", permissions: [.readFacts, .writeFacts], intentTags: [.exactFact]))

        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )

        XCTAssertEqual(decision.status, .blocked)
        XCTAssertEqual(decision.eligibleToolNames, [])
        XCTAssertTrue(decision.audit.reason.contains("no read-only fact tools"))
    }

    func testFactStoreOnlyRouteDoesNotExposeTools() {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        let memoryPlan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 4_096,
            hasAuditableFactRequirement: true,
            factStoreAvailable: true,
            toolRecallAvailable: false
        ))

        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: memoryPlan,
            registry: registry
        )

        XCTAssertEqual(memoryPlan.recall.route, .factStore)
        XCTAssertEqual(decision.status, .notApplicable)
        XCTAssertEqual(decision.eligibleToolNames, [])
        XCTAssertEqual(decision.rejectedTools, [])
        XCTAssertTrue(decision.audit.reason.contains("does not require tool recall"))
    }

    func testNoAuditableFactRequirementIsNotApplicable() {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        let memoryPlan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 512,
            modelContextLimitTokens: 4_096,
            hasAuditableFactRequirement: false,
            factStoreAvailable: true,
            toolRecallAvailable: true
        ))

        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: memoryPlan,
            registry: registry
        )

        XCTAssertEqual(memoryPlan.recall.status, .notApplicable)
        XCTAssertEqual(decision.status, .notApplicable)
        XCTAssertEqual(decision.audit.recallStatus, MemoryPolicyPlanner.RecallStatus.notApplicable.rawValue)
    }

    func testAuditRoundTripsWithoutToolSpecsOrCalls() throws {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )

        let data = try JSONEncoder().encode(decision.audit)
        let decoded = try JSONDecoder().decode(MemoryPolicyRecallToolPlanner.Audit.self, from: data)

        XCTAssertEqual(decoded, decision.audit)
        XCTAssertEqual(decoded.schemaVersion, MemoryPolicyRecallToolPlanner.schemaVersion)
        XCTAssertEqual(decoded.eligibleToolNames, ["query_facts"])
        XCTAssertFalse(decoded.toolSpecsGenerated)
        XCTAssertFalse(decoded.toolsExecuted)
        XCTAssertFalse(decoded.toolCallsCreated)
    }

    func testToolPromptBuildsSpecsOnlyForReadyDecision() throws {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        registry.register(tool(name: "write_facts", permissions: [.readFacts, .writeFacts], intentTags: [.exactFact]))
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )

        let prompt = try decision.toolPrompt()
        let function = try XCTUnwrap(prompt.toolSpecs.first?["function"] as? [String: any Sendable])

        XCTAssertEqual(prompt.allowedToolNames, ["query_facts"])
        XCTAssertEqual(prompt.toolSpecs.count, 1)
        XCTAssertEqual(function["name"] as? String, "query_facts")
        XCTAssertFalse(decision.audit.toolSpecsGenerated)
        XCTAssertTrue(prompt.audit.toolSpecsGenerated)
        XCTAssertEqual(prompt.audit.schemaVersion, MemoryPolicyRecallToolPlanner.toolPromptSchemaVersion)
        XCTAssertEqual(prompt.audit.eligibilitySchemaVersion, MemoryPolicyRecallToolPlanner.schemaVersion)
        XCTAssertEqual(prompt.audit.eligibilityStatus, MemoryPolicyRecallToolPlanner.Status.ready.rawValue)
        XCTAssertEqual(prompt.audit.eligibleToolNames, ["query_facts"])
        XCTAssertEqual(prompt.audit.allowedToolNames, ["query_facts"])
        XCTAssertEqual(prompt.audit.toolSpecCount, 1)
        XCTAssertFalse(prompt.audit.toolsExecuted)
        XCTAssertFalse(prompt.audit.toolCallsCreated)
        XCTAssertFalse(prompt.audit.usesRegexOrKeywordFactDetection)

        let auditData = try JSONEncoder().encode(prompt.audit)
        let decodedAudit = try JSONDecoder().decode(
            MemoryPolicyRecallToolPlanner.ToolPromptAudit.self,
            from: auditData
        )
        XCTAssertEqual(decodedAudit, prompt.audit)
    }

    func testToolPromptFailClosesForBlockedDecision() throws {
        let registry = ToolRegistry()
        registry.register(tool(name: "write_facts", permissions: [.readFacts, .writeFacts], intentTags: [.exactFact]))
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )

        let prompt = try decision.toolPrompt()

        XCTAssertEqual(decision.status, .blocked)
        XCTAssertEqual(prompt.toolSpecs.count, 0)
        XCTAssertTrue(prompt.allowedToolNames.isEmpty)
        XCTAssertFalse(prompt.audit.toolSpecsGenerated)
        XCTAssertEqual(prompt.audit.eligibilityStatus, MemoryPolicyRecallToolPlanner.Status.blocked.rawValue)
        XCTAssertTrue(prompt.audit.reason.contains("not generated"))
    }

    func testToolPromptFailClosesForNotApplicableDecision() throws {
        let registry = ToolRegistry()
        registry.register(tool(name: "query_facts", permissions: [.readFacts], intentTags: [.exactFact]))
        let memoryPlan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 512,
            modelContextLimitTokens: 4_096,
            hasAuditableFactRequirement: false,
            factStoreAvailable: true,
            toolRecallAvailable: true
        ))
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: memoryPlan,
            registry: registry
        )

        let prompt = try decision.toolPrompt()

        XCTAssertEqual(decision.status, .notApplicable)
        XCTAssertEqual(prompt.toolSpecs.count, 0)
        XCTAssertTrue(prompt.allowedToolNames.isEmpty)
        XCTAssertFalse(prompt.audit.toolSpecsGenerated)
        XCTAssertEqual(prompt.audit.eligibilityStatus, MemoryPolicyRecallToolPlanner.Status.notApplicable.rawValue)
    }

    func testToolPromptFailClosesWhenReadyDecisionHasNoEligibleTools() throws {
        let decision = MemoryPolicyRecallToolPlanner.Decision(
            status: .ready,
            eligibleTools: [],
            eligibleToolNames: [],
            rejectedTools: [],
            audit: MemoryPolicyRecallToolPlanner.Audit(
                schemaVersion: MemoryPolicyRecallToolPlanner.schemaVersion,
                recallMode: MemoryPolicyPlanner.RecallMode.required.rawValue,
                recallRoute: MemoryPolicyPlanner.RecallRoute.tool.rawValue,
                recallStatus: MemoryPolicyPlanner.RecallStatus.ready.rawValue,
                eligibleToolNames: [],
                rejectedTools: [],
                toolSpecsGenerated: false,
                toolsExecuted: false,
                toolCallsCreated: false,
                usesRegexOrKeywordFactDetection: false,
                reason: "synthetic ready decision without eligible tools"
            )
        )

        let prompt = try decision.toolPrompt()

        XCTAssertEqual(prompt.toolSpecs.count, 0)
        XCTAssertTrue(prompt.allowedToolNames.isEmpty)
        XCTAssertFalse(prompt.audit.toolSpecsGenerated)
        XCTAssertEqual(prompt.audit.eligibilityStatus, MemoryPolicyRecallToolPlanner.Status.ready.rawValue)
        XCTAssertTrue(prompt.audit.reason.contains("not generated"))
    }

    private func toolReadyMemoryPlan() -> MemoryPolicyPlanner.Plan {
        MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 4_096,
            hasAuditableFactRequirement: true,
            factStoreAvailable: false,
            toolRecallAvailable: true
        ))
    }

    private func tool(
        name: String,
        permissions: [ToolPermission],
        intentTags: [PersonalIntentTag]
    ) -> RegisteredTool {
        RegisteredTool(
            metadata: ToolMetadata(
                name: name,
                description: name,
                argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
                permissions: permissions,
                intentTags: intentTags
            )
        ) { _ in
            XCTFail("Recall tool eligibility planner must not execute tools")
            return "unexpected"
        }
    }
}
