// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
@testable import EdgeSession
import XCTest

final class ToolChatLoopMemoryToolPromptBridgeTests: XCTestCase {
    func testAppendAppliesReadyMemoryToolPromptWithoutTouchingExecutionState() throws {
        let prompt = try memoryToolPrompt(toolNames: ["neutral_recall_alpha"])
        let planned = ToolChatLoop.PlannedToolCall(
            plan: ToolCallPlan(toolName: "existing_tool", reason: "caller planned"),
            source: "caller",
            reason: "preserve"
        )
        let request = baseRequest(
            tools: [existingToolSpec(name: "existing_tool")],
            allowedToolNames: ["existing_tool"],
            plannedToolCalls: [planned],
            maxRounds: 7
        )

        let updated = request.applying(memoryToolPrompt: prompt)

        XCTAssertEqual(toolNames(updated.tools), ["existing_tool", "neutral_recall_alpha"])
        XCTAssertEqual(updated.allowedToolNames, ["existing_tool", "neutral_recall_alpha"])
        XCTAssertEqual(updated.plannedToolCalls.map(\.plan.toolName), ["existing_tool"])
        XCTAssertEqual(updated.plannedToolCalls.map(\.source), ["caller"])
        XCTAssertEqual(updated.maxRounds, 7)
        XCTAssertEqual(updated.mode, request.mode)
        XCTAssertEqual(updated.emptyFinalText, request.emptyFinalText)
        XCTAssertEqual(updated.suppressNextChunkAfterToolCall, request.suppressNextChunkAfterToolCall)
    }

    func testAppendDeduplicatesPromptToolNames() throws {
        let prompt = try memoryToolPrompt(toolNames: ["neutral_recall_alpha"])
        let request = baseRequest(
            tools: prompt.toolSpecs,
            allowedToolNames: ["neutral_recall_alpha"]
        )

        let updated = request.applying(memoryToolPrompt: prompt, mergePolicy: .append)

        XCTAssertEqual(toolNames(updated.tools), ["neutral_recall_alpha"])
        XCTAssertEqual(updated.allowedToolNames, ["neutral_recall_alpha"])
    }

    func testAppendEmptyPromptIsNoOp() throws {
        let prompt = try blockedMemoryToolPrompt()
        let request = baseRequest(
            tools: [existingToolSpec(name: "existing_tool")],
            allowedToolNames: ["existing_tool"]
        )

        let updated = request.applying(memoryToolPrompt: prompt, mergePolicy: .append)

        XCTAssertEqual(toolNames(updated.tools), ["existing_tool"])
        XCTAssertEqual(updated.allowedToolNames, ["existing_tool"])
    }

    func testReplaceUsesOnlyPromptMaterialAndCanFailClosedToEmpty() throws {
        let readyPrompt = try memoryToolPrompt(toolNames: ["neutral_recall_alpha", "neutral_recall_beta"])
        let request = baseRequest(
            tools: [existingToolSpec(name: "existing_tool")],
            allowedToolNames: ["existing_tool"]
        )

        let replaced = request.applying(memoryToolPrompt: readyPrompt, mergePolicy: .replace)

        XCTAssertEqual(toolNames(replaced.tools), ["neutral_recall_alpha", "neutral_recall_beta"])
        XCTAssertEqual(replaced.allowedToolNames, ["neutral_recall_alpha", "neutral_recall_beta"])

        let emptyPrompt = try blockedMemoryToolPrompt()
        let failClosed = request.applying(memoryToolPrompt: emptyPrompt, mergePolicy: .replace)

        XCTAssertTrue(failClosed.tools.isEmpty)
        XCTAssertTrue(failClosed.allowedToolNames.isEmpty)
    }

    private func memoryToolPrompt(
        toolNames: [String]
    ) throws -> MemoryPolicyRecallToolPlanner.ToolPrompt {
        let registry = ToolRegistry()
        for name in toolNames {
            registry.register(readOnlyFactTool(name: name))
        }
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )
        return try decision.toolPrompt()
    }

    private func blockedMemoryToolPrompt() throws -> MemoryPolicyRecallToolPlanner.ToolPrompt {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(
            metadata: ToolMetadata(
                name: "neutral_write_tool",
                description: "neutral_write_tool",
                argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
                permissions: [.writeFacts],
                intentTags: [.exactFact]
            )
        ) { _ in
            XCTFail("Memory prompt bridge must not execute tools")
            return "unexpected"
        })
        let decision = MemoryPolicyRecallToolPlanner().plan(
            memoryPlan: toolReadyMemoryPlan(),
            registry: registry
        )
        return try decision.toolPrompt()
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

    private func readOnlyFactTool(name: String) -> RegisteredTool {
        RegisteredTool(
            metadata: ToolMetadata(
                name: name,
                description: name,
                argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
                permissions: [.readFacts],
                intentTags: [.exactFact]
            )
        ) { _ in
            XCTFail("Memory prompt bridge must not execute tools")
            return "unexpected"
        }
    }

    private func baseRequest(
        tools: [EdgeSessionToolSpec],
        allowedToolNames: [String],
        plannedToolCalls: [ToolChatLoop.PlannedToolCall] = [],
        maxRounds: Int = 3
    ) -> ToolChatLoop.Request {
        ToolChatLoop.Request(
            messages: [.system("system"), .user("question")],
            mode: .tool,
            tools: tools,
            allowedToolNames: allowedToolNames,
            plannedToolCalls: plannedToolCalls,
            maxRounds: maxRounds,
            emptyFinalText: "empty",
            suppressNextChunkAfterToolCall: false
        )
    }

    private func existingToolSpec(name: String) -> EdgeSessionToolSpec {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": name,
                "parameters": [
                    "type": "object",
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    private func toolNames(_ specs: [EdgeSessionToolSpec]) -> [String] {
        specs.compactMap { spec in
            guard let function = spec["function"] as? [String: any Sendable] else {
                return nil
            }
            return function["name"] as? String
        }
    }
}
