// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference
@testable import EdgeSession
import XCTest

@MainActor
final class ToolChatLoopTests: XCTestCase {
    func testRunsFallbackWhenToolsUnavailable() async throws {
        let session = ChatSessionController(client: ToolLoopFakeClient(rounds: []))
        var unavailable: ([String], Int)?

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [.system("system"), .user("question")],
                mode: .tool,
                tools: [],
                allowedToolNames: ["query"],
                emptyFinalText: "empty"
            ),
            hooks: ToolChatLoop.Hooks(
                executePlannedTool: { _ in XCTFail("planned tool should not run"); return "" },
                executeModelTool: { _ in XCTFail("model tool should not run"); return "" },
                summarizeToolResults: { _ in "" },
                fallbackWhenToolsUnavailable: { "fallback" },
                onToolsUnavailable: { names, count in unavailable = (names, count) }
            ),
            onChunk: { _ in }
        )

        XCTAssertEqual(reply, "fallback")
        XCTAssertEqual(unavailable?.0, ["query"])
        XCTAssertEqual(unavailable?.1, 0)
    }

    func testPlannedToolSummaryShortCircuitsModelGeneration() async throws {
        let client = ToolLoopFakeClient(rounds: [
            .init(chunks: ["should not run"], toolCalls: []),
        ])
        let session = ChatSessionController(client: client)
        let planned = ToolCallPlan(toolName: "query", arguments: ["limit": .int(1)])
        var streamed: [String] = []
        var results: [ToolChatLoop.ToolResult] = []

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [.system("system"), .user("question")],
                mode: .tool,
                tools: [[:]],
                allowedToolNames: ["query"],
                plannedToolCalls: [.init(plan: planned, source: "planned")]
            ),
            hooks: ToolChatLoop.Hooks(
                executePlannedTool: { plan in
                    XCTAssertEqual(plan.toolName, "query")
                    return "planned result"
                },
                executeModelTool: { _ in XCTFail("model tool should not run"); return "" },
                summarizeToolResults: { toolResults in
                    results = toolResults
                    return "summary"
                },
                streamSummary: { summary, onChunk in onChunk(summary) }
            ),
            onChunk: { streamed.append($0) }
        )

        XCTAssertEqual(reply, "summary")
        XCTAssertEqual(streamed, ["summary"])
        XCTAssertEqual(results.map(\.name), ["query"])
        XCTAssertEqual(results.map(\.source), ["planned"])
        XCTAssertEqual(client.generateCallCount, 0)
    }

    func testModelToolRoundCompactsConsumedToolMessagesBeforeNextRound() async throws {
        let client = ToolLoopFakeClient(rounds: [
            .init(
                chunks: ["first round"],
                toolCalls: [
                    ToolCall(function: .init(name: "query", arguments: ["q": "hello"])),
                ]
            ),
            .init(chunks: ["final answer"], toolCalls: []),
        ])
        let session = ChatSessionController(client: client)
        var roundContinueCount = 0
        var observedMetricRounds: [Int] = []

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [
                    .system("system"),
                    .tool("[old_tool] old verbose payload"),
                    .user("question"),
                ],
                mode: .tool,
                tools: [[:]],
                allowedToolNames: ["query"],
                maxRounds: 2
            ),
            hooks: ToolChatLoop.Hooks(
                executePlannedTool: { _ in XCTFail("planned tool should not run"); return "" },
                executeModelTool: { toolCall in
                    XCTAssertEqual(toolCall.function.name, "query")
                    return "tool result"
                },
                summarizeToolResults: { _ in "" },
                onRoundWillContinue: { _ in roundContinueCount += 1 },
                onRoundMetrics: { roundIndex, _ in
                    observedMetricRounds.append(roundIndex)
                }
            ),
            onChunk: { _ in }
        )

        XCTAssertEqual(reply, "final answer")
        XCTAssertEqual(roundContinueCount, 1)
        XCTAssertEqual(observedMetricRounds, [0, 1])
        XCTAssertEqual(client.generateCallCount, 2)
        XCTAssertEqual(
            client.messagesByRound[1].map(\.content),
            [
                "system",
                "[old_tool] (consumed in earlier round)",
                "question",
                "first round",
                "[query] tool result",
            ]
        )
    }

    func testModelToolOnlyRoundDoesNotAppendEmptyAssistantMessageBeforeNextRound() async throws {
        let client = ToolLoopFakeClient(rounds: [
            .init(
                chunks: [],
                toolCalls: [
                    ToolCall(function: .init(name: "query", arguments: ["q": "hello"])),
                ]
            ),
            .init(chunks: ["final answer"], toolCalls: []),
        ])
        let session = ChatSessionController(client: client)

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [
                    .system("system"),
                    .user("question"),
                ],
                mode: .tool,
                tools: [[:]],
                allowedToolNames: ["query"],
                maxRounds: 2
            ),
            hooks: ToolChatLoop.Hooks(
                executePlannedTool: { _ in XCTFail("planned tool should not run"); return "" },
                executeModelTool: { toolCall in
                    XCTAssertEqual(toolCall.function.name, "query")
                    return "tool result"
                },
                summarizeToolResults: { _ in "" }
            ),
            onChunk: { _ in }
        )

        XCTAssertEqual(reply, "final answer")
        XCTAssertEqual(client.generateCallCount, 2)
        XCTAssertEqual(
            client.messagesByRound[1].map(\.content),
            [
                "system",
                "question",
                "[query] tool result",
            ]
        )
    }

    func testModelToolExecutionErrorReturnsToolResultAndContinues() async throws {
        let toolErrorMessage = "update_expense does not accept `type`; use `category` for category changes."
        let client = ToolLoopFakeClient(rounds: [
            .init(
                chunks: [],
                toolCalls: [
                    ToolCall(function: .init(name: "update_expense", arguments: ["type": "Food"])),
                ]
            ),
            .init(chunks: ["final answer"], toolCalls: []),
        ])
        let session = ChatSessionController(client: client)
        var toolResults: [ToolChatLoop.ToolResult] = []

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [.system("system"), .user("change the category")],
                mode: .tool,
                tools: [[:]],
                allowedToolNames: ["update_expense"],
                maxRounds: 2
            ),
            hooks: ToolChatLoop.Hooks(
                executePlannedTool: { _ in XCTFail("planned tool should not run"); return "" },
                executeModelTool: { _ in
                    throw DecodingError.dataCorrupted(
                        .init(codingPath: [], debugDescription: toolErrorMessage)
                    )
                },
                summarizeToolResults: { _ in "" },
                onToolResult: { toolResults.append($0) }
            ),
            onChunk: { _ in }
        )

        XCTAssertEqual(reply, "final answer")
        XCTAssertEqual(client.generateCallCount, 2)
        XCTAssertEqual(toolResults.map(\.name), ["update_expense"])
        let resultData = try XCTUnwrap(toolResults.first?.result.data(using: .utf8))
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: resultData) as? [String: String])
        let error = try XCTUnwrap(payload["error"])
        XCTAssertTrue(error.contains("tool 'update_expense' failed"))
        XCTAssertTrue(error.contains(toolErrorMessage))
        XCTAssertTrue(
            client.messagesByRound[1].map(\.content).contains { message in
                message.contains("[update_expense]")
                    && message.contains("\"error\"")
                    && message.contains(toolErrorMessage)
            }
        )
    }
}

@MainActor
private final class ToolLoopFakeClient: EdgeGenerationClient {
    struct Round {
        var chunks: [String]
        var toolCalls: [ToolCall]
    }

    var currentInferenceMetrics: InferenceMetrics?
    private let rounds: [Round]
    private(set) var generateCallCount = 0
    private(set) var messagesByRound: [[ChatMessage]] = []

    init(rounds: [Round]) {
        self.rounds = rounds
    }

    func generate(
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [EdgeSessionToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters: EdgeGenerateParameters?,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String {
        messagesByRound.append(messages)
        let round = generateCallCount < rounds.count
            ? rounds[generateCallCount]
            : Round(chunks: [], toolCalls: [])
        generateCallCount += 1

        var output = ""
        for toolCall in round.toolCalls {
            if let onToolCall {
                let result = try await onToolCall(toolCall)
                output += result
                onChunk(result)
            }
        }
        for chunk in round.chunks {
            output += chunk
            onChunk(chunk)
        }
        return output
    }

    func resetRuntime(reason: String) async {}
}
