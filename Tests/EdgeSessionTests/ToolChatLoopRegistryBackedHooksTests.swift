// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation
@testable import EdgeSession
import XCTest

@MainActor
final class ToolChatLoopRegistryBackedHooksTests: XCTestCase {
    func testRegistryBackedHooksExecuteSchemaDrivenModelToolCall() async throws {
        let fixture = try metricFixture()
        defer { try? FileManager.default.removeItem(at: fixture.tmpDir) }
        let toolRegistry = ToolRegistry()
        toolRegistry.register(FactQueryPlanToolAdapter.registeredTool(
            store: fixture.store,
            registry: fixture.schemaRegistry
        ))
        let client = RegistryBackedToolLoopFakeClient(rounds: [
            .init(
                chunks: [],
                toolCalls: [Self.countAlphaToolCall()]
            ),
        ])
        let session = ChatSessionController(client: client)
        var streamed: [String] = []
        var toolResults: [ToolChatLoop.ToolResult] = []

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [.system("system"), .user("question")],
                mode: .tool,
                tools: [try FactQueryPlanToolAdapter.defaultMetadata.toolSpec()],
                allowedToolNames: [FactQueryPlanToolAdapter.defaultToolName],
                maxRounds: 1
            ),
            hooks: .registryBacked(
                registry: toolRegistry,
                summarizeToolResults: Self.countSummary,
                streamSummary: { summary, onChunk in onChunk(summary) },
                onToolResult: { toolResults.append($0) }
            ),
            onChunk: { streamed.append($0) }
        )

        XCTAssertEqual(reply, "count=2")
        XCTAssertEqual(streamed, ["count=2"])
        XCTAssertEqual(toolResults.map(\.name), [FactQueryPlanToolAdapter.defaultToolName])
        XCTAssertEqual(toolResults.map(\.source), ["model_generated"])
        XCTAssertEqual(client.generateCallCount, 1)
    }

    func testRegistryBackedHooksStillRespectAllowedToolNamesGate() async throws {
        let toolRegistry = ToolRegistry()
        let client = RegistryBackedToolLoopFakeClient(rounds: [
            .init(
                chunks: [],
                toolCalls: [Self.countAlphaToolCall()]
            ),
        ])
        let session = ChatSessionController(client: client)
        var toolResults: [ToolChatLoop.ToolResult] = []

        let reply = try await ToolChatLoop.run(
            session: session,
            request: ToolChatLoop.Request(
                messages: [.system("system"), .user("question")],
                mode: .tool,
                tools: [try FactQueryPlanToolAdapter.defaultMetadata.toolSpec()],
                allowedToolNames: ["other_fact_tool"],
                maxRounds: 1
            ),
            hooks: .registryBacked(
                registry: toolRegistry,
                summarizeToolResults: { $0.first?.result ?? "" },
                streamSummary: { _, _ in },
                onToolResult: { toolResults.append($0) }
            ),
            onChunk: { _ in }
        )

        XCTAssertTrue(reply.contains("was not selected for this route"))
        XCTAssertFalse(reply.contains("is not registered"))
        XCTAssertEqual(toolResults.map(\.name), [FactQueryPlanToolAdapter.defaultToolName])
    }

    private static func countAlphaToolCall() -> ToolCall {
        ToolCall(function: ToolCall.Function(
            name: FactQueryPlanToolAdapter.defaultToolName,
            arguments: [
                "plan": [
                    "schema": Self.metricSchemaName,
                    "filters": [
                        [
                            "field": "label",
                            "op": "contains",
                            "value": "alpha",
                        ] as [String: any Sendable],
                    ] as [any Sendable],
                    "aggregation": [
                        "function": "count",
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ))
    }

    private static func countSummary(_ results: [ToolChatLoop.ToolResult]) -> String {
        guard let result = results.first,
              let data = result.result.data(using: .utf8),
              let output = try? JSONDecoder().decode(FactQueryPlanToolAdapter.Output.self, from: data),
              let count = output.aggregate?.intValue else {
            return "count=unavailable"
        }
        return "count=\(count)"
    }

    private func metricFixture() throws -> MetricFixture {
        let schemaRegistry = FactSchemaRegistry.shared
        schemaRegistry.register(Self.metricSchema)
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tool-chat-loop-registry-backed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let store = try FactStore(
            path: tmpDir.appendingPathComponent("facts.sqlite3"),
            registry: schemaRegistry
        )
        _ = try store.record(metric(
            score: 10,
            label: "alpha-early",
            bucket: "control",
            observedAt: 100
        ))
        _ = try store.record(metric(
            score: 20,
            label: "alpha-late",
            bucket: "control",
            observedAt: 200
        ))
        _ = try store.record(metric(
            score: 30,
            label: "beta-late",
            bucket: "control",
            observedAt: 300
        ))
        return MetricFixture(tmpDir: tmpDir, schemaRegistry: schemaRegistry, store: store)
    }

    private func metric(
        score: Double,
        label: String,
        bucket: String,
        observedAt: Int64
    ) -> FactRecord {
        FactRecord.new(
            schemaName: Self.metricSchemaName,
            payload: [
                "score": .double(score),
                "label": .string(label),
                "bucket": .string(bucket),
                "observed_at": .int(observedAt),
            ]
        )
    }

    private struct MetricFixture {
        let tmpDir: URL
        let schemaRegistry: FactSchemaRegistry
        let store: FactStore
    }

    private static let metricSchemaName = "test.registry_backed_metric_event"

    private static let metricSchema: FactSchema = {
        try! FactSchema(
            name: metricSchemaName,
            fields: [
                "score": FactFieldDef(type: .float, required: true, indexed: true),
                "label": FactFieldDef(type: .str, required: true, indexed: true),
                "bucket": FactFieldDef(type: .str, required: true, indexed: true),
                "observed_at": FactFieldDef(type: .datetime, required: true, indexed: true),
            ],
            primaryTimeField: "observed_at",
            description: "Generic metric event fixture"
        )
    }()
}

@MainActor
private final class RegistryBackedToolLoopFakeClient: EdgeGenerationClient {
    struct Round {
        var chunks: [String]
        var toolCalls: [ToolCall]
    }

    var currentInferenceMetrics: InferenceMetrics?
    private let rounds: [Round]
    private(set) var generateCallCount = 0

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
        _ = messages
        _ = ciImages
        _ = tools
        _ = parameters
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

    func resetRuntime(reason: String) async {
        _ = reason
    }
}
