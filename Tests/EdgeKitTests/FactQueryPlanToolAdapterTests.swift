// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class FactQueryPlanToolAdapterTests: XCTestCase {
    private var tmpDir: URL!
    private var registry: FactSchemaRegistry!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fact-query-plan-tool-adapter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        registry = FactSchemaRegistry()
        registry.register(Self.metricSchema)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
        registry = nil
    }

    func testDefaultMetadataIsReadOnlyAndGeneric() throws {
        let metadata = FactQueryPlanToolAdapter.defaultMetadata

        XCTAssertEqual(metadata.name, FactQueryPlanToolAdapter.defaultToolName)
        XCTAssertTrue(metadata.isReadOnly)
        XCTAssertTrue(metadata.supports(intentTag: .exactFact))
        XCTAssertTrue(metadata.supports(intentTag: .aggregateFact))
        XCTAssertFalse(metadata.supports(intentTag: .userProfile))

        guard case .jsonSchema(let argumentsSchema) = metadata.argumentsSchema else {
            return XCTFail("Expected JSON schema arguments")
        }
        XCTAssertTrue(argumentsSchema.contains("\"schema\""))
        XCTAssertTrue(argumentsSchema.contains("\"filters\""))
        XCTAssertFalse(argumentsSchema.contains("score"))
        XCTAssertFalse(argumentsSchema.contains("label"))
        XCTAssertFalse(argumentsSchema.contains("bucket"))
        XCTAssertFalse(argumentsSchema.contains("observed_at"))
    }

    func testRegistryExecutesExactQueryFromTopLevelPlanArguments() async throws {
        let toolRegistry = ToolRegistry()
        toolRegistry.register(FactQueryPlanToolAdapter.registeredTool(
            store: try populatedStore(),
            registry: registry
        ))
        let call = ToolCall(function: ToolCall.Function(
            name: FactQueryPlanToolAdapter.defaultToolName,
            arguments: [
                "schema": Self.metricSchemaName,
                "filters": [
                    [
                        "field": "label",
                        "op": "contains",
                        "value": "alpha",
                    ] as [String: any Sendable],
                    [
                        "field": "bucket",
                        "op": "equals",
                        "value": "control",
                    ] as [String: any Sendable],
                ] as [any Sendable],
                "sort": [
                    [
                        "field": "observed_at",
                        "direction": "desc",
                    ] as [String: any Sendable],
                ] as [any Sendable],
                "limit": 1,
            ] as [String: any Sendable]
        ))

        let result = try await toolRegistry.execute(call)
        let output = try decodeOutput(result)

        XCTAssertEqual(output.records.map { $0.payload["label"] }, [.string("alpha-late")])
        XCTAssertNil(output.aggregate)
        XCTAssertEqual(output.audit.querySource, "explicit_schema_driven_fact_query_plan")
        XCTAssertEqual(output.audit.executionType, "exact_query")
        XCTAssertTrue(output.audit.planExecuted)
        XCTAssertFalse(output.audit.usesRegexOrKeywordFactDetection)
    }

    func testRegistryExecutesAggregateThroughNestedPlanWrapper() async throws {
        let toolRegistry = ToolRegistry()
        toolRegistry.register(FactQueryPlanToolAdapter.registeredTool(
            store: try populatedStore(),
            registry: registry
        ))
        let call = ToolCall(function: ToolCall.Function(
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

        let result = try await toolRegistry.execute(call)
        let output = try decodeOutput(result)

        XCTAssertEqual(output.records.count, 2)
        XCTAssertEqual(output.aggregate?.function, .count)
        XCTAssertEqual(output.aggregate?.intValue, 2)
        XCTAssertEqual(output.audit.executionType, "aggregate_count")
        XCTAssertEqual(output.audit.aggregateFunction, "count")
    }

    func testUnsupportedPlanFailsClosedThroughExecutor() async throws {
        let toolRegistry = ToolRegistry()
        toolRegistry.register(FactQueryPlanToolAdapter.registeredTool(
            store: try populatedStore(),
            registry: registry
        ))
        let call = ToolCall(function: ToolCall.Function(
            name: FactQueryPlanToolAdapter.defaultToolName,
            arguments: [
                "namespace": "external.metrics",
                "schema": Self.metricSchemaName,
            ] as [String: any Sendable]
        ))

        do {
            _ = try await toolRegistry.execute(call)
            XCTFail("Expected unsupported namespace failure")
        } catch let error as FactQueryPlanExecutorError {
            XCTAssertEqual(error, .unsupportedNamespace("external.metrics"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private func populatedStore() throws -> FactStore {
        let store = try FactStore(
            path: tmpDir.appendingPathComponent("facts.sqlite3"),
            registry: registry
        )
        _ = try store.record(metric(score: 10, label: "alpha-early", bucket: "control", observedAt: 100))
        _ = try store.record(metric(score: 20, label: "alpha-late", bucket: "control", observedAt: 200))
        _ = try store.record(metric(score: 30, label: "beta-late", bucket: "control", observedAt: 300))
        _ = try store.record(metric(score: 40, label: "gamma", bucket: "variant", observedAt: 400))
        return store
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

    private func decodeOutput(_ text: String) throws -> FactQueryPlanToolAdapter.Output {
        try JSONDecoder().decode(FactQueryPlanToolAdapter.Output.self, from: Data(text.utf8))
    }

    private static let metricSchemaName = "test.metric_event"

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
