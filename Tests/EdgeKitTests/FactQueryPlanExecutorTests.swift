// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class FactQueryPlanExecutorTests: XCTestCase {
    private var tmpDir: URL!
    private var registry: FactSchemaRegistry!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fact-query-plan-executor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        registry = FactSchemaRegistry()
        registry.register(Self.metricSchema)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
        registry = nil
    }

    func testExecutesExactQueryFromExplicitSchemaDrivenPlan() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [
                .init(field: "label", op: .contains, value: .string("alpha")),
                .init(field: "bucket", op: .equals, value: .string("control")),
            ],
            sort: [.init(field: "observed_at", direction: .descending)],
            limit: 1
        )

        let result = try executor.execute(plan)

        XCTAssertEqual(result.records.map { $0.payload["label"]?.stringValue }, ["alpha-late"])
        XCTAssertNil(result.aggregate)
        XCTAssertEqual(result.audit.querySource, "explicit_schema_driven_fact_query_plan")
        XCTAssertEqual(result.audit.executionType, "exact_query")
        XCTAssertEqual(result.audit.recordCount, 1)
        XCTAssertTrue(result.audit.planExecuted)
        XCTAssertFalse(result.audit.toolsExecuted)
        XCTAssertFalse(result.audit.toolCallsCreated)
        XCTAssertFalse(result.audit.usesRegexOrKeywordFactDetection)
    }

    func testExecutesSumAggregateFromExplicitSchemaDrivenPlan() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [
                .init(field: "bucket", op: .equals, value: .string("control")),
                .init(field: "observed_at", op: .greaterThanOrEqual, value: .int(150)),
            ],
            aggregation: .init(function: .sum, field: "score")
        )

        let result = try executor.execute(plan)

        XCTAssertEqual(result.records.count, 2)
        XCTAssertEqual(result.aggregate?.function, .sum)
        XCTAssertEqual(result.aggregate?.field, "score")
        XCTAssertEqual(result.aggregate?.doubleValue ?? -1, 50, accuracy: 1e-9)
        XCTAssertEqual(result.audit.executionType, "aggregate_sum")
        XCTAssertEqual(result.audit.aggregateFunction, "sum")
        XCTAssertEqual(result.audit.aggregateField, "score")
    }

    func testExecutesCountAggregateFromExplicitSchemaDrivenPlan() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [
                .init(field: "label", op: .contains, value: .string("alpha")),
            ],
            aggregation: .init(function: .count)
        )

        let result = try executor.execute(plan)

        XCTAssertEqual(result.records.count, 2)
        XCTAssertEqual(result.aggregate?.function, .count)
        XCTAssertEqual(result.aggregate?.intValue, 2)
        XCTAssertEqual(result.audit.executionType, "aggregate_count")
        XCTAssertEqual(result.audit.aggregateFunction, "count")
    }

    func testAggregationDoesNotUseExactQueryDefaultLimit() throws {
        let store = try FactStore(
            path: tmpDir.appendingPathComponent("many-facts.sqlite3"),
            registry: registry
        )
        for index in 0..<120 {
            _ = try store.record(metric(
                score: 1,
                label: "bulk",
                bucket: "control",
                observedAt: Int64(index)
            ))
        }
        let executor = FactQueryPlanExecutor(store: store, registry: registry)

        let result = try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .count)
        ))

        XCTAssertEqual(result.aggregate?.intValue, 120)
        XCTAssertEqual(result.records.count, 120)
    }

    func testExactQueryUsesDefaultLimitAfterSchemaDrivenFiltering() throws {
        let store = try FactStore(
            path: tmpDir.appendingPathComponent("exact-limit-facts.sqlite3"),
            registry: registry
        )
        for index in 0..<120 {
            _ = try store.record(metric(
                score: 1,
                label: "bulk",
                bucket: "control",
                observedAt: Int64(index)
            ))
        }
        let executor = FactQueryPlanExecutor(store: store, registry: registry)

        let result = try executor.execute(FactQueryPlan(schema: Self.metricSchemaName))

        XCTAssertEqual(result.records.count, FactQueryPlanExecutor.defaultExactLimit)
    }

    func testAuditRoundTrips() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let result = try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "score", op: .between, value: .array([.int(10), .int(20)]))]
        ))

        let data = try JSONEncoder().encode(result.audit)
        let decoded = try JSONDecoder().decode(FactQueryPlanExecutor.Audit.self, from: data)

        XCTAssertEqual(decoded, result.audit)
        XCTAssertEqual(decoded.schemaVersion, FactQueryPlanExecutor.schemaVersion)
    }

    func testUnsupportedNamespaceFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(namespace: "app.metrics"))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedNamespace("app.metrics"))
        }
    }

    func testFilteredQueryRequiresSchema() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(filters: [
            .init(field: "label", op: .contains, value: .string("alpha")),
        ])

        XCTAssertThrowsError(try executor.execute(plan)) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .missingSchemaForFilteredQuery)
        }
    }

    func testUnsupportedFilterFieldFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let unknownField = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "unknown", op: .equals, value: .string("value"))]
        )
        let nonIndexedField = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "note", op: .contains, value: .string("hidden"))]
        )

        XCTAssertThrowsError(try executor.execute(unknownField)) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedFilterField("unknown"))
        }
        XCTAssertThrowsError(try executor.execute(nonIndexedField)) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedFilterField("note"))
        }
    }

    func testUnsupportedFilterOperatorFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "label", op: .notEquals, value: .string("alpha-early"))]
        )

        XCTAssertThrowsError(try executor.execute(plan)) { error in
            XCTAssertEqual(
                error as? FactQueryPlanExecutorError,
                .unsupportedFilterOperator(field: "label", op: .notEquals)
            )
        }
    }

    func testUnsupportedSortFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            sort: [.init(field: "score", direction: .descending)]
        )

        XCTAssertThrowsError(try executor.execute(plan)) { error in
            XCTAssertEqual(
                error as? FactQueryPlanExecutorError,
                .unsupportedSort([.init(field: "score", direction: .descending)])
            )
        }
    }

    func testUnsupportedGroupByAndRequestedFieldsFailClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(groupBy: ["bucket"]))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedGroupBy(["bucket"]))
        }

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(requestedFields: ["score"]))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedRequestedFields(["score"]))
        }
    }

    func testAggregationGuardsFailClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(
            aggregation: .init(function: .sum, field: "score")
        ))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .missingSchemaForAggregation)
        }

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .sum)
        ))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedAggregationField(nil))
        }

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .average, field: "score")
        ))) { error in
            XCTAssertEqual(
                error as? FactQueryPlanExecutorError,
                .unsupportedAggregationFunction(.average)
            )
        }

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .sum, field: "label")
        ))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedAggregationField("label"))
        }

        XCTAssertThrowsError(try executor.execute(FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .count, field: "score")
        ))) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedAggregationField("score"))
        }
    }

    func testAggregationLimitFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            aggregation: .init(function: .count),
            limit: 1
        )

        XCTAssertThrowsError(try executor.execute(plan)) { error in
            XCTAssertEqual(error as? FactQueryPlanExecutorError, .unsupportedLimitForAggregation(1))
        }
    }

    func testInvalidFilterValueFailsClosed() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "score", op: .equals, value: .string("10"))]
        )

        XCTAssertThrowsError(try executor.execute(plan)) { error in
            XCTAssertEqual(
                error as? FactQueryPlanExecutorError,
                .invalidFilterValue(field: "score", op: .equals)
            )
        }
    }

    func testBooleanEqualsFilterUsesSchemaFieldType() throws {
        let executor = FactQueryPlanExecutor(store: try populatedStore(), registry: registry)
        let plan = FactQueryPlan(
            schema: Self.metricSchemaName,
            filters: [.init(field: "enabled", op: .equals, value: .bool(true))]
        )

        let result = try executor.execute(plan)

        XCTAssertEqual(result.records.count, 3)
    }

    private func populatedStore() throws -> FactStore {
        let store = try FactStore(
            path: tmpDir.appendingPathComponent("facts.sqlite3"),
            registry: registry
        )
        _ = try store.record(metric(score: 10, label: "alpha-early", bucket: "control", observedAt: 100))
        _ = try store.record(metric(score: 20, label: "alpha-late", bucket: "control", observedAt: 200))
        _ = try store.record(metric(score: 30, label: "beta-late", bucket: "control", observedAt: 300))
        _ = try store.record(metric(score: 40, label: "gamma", bucket: "variant", observedAt: 400, enabled: false))
        return store
    }

    private func metric(
        score: Double,
        label: String,
        bucket: String,
        observedAt: Int64,
        enabled: Bool = true
    ) -> FactRecord {
        FactRecord.new(
            schemaName: Self.metricSchemaName,
            payload: [
                "score": .double(score),
                "label": .string(label),
                "bucket": .string(bucket),
                "observed_at": .int(observedAt),
                "enabled": .bool(enabled),
                "note": .string("hidden"),
            ]
        )
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
                "enabled": FactFieldDef(type: .bool, required: true, indexed: true),
                "note": FactFieldDef(type: .str),
            ],
            primaryTimeField: "observed_at",
            description: "Generic metric event fixture"
        )
    }()
}
