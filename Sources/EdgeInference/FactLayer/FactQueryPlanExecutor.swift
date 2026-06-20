// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum FactQueryPlanExecutorError: Error, Equatable {
    case unsupportedNamespace(String)
    case unsupportedGroupBy([String])
    case unsupportedRequestedFields([String])
    case unsupportedSort([FactSort])
    case invalidLimit(Int)
    case missingSchemaForFilteredQuery
    case unsupportedFilterField(String)
    case unsupportedFilterOperator(field: String, op: FactFilterOperator)
    case invalidFilterValue(field: String, op: FactFilterOperator)
    case missingSchemaForAggregation
    case unsupportedAggregationFunction(FactAggregationFunction)
    case unsupportedAggregationField(String?)
    case unsupportedLimitForAggregation(Int)
}

/// Executes explicit structured fact query plans against registered schemas.
///
/// This executor does not parse natural language, classify user input, create
/// tool calls, execute tools, or hard-code application/domain fields.
/// Unsupported plan shapes fail closed.
public struct FactQueryPlanExecutor: Sendable {
    public struct Result: Sendable, Equatable {
        public let records: [FactRecord]
        public let aggregate: AggregateResult?
        public let audit: Audit
    }

    public struct AggregateResult: Sendable, Codable, Equatable {
        public let function: FactAggregationFunction
        public let field: String?
        public let doubleValue: Double?
        public let intValue: Int?

        public init(
            function: FactAggregationFunction,
            field: String?,
            doubleValue: Double? = nil,
            intValue: Int? = nil
        ) {
            self.function = function
            self.field = field
            self.doubleValue = doubleValue
            self.intValue = intValue
        }
    }

    public struct Audit: Sendable, Codable, Equatable {
        public let schemaVersion: String
        public let querySource: String
        public let executionType: String
        public let schema: String?
        public let recordCount: Int
        public let aggregateFunction: String?
        public let aggregateField: String?
        public let planExecuted: Bool
        public let toolsExecuted: Bool
        public let toolCallsCreated: Bool
        public let usesRegexOrKeywordFactDetection: Bool
        public let reason: String
    }

    public static let schemaVersion = "edge.fact_query_plan_executor.v1"
    public static let defaultExactLimit = 100

    private let store: FactStore
    private let registry: FactSchemaRegistry

    public init(
        store: FactStore,
        registry: FactSchemaRegistry = .shared
    ) {
        self.store = store
        self.registry = registry
    }

    public func execute(_ plan: FactQueryPlan) throws -> Result {
        try validateBaseShape(plan)
        let schema = try schemaDefinition(for: plan)
        try validateSort(plan.sort, schema: schema)

        let matchedRecords = try matchingRecords(for: plan, schema: schema)

        guard let aggregation = plan.aggregation else {
            let limitedRecords = try applyExactLimit(plan.limit, to: matchedRecords)
            return Result(
                records: limitedRecords,
                aggregate: nil,
                audit: audit(
                    plan: plan,
                    executionType: "exact_query",
                    records: limitedRecords,
                    aggregate: nil,
                    reason: "executed explicit schema-driven fact query plan"
                )
            )
        }

        guard plan.limit == nil else {
            throw FactQueryPlanExecutorError.unsupportedLimitForAggregation(plan.limit ?? 0)
        }
        guard schema != nil else {
            throw FactQueryPlanExecutorError.missingSchemaForAggregation
        }

        switch aggregation.function {
        case .count:
            guard aggregation.field == nil else {
                throw FactQueryPlanExecutorError.unsupportedAggregationField(aggregation.field)
            }
            let aggregate = AggregateResult(
                function: .count,
                field: nil,
                intValue: matchedRecords.count
            )
            return Result(
                records: matchedRecords,
                aggregate: aggregate,
                audit: audit(
                    plan: plan,
                    executionType: "aggregate_count",
                    records: matchedRecords,
                    aggregate: aggregate,
                    reason: "counted records from explicit schema-driven fact query plan"
                )
            )
        case .sum:
            guard let field = aggregation.field else {
                throw FactQueryPlanExecutorError.unsupportedAggregationField(nil)
            }
            try validateNumericAggregationField(field, schema: schema)
            let total = matchedRecords.reduce(0.0) { partial, record in
                partial + (record.payload[field]?.doubleValue ?? 0)
            }
            let aggregate = AggregateResult(
                function: .sum,
                field: field,
                doubleValue: total
            )
            return Result(
                records: matchedRecords,
                aggregate: aggregate,
                audit: audit(
                    plan: plan,
                    executionType: "aggregate_sum",
                    records: matchedRecords,
                    aggregate: aggregate,
                    reason: "summed schema field from explicit schema-driven fact query plan"
                )
            )
        case .average, .minimum, .maximum:
            throw FactQueryPlanExecutorError.unsupportedAggregationFunction(aggregation.function)
        }
    }

    private func validateBaseShape(_ plan: FactQueryPlan) throws {
        if let namespace = plan.namespace {
            throw FactQueryPlanExecutorError.unsupportedNamespace(namespace)
        }
        if !plan.groupBy.isEmpty {
            throw FactQueryPlanExecutorError.unsupportedGroupBy(plan.groupBy)
        }
        if !plan.requestedFields.isEmpty {
            throw FactQueryPlanExecutorError.unsupportedRequestedFields(plan.requestedFields)
        }
        if let limit = plan.limit, limit <= 0 {
            throw FactQueryPlanExecutorError.invalidLimit(limit)
        }
    }

    private func schemaDefinition(for plan: FactQueryPlan) throws -> FactSchema? {
        if let schemaName = plan.schema {
            return try registry.get(schemaName)
        }
        if !plan.filters.isEmpty {
            throw FactQueryPlanExecutorError.missingSchemaForFilteredQuery
        }
        if plan.aggregation != nil {
            throw FactQueryPlanExecutorError.missingSchemaForAggregation
        }
        return nil
    }

    private func validateSort(_ sort: [FactSort], schema: FactSchema?) throws {
        guard !sort.isEmpty else {
            return
        }
        guard sort.count == 1,
              sort[0].direction == .descending,
              let schema,
              let primaryTimeField = schema.primaryTimeField,
              sort[0].field == primaryTimeField,
              schema.fields[primaryTimeField]?.indexed == true else {
            throw FactQueryPlanExecutorError.unsupportedSort(sort)
        }
    }

    private func matchingRecords(
        for plan: FactQueryPlan,
        schema: FactSchema?
    ) throws -> [FactRecord] {
        var queryFilter = FactStore.QueryFilter(schema: plan.schema)
        queryFilter.limit = Int.max
        let candidates = try store.query(queryFilter)

        var matched: [FactRecord] = []
        for record in candidates {
            if try matches(record, filters: plan.filters, schema: schema) {
                matched.append(record)
            }
        }

        return try sort(matched, sort: plan.sort, schema: schema)
    }

    private func matches(
        _ record: FactRecord,
        filters: [FactQueryFilter],
        schema: FactSchema?
    ) throws -> Bool {
        for filter in filters {
            guard try matches(record, filter: filter, schema: schema) else {
                return false
            }
        }
        return true
    }

    private func matches(
        _ record: FactRecord,
        filter: FactQueryFilter,
        schema: FactSchema?
    ) throws -> Bool {
        let field = try fieldDefinition(filter.field, schema: schema)
        let value = record.payload[filter.field]
        switch field.type {
        case .str:
            return try matchesString(value, filter: filter)
        case .int, .datetime:
            return try matchesInt(value, filter: filter)
        case .float:
            return try matchesDouble(value, filter: filter)
        case .bool:
            return try matchesBool(value, filter: filter)
        }
    }

    private func fieldDefinition(_ field: String, schema: FactSchema?) throws -> FactFieldDef {
        guard let schema,
              let definition = schema.fields[field],
              definition.indexed else {
            throw FactQueryPlanExecutorError.unsupportedFilterField(field)
        }
        return definition
    }

    private func matchesString(_ value: FactValue?, filter: FactQueryFilter) throws -> Bool {
        guard let actual = value?.stringValue else {
            return false
        }
        let expected = try string(from: filter)
        switch filter.op {
        case .equals:
            return actual == expected
        case .contains:
            return actual.localizedCaseInsensitiveContains(expected)
        default:
            throw FactQueryPlanExecutorError.unsupportedFilterOperator(field: filter.field, op: filter.op)
        }
    }

    private func matchesInt(_ value: FactValue?, filter: FactQueryFilter) throws -> Bool {
        guard let actual = value?.intValue else {
            return false
        }
        switch filter.op {
        case .equals:
            let expected = try int64(from: filter)
            return actual == expected
        case .greaterThanOrEqual:
            let expected = try int64(from: filter)
            return actual >= expected
        case .lessThanOrEqual:
            let expected = try int64(from: filter)
            return actual <= expected
        case .between:
            let range = try int64Range(from: filter)
            return actual >= range.lower && actual <= range.upper
        default:
            throw FactQueryPlanExecutorError.unsupportedFilterOperator(field: filter.field, op: filter.op)
        }
    }

    private func matchesDouble(_ value: FactValue?, filter: FactQueryFilter) throws -> Bool {
        guard let actual = value?.doubleValue else {
            return false
        }
        switch filter.op {
        case .equals:
            let expected = try number(from: filter)
            return actual == expected
        case .greaterThanOrEqual:
            let expected = try number(from: filter)
            return actual >= expected
        case .lessThanOrEqual:
            let expected = try number(from: filter)
            return actual <= expected
        case .between:
            let range = try numberRange(from: filter)
            return actual >= range.lower && actual <= range.upper
        default:
            throw FactQueryPlanExecutorError.unsupportedFilterOperator(field: filter.field, op: filter.op)
        }
    }

    private func matchesBool(_ value: FactValue?, filter: FactQueryFilter) throws -> Bool {
        guard filter.op == .equals else {
            throw FactQueryPlanExecutorError.unsupportedFilterOperator(field: filter.field, op: filter.op)
        }
        guard case .bool(let expected) = filter.value else {
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
        return value?.boolValue == expected
    }

    private func validateNumericAggregationField(_ field: String, schema: FactSchema?) throws {
        guard let schema,
              let definition = schema.fields[field],
              definition.indexed,
              definition.type == .float || definition.type == .int else {
            throw FactQueryPlanExecutorError.unsupportedAggregationField(field)
        }
    }

    private func applyExactLimit(_ limit: Int?, to records: [FactRecord]) throws -> [FactRecord] {
        let resolvedLimit = limit ?? Self.defaultExactLimit
        guard resolvedLimit > 0 else {
            throw FactQueryPlanExecutorError.invalidLimit(resolvedLimit)
        }
        return Array(records.prefix(resolvedLimit))
    }

    private func sort(
        _ records: [FactRecord],
        sort: [FactSort],
        schema: FactSchema?
    ) throws -> [FactRecord] {
        guard !sort.isEmpty else {
            return records
        }
        guard let primaryTimeField = schema?.primaryTimeField else {
            throw FactQueryPlanExecutorError.unsupportedSort(sort)
        }
        return records.sorted {
            ($0.payload[primaryTimeField]?.intValue ?? Int64.min)
                > ($1.payload[primaryTimeField]?.intValue ?? Int64.min)
        }
    }

    private func audit(
        plan: FactQueryPlan,
        executionType: String,
        records: [FactRecord],
        aggregate: AggregateResult?,
        reason: String
    ) -> Audit {
        Audit(
            schemaVersion: Self.schemaVersion,
            querySource: "explicit_schema_driven_fact_query_plan",
            executionType: executionType,
            schema: plan.schema,
            recordCount: records.count,
            aggregateFunction: aggregate?.function.rawValue,
            aggregateField: aggregate?.field,
            planExecuted: true,
            toolsExecuted: false,
            toolCallsCreated: false,
            usesRegexOrKeywordFactDetection: false,
            reason: reason
        )
    }

    private func string(from filter: FactQueryFilter) throws -> String {
        guard case .string(let value) = filter.value else {
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
        return value
    }

    private func number(from filter: FactQueryFilter) throws -> Double {
        switch filter.value {
        case .int(let value):
            return Double(value)
        case .double(let value):
            return value
        default:
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
    }

    private func int64(from filter: FactQueryFilter) throws -> Int64 {
        switch filter.value {
        case .int(let value):
            return Int64(value)
        default:
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
    }

    private func numberRange(from filter: FactQueryFilter) throws -> (lower: Double, upper: Double) {
        guard case .array(let values) = filter.value, values.count == 2 else {
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
        return (
            lower: try number(from: FactQueryFilter(field: filter.field, op: filter.op, value: values[0])),
            upper: try number(from: FactQueryFilter(field: filter.field, op: filter.op, value: values[1]))
        )
    }

    private func int64Range(from filter: FactQueryFilter) throws -> (lower: Int64, upper: Int64) {
        guard case .array(let values) = filter.value, values.count == 2 else {
            throw FactQueryPlanExecutorError.invalidFilterValue(field: filter.field, op: filter.op)
        }
        return (
            lower: try int64(from: FactQueryFilter(field: filter.field, op: filter.op, value: values[0])),
            upper: try int64(from: FactQueryFilter(field: filter.field, op: filter.op, value: values[1]))
        )
    }
}
