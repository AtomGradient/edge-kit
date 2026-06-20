// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Explicit-registration adapter for schema-driven fact query plans.
///
/// The adapter does not infer plans from natural language, choose schemas,
/// synthesize fields, or register itself globally. Callers provide the store
/// and schema registry, then register the returned tool when their runtime
/// policy allows fact recall.
public struct FactQueryPlanToolAdapter: Sendable {
    public struct Output: Sendable, Codable, Equatable {
        public let records: [Record]
        public let aggregate: FactQueryPlanExecutor.AggregateResult?
        public let audit: FactQueryPlanExecutor.Audit
    }

    public struct Record: Sendable, Codable, Equatable {
        public let id: String
        public let schemaName: String
        public let payload: [String: AuditValue]
        public let createdAt: Int64
        public let sensitivity: String
        public let ttlSeconds: Int64?
        public let derivedFrom: String?
        public let sourceType: String
    }

    public static let defaultToolName = "edge_fact_query_plan"

    public static var defaultMetadata: ToolMetadata {
        ToolMetadata(
            name: defaultToolName,
            description: "Execute an explicit schema-driven fact query plan.",
            argumentsSchema: .jsonSchema(defaultArgumentsSchema),
            resultSchema: .jsonSchema(defaultResultSchema),
            permissions: [.readFacts],
            sensitivity: .sensitive,
            intentTags: [.exactFact, .aggregateFact]
        )
    }

    public let metadata: ToolMetadata
    private let executor: FactQueryPlanExecutor

    public init(
        store: FactStore,
        registry: FactSchemaRegistry = .shared,
        metadata: ToolMetadata = Self.defaultMetadata
    ) {
        self.metadata = metadata
        self.executor = FactQueryPlanExecutor(store: store, registry: registry)
    }

    public func registeredTool() -> RegisteredTool {
        RegisteredTool(metadata: metadata) { arguments in
            let plan = try Self.decodePlan(from: arguments)
            let result = try executor.execute(plan)
            return try ToolJSON.encodeOutput(Self.output(from: result))
        }
    }

    public static func registeredTool(
        store: FactStore,
        registry: FactSchemaRegistry = .shared,
        metadata: ToolMetadata = Self.defaultMetadata
    ) -> RegisteredTool {
        Self(store: store, registry: registry, metadata: metadata).registeredTool()
    }

    private static func decodePlan(from arguments: [String: AuditValue]) throws -> FactQueryPlan {
        if case .object(let planObject) = arguments["plan"] {
            return try decodePlanFields(from: planObject)
        }
        return try decodePlanFields(from: arguments)
    }

    private static func decodePlanFields(from arguments: [String: AuditValue]) throws -> FactQueryPlan {
        FactQueryPlan(
            namespace: try decodeIfPresent(String.self, from: arguments["namespace"]),
            schema: try decodeIfPresent(String.self, from: arguments["schema"]),
            filters: try decodeIfPresent([FactQueryFilter].self, from: arguments["filters"]) ?? [],
            aggregation: try decodeIfPresent(FactAggregation.self, from: arguments["aggregation"]),
            groupBy: try decodeIfPresent([String].self, from: arguments["groupBy"]) ?? [],
            sort: try decodeIfPresent([FactSort].self, from: arguments["sort"]) ?? [],
            limit: try decodeIfPresent(Int.self, from: arguments["limit"]),
            requestedFields: try decodeIfPresent([String].self, from: arguments["requestedFields"]) ?? []
        )
    }

    private static func decodeIfPresent<T: Decodable>(
        _ type: T.Type,
        from value: AuditValue?
    ) throws -> T? {
        guard let value, value != .null else {
            return nil
        }
        return try decode(type, from: value)
    }

    private static func decode<T: Decodable>(
        _ type: T.Type,
        from value: AuditValue
    ) throws -> T {
        let data = try JSONEncoder().encode(value)
        return try JSONDecoder().decode(type, from: data)
    }

    private static func decode<T: Decodable>(
        _ type: T.Type,
        from arguments: [String: AuditValue]
    ) throws -> T {
        let data = try JSONEncoder().encode(arguments)
        return try JSONDecoder().decode(type, from: data)
    }

    private static func output(from result: FactQueryPlanExecutor.Result) -> Output {
        Output(
            records: result.records.map(record(from:)),
            aggregate: result.aggregate,
            audit: result.audit
        )
    }

    private static func record(from record: FactRecord) -> Record {
        Record(
            id: record.id,
            schemaName: record.schemaName,
            payload: record.payload.mapValues(auditValue(from:)),
            createdAt: record.createdAt,
            sensitivity: record.sensitivity,
            ttlSeconds: record.ttlSeconds,
            derivedFrom: record.derivedFrom,
            sourceType: record.sourceType
        )
    }

    private static func auditValue(from value: FactValue) -> AuditValue {
        switch value {
        case .string(let value):
            return .string(value)
        case .int(let value):
            if value >= Int64(Int.min), value <= Int64(Int.max) {
                return .int(Int(value))
            }
            return .double(Double(value))
        case .double(let value):
            return .double(value)
        case .bool(let value):
            return .bool(value)
        case .null:
            return .null
        }
    }

    private static let defaultArgumentsSchema = """
    {
      "type": "object",
      "properties": {
        "plan": {
          "type": "object",
          "description": "Optional wrapper for a FactQueryPlan. Callers may also pass plan fields at the top level."
        },
        "namespace": { "type": "string" },
        "schema": { "type": "string" },
        "filters": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "field": { "type": "string" },
              "op": { "type": "string" },
              "value": {}
            },
            "required": ["field", "op", "value"]
          }
        },
        "aggregation": {
          "type": "object",
          "properties": {
            "function": { "type": "string" },
            "field": { "type": "string" }
          },
          "required": ["function"]
        },
        "groupBy": {
          "type": "array",
          "items": { "type": "string" }
        },
        "sort": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "field": { "type": "string" },
              "direction": { "type": "string" }
            },
            "required": ["field"]
          }
        },
        "limit": { "type": "integer" },
        "requestedFields": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    }
    """

    private static let defaultResultSchema = """
    {
      "type": "object",
      "properties": {
        "records": { "type": "array" },
        "aggregate": { "type": ["object", "null"] },
        "audit": { "type": "object" }
      },
      "required": ["records", "audit"]
    }
    """
}
