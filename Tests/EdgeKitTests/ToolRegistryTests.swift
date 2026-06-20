// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class ToolRegistryTests: XCTestCase {

    func test_toolMetadataCodable_roundTripsSchemaAndPermissions() throws {
        let metadata = ToolMetadata(
            name: "query_facts",
            description: "Read facts",
            argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
            resultSchema: .jsonSchema("{\"type\":\"object\"}"),
            permissions: [.readFacts, .readProfile],
            sensitivity: .sensitive,
            timeoutSeconds: 2.5,
            intentTags: [.exactFact, .aggregateFact]
        )

        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode(ToolMetadata.self, from: data)

        XCTAssertEqual(decoded, metadata)
        XCTAssertTrue(decoded.isReadOnly)
        XCTAssertTrue(decoded.supports(intentTag: .exactFact))
        XCTAssertFalse(decoded.supports(intentTag: .userProfile))
    }

    func test_toolPermission_readOnlyClassification() {
        XCTAssertTrue(ToolPermission.readFacts.isReadOnly)
        XCTAssertTrue(ToolPermission.readEvents.isReadOnly)
        XCTAssertTrue(ToolPermission.readProfile.isReadOnly)
        XCTAssertTrue(ToolPermission.fileRead.isReadOnly)

        XCTAssertFalse(ToolPermission.writeFacts.isReadOnly)
        XCTAssertFalse(ToolPermission.writeEvents.isReadOnly)
        XCTAssertFalse(ToolPermission.writeProfile.isReadOnly)
        XCTAssertFalse(ToolPermission.network.isReadOnly)
        XCTAssertFalse(ToolPermission.fileWrite.isReadOnly)
        XCTAssertFalse(ToolPermission.appAction.isReadOnly)
    }

    func test_auditValueObjectCodableRoundTripsNestedJSON() throws {
        let value = AuditValue.object([
            "groups": .array([
                .object([
                    "field": .string("score"),
                    "value": .double(2.5),
                ]),
            ]),
        ])

        let data = try JSONEncoder().encode(value)
        let decoded = try JSONDecoder().decode(AuditValue.self, from: data)

        XCTAssertEqual(decoded, value)
    }

    func test_registrySchemas_areStableAndFilterByRequestedNames() {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(name: "b")) { _ in "b" })
        registry.register(RegisteredTool(metadata: metadata(name: "a")) { _ in "a" })

        XCTAssertEqual(registry.allSchemas().map(\.name), ["a", "b"])
        XCTAssertEqual(
            registry.schemas(forNames: ["b", "missing", "a"]).map(\.name),
            ["b", "a"]
        )
    }

    func test_registrySchemasForIntentTag_trimByDeclaredIntentTags() {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(
            name: "fact",
            intentTags: [.exactFact, .aggregateFact]
        )) { _ in "fact" })
        registry.register(RegisteredTool(metadata: metadata(
            name: "profile",
            intentTags: [.userProfile]
        )) { _ in "profile" })
        registry.register(RegisteredTool(metadata: metadata(
            name: "action",
            intentTags: [.appAction]
        )) { _ in "action" })

        XCTAssertEqual(
            registry.schemas(forIntentTag: .exactFact).map(\.name),
            ["fact"]
        )
        XCTAssertEqual(
            registry.schemas(forIntentTag: .aggregateFact).map(\.name),
            ["fact"]
        )
        XCTAssertEqual(
            registry.schemas(forIntentTag: .userProfile).map(\.name),
            ["profile"]
        )
        XCTAssertEqual(registry.schemas(forIntentTag: .baseChat), [])
    }

    func test_registrySchemasForMixedIntent_unionCandidateToolsWithoutBaseChat() {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(
            name: "fact",
            intentTags: [.exactFact]
        )) { _ in "fact" })
        registry.register(RegisteredTool(metadata: metadata(
            name: "profile",
            intentTags: [.userProfile]
        )) { _ in "profile" })
        registry.register(RegisteredTool(metadata: metadata(
            name: "action",
            intentTags: [.appAction]
        )) { _ in "action" })

        let intent = PersonalIntent.mixed(candidates: [
            .exactFact(plan: FactQueryPlan()),
            .userProfile(detail: ProfileDetail(kind: .summary)),
            .baseChat,
        ])

        XCTAssertEqual(
            registry.schemas(forIntent: intent).map(\.name),
            ["fact", "profile"]
        )
    }

    func test_registryExecute_decodesAuditValueArguments() async throws {
        let registry = ToolRegistry()
        registry.register(EchoTool())

        let result = try await registry.execute(ToolCallPlan(
            toolName: "echo",
            arguments: [
                "text": .string("hello"),
                "count": .int(3),
                "flags": .array([.bool(true), .bool(false)]),
            ]
        ))

        XCTAssertEqual(result, "hello:3:2")
    }

    func test_registryExecute_normalizesStringArgumentsAgainstJSONSchema() async throws {
        let registry = ToolRegistry()
        registry.register(CoercionTool())

        let result = try await registry.execute(ToolCallPlan(
            toolName: "coerce",
            arguments: [
                "ascending": .string("False"),
                "limit": .string("1"),
                "score": .string("2.5"),
            ]
        ))

        XCTAssertEqual(result, "all:false:1:2.5")
    }

    func test_registryExecute_encodesStructuredOutputAsJSON() async throws {
        let registry = ToolRegistry()
        registry.register(SummaryTool())

        let result = try await registry.execute(ToolCallPlan(
            toolName: "summary",
            arguments: ["value": .double(2.5)]
        ))

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(result.utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["ok"] as? Bool, true)
        XCTAssertEqual(object["value"] as? Double, 2.5)
    }

    func test_registryExecute_missingToolThrows() async {
        let registry = ToolRegistry()

        do {
            _ = try await registry.execute(ToolCallPlan(toolName: "missing"))
            XCTFail("Expected missing tool failure")
        } catch let error as ToolRegistryError {
            XCTAssertEqual(error, .toolNotFound("missing"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func test_registryExecute_acceptsToolCallBridge() async throws {
        let registry = ToolRegistry()
        registry.register(EchoTool())

        let call = ToolCall(function: ToolCall.Function(
            name: "echo",
            arguments: [
                "text": "hello",
                "count": 3,
                "flags": [true, false],
            ] as [String: any Sendable]
        ))

        let result = try await registry.execute(call)

        XCTAssertEqual(result, "hello:3:2")
    }

    func test_toolCallBridgeAcceptsNestedObjectArguments() throws {
        let call = ToolCall(function: ToolCall.Function(
            name: "nested",
            arguments: [
                "filter": [
                    "field": "score",
                    "value": 2.5,
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ))

        let plan = try ToolCallPlan(toolCall: call)

        XCTAssertEqual(plan.toolName, "nested")
        XCTAssertEqual(plan.arguments["filter"], .object([
            "field": .string("score"),
            "value": .double(2.5),
        ]))
    }

    func test_registryExecuteDecodesNestedObjectsInsideArraysFromToolCallBridge() async throws {
        let registry = ToolRegistry()
        registry.register(NestedTool())
        let data = Data("""
        {
          "function": {
            "name": "nested",
            "arguments": {
              "groups": [
                [
                  {
                    "field": "score",
                    "value": 2.5
                  }
                ]
              ]
            }
          }
        }
        """.utf8)
        let call = try JSONDecoder().decode(ToolCall.self, from: data)

        let result = try await registry.execute(call)

        XCTAssertEqual(result, "score:2.5")
    }

    func test_toolCallBridgeReportsUnsupportedNestedValuePath() {
        let call = ToolCall(function: ToolCall.Function(
            name: "nested",
            arguments: [
                "filter": [
                    "field": "score",
                    "unsupported": Date(),
                ] as [String: any Sendable],
            ] as [String: any Sendable]
        ))

        XCTAssertThrowsError(try ToolCallPlan(toolCall: call)) { error in
            XCTAssertEqual(
                error as? ToolArgumentConversionError,
                .valueUnsupported("filter.unsupported")
            )
        }
    }

    func test_toolMetadataBuildsToolSpecFromJSONSchema() throws {
        let metadata = ToolMetadata(
            name: "query_facts",
            description: "Read facts",
            argumentsSchema: .jsonSchema("""
            {
              "type": "object",
              "properties": {
                "query": {
                  "type": "string",
                  "description": "Search text"
                },
                "limit": {
                  "type": "integer"
                }
              },
              "required": ["query"]
            }
            """),
            permissions: [.readFacts]
        )

        let spec = try metadata.toolSpec()
        XCTAssertEqual(spec["type"] as? String, "function")

        let function = try XCTUnwrap(spec["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "query_facts")
        XCTAssertEqual(function["description"] as? String, "Read facts")

        let parameters = try XCTUnwrap(function["parameters"] as? [String: any Sendable])
        XCTAssertEqual(parameters["type"] as? String, "object")
        XCTAssertEqual(parameters["required"] as? [String], ["query"])

        let properties = try XCTUnwrap(parameters["properties"] as? [String: any Sendable])
        let query = try XCTUnwrap(properties["query"] as? [String: any Sendable])
        XCTAssertEqual(query["type"] as? String, "string")
        XCTAssertEqual(query["description"] as? String, "Search text")
    }

    func test_toolMetadataBuildsToolSpecWithNestedObjectSchema() throws {
        let metadata = ToolMetadata(
            name: "query_facts",
            description: "Read facts",
            argumentsSchema: .jsonSchema("""
            {
              "type": "object",
              "properties": {
                "filter": {
                  "type": "object",
                  "properties": {
                    "field": { "type": "string" },
                    "value": { "type": "number" }
                  },
                  "required": ["field"]
                }
              }
            }
            """),
            permissions: [.readFacts]
        )

        let spec = try metadata.toolSpec()
        let function = try XCTUnwrap(spec["function"] as? [String: any Sendable])
        let parameters = try XCTUnwrap(function["parameters"] as? [String: any Sendable])
        let properties = try XCTUnwrap(parameters["properties"] as? [String: any Sendable])
        let filter = try XCTUnwrap(properties["filter"] as? [String: any Sendable])
        let filterProperties = try XCTUnwrap(filter["properties"] as? [String: any Sendable])
        let field = try XCTUnwrap(filterProperties["field"] as? [String: any Sendable])
        let value = try XCTUnwrap(filterProperties["value"] as? [String: any Sendable])

        XCTAssertEqual(filter["type"] as? String, "object")
        XCTAssertEqual(filter["required"] as? [String], ["field"])
        XCTAssertEqual(field["type"] as? String, "string")
        XCTAssertEqual(value["type"] as? String, "number")
    }

    func test_registryBuildsToolSpecsForRequestedNames() throws {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(name: "b")) { _ in "b" })
        registry.register(RegisteredTool(metadata: metadata(name: "a")) { _ in "a" })

        let specs = try registry.toolSpecs(forNames: ["b", "missing", "a"])
        let names = try specs.map { spec in
            let function = try XCTUnwrap(spec["function"] as? [String: any Sendable])
            return try XCTUnwrap(function["name"] as? String)
        }

        XCTAssertEqual(names, ["b", "a"])
    }

    func test_registryBuildsToolSpecsForIntent() throws {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(
            name: "fact",
            intentTags: [.exactFact]
        )) { _ in "fact" })
        registry.register(RegisteredTool(metadata: metadata(
            name: "profile",
            intentTags: [.userProfile]
        )) { _ in "profile" })

        let specs = try registry.toolSpecs(forIntentTag: .exactFact)
        let function = try XCTUnwrap(specs.first?["function"] as? [String: any Sendable])

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(function["name"] as? String, "fact")
        XCTAssertEqual(try registry.toolSpecs(forIntentTag: .baseChat).count, 0)
    }

    func test_toolSchemaSnapshot_isStableAcrossRegistrationOrder() throws {
        let registryA = ToolRegistry()
        registryA.register(RegisteredTool(metadata: metadata(name: "b")) { _ in "b" })
        registryA.register(RegisteredTool(metadata: metadata(name: "a")) { _ in "a" })

        let registryB = ToolRegistry()
        registryB.register(RegisteredTool(metadata: metadata(name: "a")) { _ in "a" })
        registryB.register(RegisteredTool(metadata: metadata(name: "b")) { _ in "b" })

        let snapshotA = try registryA.toolSchemaSnapshot()
        let snapshotB = try registryB.toolSchemaSnapshot()
        let json = try XCTUnwrap(snapshotA.jsonString)

        XCTAssertEqual(snapshotA.export.schemaVersion, "edgestudio.tool_schema_export.v1")
        XCTAssertEqual(snapshotA.export.tools.map(\.name), ["a", "b"])
        XCTAssertEqual(snapshotA.jsonData, snapshotB.jsonData)
        XCTAssertEqual(snapshotA.sha256, snapshotB.sha256)
        XCTAssertEqual(snapshotA.sha256.count, 64)
        XCTAssertTrue(snapshotA.sha256.allSatisfy { $0.isHexDigit })
        XCTAssertTrue(json.contains(#""schema_version":"edgestudio.tool_schema_export.v1""#))
        XCTAssertTrue(json.range(of: #""name":"a""#)!.lowerBound < json.range(of: #""name":"b""#)!.lowerBound)
    }

    func test_toolSchemaSnapshot_filtersRequestedNamesAndChangesHash() throws {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(metadata: metadata(name: "a")) { _ in "a" })
        registry.register(RegisteredTool(metadata: metadata(name: "b")) { _ in "b" })

        let all = try registry.toolSchemaSnapshot()
        let filtered = try registry.toolSchemaSnapshot(forNames: ["b", "missing", "a"])

        XCTAssertEqual(filtered.export.tools.map(\.name), ["b", "a"])
        XCTAssertNotEqual(filtered.jsonData, all.jsonData)
        XCTAssertNotEqual(filtered.sha256, all.sha256)
    }

    func test_toolSpecRejectsInvalidJSONSchema() {
        let metadata = ToolMetadata(
            name: "bad",
            description: "Bad schema",
            argumentsSchema: .jsonSchema("not json"),
            permissions: []
        )

        XCTAssertThrowsError(try metadata.toolSpec()) { error in
            XCTAssertEqual(error as? ToolSpecConversionError, .invalidJSONSchema("not json"))
        }
    }

    func test_toolSpecRejectsNullValues() {
        let metadata = ToolMetadata(
            name: "bad",
            description: "Bad schema",
            argumentsSchema: .jsonSchema("""
            {
              "type": "object",
              "properties": {
                "query": {
                  "type": "string",
                  "default": null
                }
              }
            }
            """),
            permissions: []
        )

        XCTAssertThrowsError(try metadata.toolSpec()) { error in
            XCTAssertEqual(error as? ToolSpecConversionError, .nullUnsupportedInToolSpec)
        }
    }

    private func metadata(
        name: String,
        intentTags: [PersonalIntentTag] = []
    ) -> ToolMetadata {
        ToolMetadata(
            name: name,
            description: "Test tool",
            argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
            permissions: [.readFacts],
            intentTags: intentTags
        )
    }
}

private struct EchoInput: Decodable, Sendable {
    let text: String
    let count: Int
    let flags: [Bool]
}

private struct EchoTool: EdgeTool {
    let metadata = ToolMetadata(
        name: "echo",
        description: "Echo arguments",
        argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
        permissions: [.readFacts]
    )

    func execute(_ input: EchoInput) async throws -> String {
        "\(input.text):\(input.count):\(input.flags.count)"
    }
}

private struct SummaryInput: Decodable, Sendable {
    let value: Double
}

private struct SummaryOutput: Encodable, Sendable {
    let ok: Bool
    let value: Double
}

private struct SummaryTool: EdgeTool {
    let metadata = ToolMetadata(
        name: "summary",
        description: "Return a structured payload",
        argumentsSchema: .jsonSchema("{\"type\":\"object\"}"),
        permissions: [.readFacts]
    )

    func execute(_ input: SummaryInput) async throws -> SummaryOutput {
        SummaryOutput(ok: true, value: input.value)
    }
}

private struct NestedFilter: Decodable, Sendable {
    let field: String
    let value: Double
}

private struct NestedInput: Decodable, Sendable {
    let groups: [[NestedFilter]]
}

private struct NestedTool: EdgeTool {
    let metadata = ToolMetadata(
        name: "nested",
        description: "Decode nested generic arguments",
        argumentsSchema: .jsonSchema("""
        {
          "type": "object",
          "properties": {
            "groups": {
              "type": "array",
              "items": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "field": { "type": "string" },
                    "value": { "type": "number" }
                  },
                  "required": ["field", "value"]
                }
              }
            }
          },
          "required": ["groups"]
        }
        """),
        permissions: [.readFacts]
    )

    func execute(_ input: NestedInput) async throws -> String {
        guard let first = input.groups.first?.first else {
            return "empty"
        }
        return "\(first.field):\(first.value)"
    }
}

private struct CoercionInput: Decodable, Sendable {
    let timeRange: String
    let ascending: Bool
    let limit: Int
    let score: Double
}

private struct CoercionTool: EdgeTool {
    let metadata = ToolMetadata(
        name: "coerce",
        description: "Check schema-based argument normalization",
        argumentsSchema: .jsonSchema("""
        {
          "type": "object",
          "properties": {
            "timeRange": { "type": "string", "default": "all" },
            "ascending": { "type": "boolean" },
            "limit": { "type": "integer" },
            "score": { "type": "number" }
          }
        }
        """),
        permissions: [.readFacts]
    )

    func execute(_ input: CoercionInput) async throws -> String {
        "\(input.timeRange):\(input.ascending):\(input.limit):\(input.score)"
    }
}
