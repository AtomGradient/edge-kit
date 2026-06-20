// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum ToolArgumentSchema: Sendable, Codable, Equatable {
    case jsonSchema(String)
    case auditMap([String: AuditValue])

    private enum CodingKeys: String, CodingKey {
        case kind
        case jsonSchema = "json_schema"
        case auditMap = "audit_map"
    }

    private enum Kind: String, Sendable, Codable {
        case jsonSchema = "json_schema"
        case auditMap = "audit_map"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .jsonSchema:
            self = .jsonSchema(try container.decode(String.self, forKey: .jsonSchema))
        case .auditMap:
            self = .auditMap(try container.decode([String: AuditValue].self, forKey: .auditMap))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .jsonSchema(let value):
            try container.encode(Kind.jsonSchema, forKey: .kind)
            try container.encode(value, forKey: .jsonSchema)
        case .auditMap(let value):
            try container.encode(Kind.auditMap, forKey: .kind)
            try container.encode(value, forKey: .auditMap)
        }
    }
}

public struct ToolMetadata: Sendable, Codable, Equatable {
    public var name: String
    public var description: String
    public var argumentsSchema: ToolArgumentSchema
    public var resultSchema: ToolArgumentSchema?
    public var permissions: [ToolPermission]
    public var sensitivity: ToolSensitivity
    public var timeoutSeconds: Double?
    public var intentTags: [PersonalIntentTag]

    public init(
        name: String,
        description: String,
        argumentsSchema: ToolArgumentSchema,
        resultSchema: ToolArgumentSchema? = nil,
        permissions: [ToolPermission] = [],
        sensitivity: ToolSensitivity = .normal,
        timeoutSeconds: Double? = nil,
        intentTags: [PersonalIntentTag] = []
    ) {
        self.name = name
        self.description = description
        self.argumentsSchema = argumentsSchema
        self.resultSchema = resultSchema
        self.permissions = permissions
        self.sensitivity = sensitivity
        self.timeoutSeconds = timeoutSeconds
        self.intentTags = intentTags
    }

    public var isReadOnly: Bool {
        permissions.allSatisfy(\.isReadOnly)
    }

    public func supports(intentTag: PersonalIntentTag) -> Bool {
        intentTags.contains(intentTag)
    }
}

public enum ToolSensitivity: String, Sendable, Codable, Equatable, CaseIterable {
    case low
    case normal
    case sensitive
    case restricted
}

public protocol EdgeTool: Sendable {
    associatedtype Input: Decodable & Sendable
    associatedtype Output: Encodable & Sendable

    var metadata: ToolMetadata { get }

    func execute(_ input: Input) async throws -> Output
}

public struct RegisteredTool: Sendable {
    public let metadata: ToolMetadata

    private let executor: @Sendable ([String: AuditValue]) async throws -> String

    public init<T: EdgeTool>(_ tool: T) {
        self.metadata = tool.metadata
        self.executor = { arguments in
            let input = try ToolJSON.decode(T.Input.self, from: arguments)
            let output = try await tool.execute(input)
            return try ToolJSON.encodeOutput(output)
        }
    }

    public init(
        metadata: ToolMetadata,
        executor: @Sendable @escaping ([String: AuditValue]) async throws -> String
    ) {
        self.metadata = metadata
        self.executor = executor
    }

    public func execute(arguments: [String: AuditValue]) async throws -> String {
        try await executor(arguments)
    }
}

enum ToolJSON {
    static func decode<T: Decodable>(
        _ type: T.Type,
        from arguments: [String: AuditValue]
    ) throws -> T {
        let object = arguments.mapValues { jsonObject(from: $0) }
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(type, from: data)
    }

    static func encodeOutput<T: Encodable>(_ output: T) throws -> String {
        if let string = output as? String {
            return string
        }
        let data = try JSONEncoder().encode(output)
        guard let text = String(data: data, encoding: .utf8) else {
            throw ToolRegistryError.outputEncodingFailed
        }
        return text
    }

    private static func jsonObject(from value: AuditValue) -> Any {
        switch value {
        case .string(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return NSNull()
        case .array(let values):
            return values.map { jsonObject(from: $0) }
        case .object(let values):
            return values.mapValues { jsonObject(from: $0) }
        }
    }
}
