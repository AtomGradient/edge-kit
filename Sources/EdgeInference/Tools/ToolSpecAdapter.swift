// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Tokenizers

public enum ToolSpecConversionError: Error, Equatable {
    case invalidJSONSchema(String)
    case jsonSchemaRootMustBeObject
    case nullUnsupportedInToolSpec
}

public extension ToolMetadata {
    func toolSpec() throws -> ToolSpec {
        let parameters: [String: any Sendable]
        switch argumentsSchema {
        case .jsonSchema(let schema):
            parameters = try ToolSpecJSON.object(fromJSONString: schema)
        case .auditMap(let map):
            parameters = try ToolSpecJSON.object(fromAuditMap: map)
        }

        return [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": parameters,
            ] as [String: any Sendable],
        ] as ToolSpec
    }
}

public extension ToolRegistry {
    func allToolSpecs() throws -> [ToolSpec] {
        try allSchemas().map { try $0.toolSpec() }
    }

    func toolSpecs(forNames names: [String]) throws -> [ToolSpec] {
        try schemas(forNames: names).map { try $0.toolSpec() }
    }

    func toolSpecs(forIntentTag tag: PersonalIntentTag) throws -> [ToolSpec] {
        try schemas(forIntentTag: tag).map { try $0.toolSpec() }
    }

    func toolSpecs(forIntent intent: PersonalIntent) throws -> [ToolSpec] {
        try schemas(forIntent: intent).map { try $0.toolSpec() }
    }
}

private enum ToolSpecJSON: Sendable, Codable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([ToolSpecJSON])
    case object([String: ToolSpecJSON])

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([ToolSpecJSON].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: ToolSpecJSON].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode JSON schema value"
            )
        }
    }

    func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }

    static func object(fromJSONString string: String) throws -> [String: any Sendable] {
        guard let data = string.data(using: .utf8) else {
            throw ToolSpecConversionError.invalidJSONSchema(string)
        }
        do {
            let value = try JSONDecoder().decode(ToolSpecJSON.self, from: data)
            guard case .object(let object) = value else {
                throw ToolSpecConversionError.jsonSchemaRootMustBeObject
            }
            return try object.mapValues { try $0.sendableValue() }
        } catch let error as ToolSpecConversionError {
            throw error
        } catch {
            throw ToolSpecConversionError.invalidJSONSchema(string)
        }
    }

    static func object(fromAuditMap map: [String: AuditValue]) throws -> [String: any Sendable] {
        try map.mapValues { try sendableValue(from: $0) }
    }

    private func sendableValue() throws -> any Sendable {
        switch self {
        case .null:
            throw ToolSpecConversionError.nullUnsupportedInToolSpec
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            if values.allSatisfy({ if case .string = $0 { true } else { false } }) {
                return values.compactMap {
                    if case .string(let value) = $0 { return value }
                    return nil
                }
            }
            if values.allSatisfy({ if case .int = $0 { true } else { false } }) {
                return values.compactMap {
                    if case .int(let value) = $0 { return value }
                    return nil
                }
            }
            if values.allSatisfy({ if case .double = $0 { true } else { false } }) {
                return values.compactMap {
                    if case .double(let value) = $0 { return value }
                    return nil
                }
            }
            if values.allSatisfy({ if case .bool = $0 { true } else { false } }) {
                return values.compactMap {
                    if case .bool(let value) = $0 { return value }
                    return nil
                }
            }
            return try values.map { try $0.sendableValue() } as [any Sendable]
        case .object(let object):
            return try object.mapValues { try $0.sendableValue() }
        }
    }

    private static func sendableValue(from value: AuditValue) throws -> any Sendable {
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
            throw ToolSpecConversionError.nullUnsupportedInToolSpec
        case .array(let values):
            return try values.map { try sendableValue(from: $0) } as [any Sendable]
        case .object(let values):
            return try values.mapValues { try sendableValue(from: $0) }
        }
    }
}
