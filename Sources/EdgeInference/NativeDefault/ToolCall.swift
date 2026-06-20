// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct ToolCall: Decodable, Sendable {
    public struct Function: Decodable, Sendable {
        public var name: String
        public var arguments: [String: any Sendable]

        public init(name: String, arguments: [String: any Sendable] = [:]) {
            self.name = name
            self.arguments = arguments
        }

        private enum CodingKeys: String, CodingKey {
            case name
            case arguments
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            let decoded = try container.decodeIfPresent(
                [String: ToolCallJSONValue].self,
                forKey: .arguments
            ) ?? [:]
            arguments = decoded.mapValues(\.sendableValue)
        }
    }

    public var function: Function

    public init(function: Function) {
        self.function = function
    }
}

enum ToolCallJSONValue: Decodable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([ToolCallJSONValue])
    case object([String: ToolCallJSONValue])

    init(from decoder: Decoder) throws {
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
        } else if let value = try? container.decode([ToolCallJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: ToolCallJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported tool-call JSON value"
            )
        }
    }

    var sendableValue: any Sendable {
        switch self {
        case .null:
            return Optional<String>.none as String?
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map(\.sendableValue) as [any Sendable]
        case .object(let object):
            return object.mapValues(\.sendableValue) as [String: any Sendable]
        }
    }
}
