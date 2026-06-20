// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum ToolArgumentConversionError: Error, Equatable {
    case objectUnsupported(String)
    case valueUnsupported(String)
}

public extension ToolCallPlan {
    /// Builds an executable tool plan from a model-emitted tool call.
    ///
    /// The bridge accepts JSON-compatible scalar, array, and object values while
    /// preserving path-specific failures for unsupported runtime values.
    init(toolCall: ToolCall) throws {
        var arguments: [String: AuditValue] = [:]
        for (key, value) in toolCall.function.arguments {
            arguments[key] = try AuditValue(sendableValue: value, path: key)
        }
        self.init(
            toolName: toolCall.function.name,
            arguments: arguments
        )
    }
}

public extension ToolRegistry {
    func execute(_ toolCall: ToolCall) async throws -> String {
        try await execute(try ToolCallPlan(toolCall: toolCall))
    }
}

private extension AuditValue {
    init(sendableValue value: any Sendable, path: String) throws {
        switch value {
        case Optional<String>.none:
            self = .null
        case let value as Bool:
            self = .bool(value)
        case let value as Int:
            self = .int(value)
        case let value as Double:
            self = .double(value)
        case let value as Float:
            self = .double(Double(value))
        case let value as String:
            self = .string(value)
        case let values as [any Sendable]:
            self = .array(try values.enumerated().map { index, value in
                try AuditValue(sendableValue: value, path: "\(path)[\(index)]")
            })
        case let object as [String: any Sendable]:
            self = .object(try object.mapValuesWithKeys { key, value in
                try AuditValue(sendableValue: value, path: "\(path).\(key)")
            })
        default:
            throw ToolArgumentConversionError.valueUnsupported(path)
        }
    }
}

private extension Dictionary where Key == String, Value == any Sendable {
    func mapValuesWithKeys<T>(
        _ transform: (String, any Sendable) throws -> T
    ) rethrows -> [String: T] {
        var result: [String: T] = [:]
        result.reserveCapacity(count)
        for (key, value) in self {
            result[key] = try transform(key, value)
        }
        return result
    }
}
