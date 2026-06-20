// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

enum ToolArgumentNormalizer {
    static func normalized(
        _ arguments: [String: AuditValue],
        for schema: ToolArgumentSchema
    ) -> [String: AuditValue] {
        guard case .jsonSchema(let schemaString) = schema,
              let properties = jsonSchemaProperties(schemaString) else {
            return arguments
        }

        var normalized = arguments
        for (key, value) in arguments {
            guard let property = properties[key],
                  let type = primaryType(from: property["type"]) else {
                continue
            }
            normalized[key] = normalize(value, as: type)
        }
        for (key, property) in properties where normalized[key] == nil {
            guard let defaultValue = property["default"],
                  let type = primaryType(from: property["type"]),
                  let value = auditValue(from: defaultValue, as: type) else {
                continue
            }
            normalized[key] = normalize(value, as: type)
        }
        return normalized
    }

    private static func jsonSchemaProperties(_ schemaString: String) -> [String: [String: Any]]? {
        guard let data = schemaString.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawProperties = object["properties"] as? [String: Any] else {
            return nil
        }

        var properties: [String: [String: Any]] = [:]
        for (key, value) in rawProperties {
            if let property = value as? [String: Any] {
                properties[key] = property
            }
        }
        return properties
    }

    private static func primaryType(from value: Any?) -> String? {
        if let type = value as? String {
            return type
        }
        if let types = value as? [String] {
            return types.first { $0 != "null" }
        }
        return nil
    }

    private static func normalize(_ value: AuditValue, as type: String) -> AuditValue {
        switch type {
        case "boolean":
            return normalizeBoolean(value)
        case "integer":
            return normalizeInteger(value)
        case "number":
            return normalizeNumber(value)
        default:
            return value
        }
    }

    private static func normalizeBoolean(_ value: AuditValue) -> AuditValue {
        switch value {
        case .string(let string):
            let text = string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            switch text {
            case "true", "1", "yes":
                return .bool(true)
            case "false", "0", "no":
                return .bool(false)
            default:
                return value
            }
        case .int(let int) where int == 0 || int == 1:
            return .bool(int == 1)
        default:
            return value
        }
    }

    private static func normalizeInteger(_ value: AuditValue) -> AuditValue {
        switch value {
        case .string(let string):
            let text = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if let int = Int(text) {
                return .int(int)
            }
            if let double = Double(text), double.rounded(.towardZero) == double {
                return .int(Int(double))
            }
            return value
        case .double(let double) where double.rounded(.towardZero) == double:
            return .int(Int(double))
        default:
            return value
        }
    }

    private static func normalizeNumber(_ value: AuditValue) -> AuditValue {
        switch value {
        case .string(let string):
            let text = string.trimmingCharacters(in: .whitespacesAndNewlines)
            if let double = Double(text) {
                return .double(double)
            }
            return value
        default:
            return value
        }
    }

    private static func auditValue(from value: Any, as type: String) -> AuditValue? {
        switch type {
        case "boolean":
            if let bool = value as? Bool {
                return .bool(bool)
            }
            if let number = value as? NSNumber,
               CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .bool(number.boolValue)
            }
            if let string = value as? String {
                return normalizeBoolean(.string(string))
            }
            return nil
        case "integer":
            if let int = value as? Int {
                return .int(int)
            }
            if let number = value as? NSNumber {
                let double = number.doubleValue
                guard double.rounded(.towardZero) == double else { return nil }
                return .int(Int(double))
            }
            if let string = value as? String {
                return normalizeInteger(.string(string))
            }
            return nil
        case "number":
            if let double = value as? Double {
                return .double(double)
            }
            if let number = value as? NSNumber {
                return .double(number.doubleValue)
            }
            if let string = value as? String {
                return normalizeNumber(.string(string))
            }
            return nil
        case "string":
            if let string = value as? String {
                return .string(string)
            }
            return nil
        default:
            return nil
        }
    }
}
