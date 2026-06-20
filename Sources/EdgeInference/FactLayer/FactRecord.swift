// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Allowed `sourceType` values for local device facts.
public let ALLOWED_FACT_SOURCE_TYPES: Set<String> = ["user_device"]

/// In-memory representation of a single fact record.
///
/// Field names and value types must stay aligned with the Python fact contract.
public struct FactRecord: Sendable, Equatable {
    /// UUID string.
    public let id: String
    /// Dotted schema name, such as `"finance.expense"`.
    public let schemaName: String
    /// Raw payload as a sendable Swift value map.
    public let payload: [String: FactValue]
    /// Creation time in Unix milliseconds.
    public let createdAt: Int64
    public let sensitivity: String
    public let ttlSeconds: Int64?
    public let derivedFrom: String?
    public let sourceType: String

    public init(
        id: String,
        schemaName: String,
        payload: [String: FactValue],
        createdAt: Int64,
        sensitivity: String = "private",
        ttlSeconds: Int64? = nil,
        derivedFrom: String? = nil,
        sourceType: String = "user_device"
    ) {
        self.id = id
        self.schemaName = schemaName
        self.payload = payload
        self.createdAt = createdAt
        self.sensitivity = sensitivity
        self.ttlSeconds = ttlSeconds
        self.derivedFrom = derivedFrom
        self.sourceType = sourceType
    }

    /// Creates a new fact, defaulting `id` to a UUID and `createdAt` to the current time.
    public static func new(
        schemaName: String,
        payload: [String: FactValue],
        createdAt: Int64? = nil,
        sensitivity: String = "private",
        ttlSeconds: Int64? = nil,
        derivedFrom: String? = nil,
        sourceType: String = "user_device",
        id: String? = nil
    ) -> FactRecord {
        FactRecord(
            id: id ?? UUID().uuidString.lowercased(),
            schemaName: schemaName,
            payload: payload,
            createdAt: createdAt ?? Int64(Date().timeIntervalSince1970 * 1000),
            sensitivity: sensitivity,
            ttlSeconds: ttlSeconds,
            derivedFrom: derivedFrom,
            sourceType: sourceType
        )
    }
}

/// JSON-serializable value subset for fact payloads.
///
/// This enum avoids `Any` so fact payloads remain sendable and encodable.
public enum FactValue: Sendable, Equatable {
    case string(String)
    case int(Int64)
    case double(Double)
    case bool(Bool)
    case null

    /// Creates a fact value from a supported Swift value.
    public static func from(_ value: Any?) -> FactValue? {
        guard let value = value else { return .null }
        if let s = value as? String { return .string(s) }
        if let b = value as? Bool { return .bool(b) }
        if let i = value as? Int { return .int(Int64(i)) }
        if let i = value as? Int64 { return .int(i) }
        if let i = value as? Int32 { return .int(Int64(i)) }
        if let d = value as? Double { return .double(d) }
        if let f = value as? Float { return .double(Double(f)) }
        return nil
    }

    /// Value representation used by `FactFieldDef.validate`.
    public var asAnyForValidation: Any? {
        switch self {
        case .string(let s): return s
        case .int(let i): return i
        case .double(let d): return d
        case .bool(let b): return b
        case .null: return nil
        }
    }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }
    public var intValue: Int64? {
        if case .int(let i) = self { return i }
        return nil
    }
    public var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }
    public var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }
    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }
}

extension FactRecord {
    public func payloadAsJSONString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(PayloadWrapper(payload: payload))
        guard let s = String(data: data, encoding: .utf8) else {
            throw FactRecordError.jsonEncodingFailed
        }
        return s
    }

    /// Decodes a stored payload JSON string into a fact value map.
    public static func parsePayloadJSON(_ json: String) throws -> [String: FactValue] {
        guard let data = json.data(using: .utf8) else {
            throw FactRecordError.jsonDecodingFailed
        }
        let wrapper = try JSONDecoder().decode(PayloadWrapper.self, from: data)
        return wrapper.payload
    }
}

public enum FactRecordError: Error, Equatable {
    case jsonEncodingFailed
    case jsonDecodingFailed
    case invalidSourceType(String)
}

private struct PayloadWrapper: Codable {
    let payload: [String: FactValue]

    init(payload: [String: FactValue]) { self.payload = payload }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode([String: FactValueBox].self)
        self.payload = raw.mapValues { $0.unbox() }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(payload.mapValues { FactValueBox.box($0) })
    }
}

private struct FactValueBox: Codable {
    let value: FactValue

    static func box(_ v: FactValue) -> FactValueBox { FactValueBox(value: v) }
    func unbox() -> FactValue { value }

    init(value: FactValue) { self.value = value }

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self.value = .null; return
        }
        if let b = try? c.decode(Bool.self) {
            self.value = .bool(b); return
        }
        if let i = try? c.decode(Int64.self) {
            self.value = .int(i); return
        }
        if let d = try? c.decode(Double.self) {
            self.value = .double(d); return
        }
        if let s = try? c.decode(String.self) {
            self.value = .string(s); return
        }
        throw DecodingError.dataCorruptedError(
            in: c, debugDescription: "Unsupported FactValue JSON type"
        )
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch value {
        case .null:         try c.encodeNil()
        case .bool(let b):  try c.encode(b)
        case .int(let i):   try c.encode(i)
        case .double(let d):try c.encode(d)
        case .string(let s):try c.encode(s)
        }
    }
}
