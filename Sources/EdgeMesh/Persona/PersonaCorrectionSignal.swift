// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

/// JSON value for correction signal payloads carried by edge-kit.
///
/// edge-kit owns transport/storage shape only. edge-halo owns the policy that
/// decides when these signals are stable enough to affect Neural Imprint.
public enum PersonaCorrectionJSONValue: Codable, Equatable, Sendable {
    case string(String)
    case integer(Int64)
    case double(Double)
    case bool(Bool)
    case null
    case array([PersonaCorrectionJSONValue])
    case object([String: PersonaCorrectionJSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([PersonaCorrectionJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: PersonaCorrectionJSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value):
            try container.encode(value)
        case let .integer(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        case let .bool(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        case let .array(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        }
    }

    public var canonicalJSON: String {
        switch self {
        case let .string(value):
            return Self.quote(value)
        case let .integer(value):
            return String(value)
        case let .double(value):
            return value.isFinite ? String(value) : Self.quote(String(value))
        case let .bool(value):
            return value ? "true" : "false"
        case .null:
            return "null"
        case let .array(value):
            return "[" + value.map(\.canonicalJSON).joined(separator: ",") + "]"
        case let .object(value):
            let body = value.keys.sorted().map { key in
                "\(Self.quote(key)):\(value[key]?.canonicalJSON ?? "null")"
            }.joined(separator: ",")
            return "{\(body)}"
        }
    }

    private static func quote(_ value: String) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: value,
                options: [.fragmentsAllowed]
            ),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "\"\(value)\""
        }
        return encoded
    }
}

public struct PersonaFactCorrectionSignal: Codable, Equatable, Sendable {
    public var normalizedFields: [String: PersonaCorrectionJSONValue]

    private enum CodingKeys: String, CodingKey {
        case normalizedFields = "normalized_fields"
    }

    public init(normalizedFields: [String: PersonaCorrectionJSONValue]) {
        self.normalizedFields = normalizedFields
    }
}

public struct PersonaProfileCorrectionSignal: Codable, Equatable, Sendable {
    public var target: [String: PersonaCorrectionJSONValue]
    public var overlay: [String: PersonaCorrectionJSONValue]

    public init(
        target: [String: PersonaCorrectionJSONValue],
        overlay: [String: PersonaCorrectionJSONValue]
    ) {
        self.target = target
        self.overlay = overlay
    }
}

public struct PersonaCorrectionSignalPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.persona_correction_signal.v1"
    public static let eventType = "persona_correction_signal"

    public var schemaVersion: String
    public var appID: String
    public var correctionFingerprint: String
    public var source: String
    public var createdAt: Double
    public var factCorrection: PersonaFactCorrectionSignal?
    public var profileOverlay: PersonaProfileCorrectionSignal?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case appID = "app_id"
        case correctionFingerprint = "correction_fingerprint"
        case source
        case createdAt = "created_at"
        case factCorrection = "fact_correction"
        case profileOverlay = "profile_overlay"
    }

    public init(
        appID: String,
        source: String,
        factCorrection: PersonaFactCorrectionSignal? = nil,
        profileOverlay: PersonaProfileCorrectionSignal? = nil,
        correctionFingerprint: String? = nil,
        createdAt: Double = Date().timeIntervalSince1970
    ) {
        self.schemaVersion = Self.schemaVersion
        self.appID = appID
        self.source = source
        self.createdAt = createdAt
        self.factCorrection = factCorrection
        self.profileOverlay = profileOverlay
        self.correctionFingerprint = correctionFingerprint ?? Self.makeFingerprint(
            appID: appID,
            source: source,
            createdAt: createdAt,
            factCorrection: factCorrection,
            profileOverlay: profileOverlay
        )
    }

    public static func makeFingerprint(
        appID: String,
        source: String,
        createdAt: Double,
        factCorrection: PersonaFactCorrectionSignal?,
        profileOverlay: PersonaProfileCorrectionSignal?
    ) -> String {
        let fields = [
            "app_id": quote(appID),
            "created_at": String(createdAt),
            "fact_correction": canonicalJSON(factCorrection?.normalizedFields ?? [:]),
            "profile_overlay": canonicalJSON([
                "overlay": .object(profileOverlay?.overlay ?? [:]),
                "target": .object(profileOverlay?.target ?? [:]),
            ]),
            "schema_version": quote(schemaVersion),
            "source": quote(source),
        ]
        let body = fields.keys.sorted().map { key in
            "\(quote(key)):\(fields[key] ?? "null")"
        }.joined(separator: ",")
        return sha256Text("{\(body)}")
    }

    public func encodedPayload() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }

    private static func canonicalJSON(_ value: [String: PersonaCorrectionJSONValue]) -> String {
        let body = value.keys.sorted().map { key in
            "\(quote(key)):\(value[key]?.canonicalJSON ?? "null")"
        }.joined(separator: ",")
        return "{\(body)}"
    }

    private static func sha256Text(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map {
            String(format: "%02x", $0)
        }.joined()
    }

    private static func quote(_ value: String) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: value,
                options: [.fragmentsAllowed]
            ),
            let encoded = String(data: data, encoding: .utf8)
        else {
            return "\"\(value)\""
        }
        return encoded
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension DataCollector {
    @discardableResult
    public func collectPersonaCorrectionSignal(
        _ signal: PersonaCorrectionSignalPayload
    ) throws -> DataEvent {
        try appendEvent(
            appId: signal.appID,
            eventType: PersonaCorrectionSignalPayload.eventType,
            payload: try signal.encodedPayload(),
            tags: [.trainingSample, .userCorrection, .preference]
        )
    }
}
