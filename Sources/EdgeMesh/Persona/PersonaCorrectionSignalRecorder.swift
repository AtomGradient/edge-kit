// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct PersonaCorrectionRecordResult: Equatable, Sendable {
    public var signal: PersonaCorrectionSignalPayload
    public var eventID: UUID?

    public var correctionFingerprint: String {
        signal.correctionFingerprint
    }

    public init(signal: PersonaCorrectionSignalPayload, eventID: UUID?) {
        self.signal = signal
        self.eventID = eventID
    }
}

public enum PersonaCorrectionSignalRecorderError: Error, Equatable, LocalizedError {
    case emptyFactCorrection
    case emptyProfileOverlay

    public var errorDescription: String? {
        switch self {
        case .emptyFactCorrection:
            return "persona correction fact fields are empty"
        case .emptyProfileOverlay:
            return "persona correction profile overlay is empty"
        }
    }
}

public struct PersonaCorrectionFieldNormalizationPolicy: Equatable, Sendable {
    public var allowedFields: Set<String>?
    public var maxStringLength: Int?
    public var includeEmptyStrings: Bool

    public init(
        allowedFields: Set<String>? = nil,
        maxStringLength: Int? = nil,
        includeEmptyStrings: Bool = false
    ) {
        self.allowedFields = allowedFields
        self.maxStringLength = maxStringLength
        self.includeEmptyStrings = includeEmptyStrings
    }
}

public enum PersonaCorrectionFieldNormalizer {
    public static func normalizeStringFields(
        _ fields: [String: String],
        policy: PersonaCorrectionFieldNormalizationPolicy = PersonaCorrectionFieldNormalizationPolicy()
    ) -> [String: PersonaCorrectionJSONValue] {
        var out: [String: PersonaCorrectionJSONValue] = [:]
        for key in fields.keys.sorted() {
            if let allowedFields = policy.allowedFields,
               !allowedFields.contains(key) {
                continue
            }

            var value = fields[key, default: ""]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !policy.includeEmptyStrings && value.isEmpty {
                continue
            }
            if let maxStringLength = policy.maxStringLength,
               maxStringLength >= 0,
               value.count > maxStringLength {
                value = String(value.prefix(maxStringLength))
            }
            if !policy.includeEmptyStrings && value.isEmpty {
                continue
            }
            out[key] = .string(value)
        }
        return out
    }
}

@available(iOS 17.0, macOS 14.0, *)
public enum PersonaCorrectionSignalRecorder {
    @discardableResult
    public static func recordFactCorrection(
        appID: String,
        source: String,
        normalizedFields: [String: PersonaCorrectionJSONValue],
        collector: DataCollector?,
        createdAt: Double = Date().timeIntervalSince1970
    ) throws -> PersonaCorrectionRecordResult {
        guard !normalizedFields.isEmpty else {
            throw PersonaCorrectionSignalRecorderError.emptyFactCorrection
        }
        let signal = PersonaCorrectionSignalPayload(
            appID: appID,
            source: source,
            factCorrection: PersonaFactCorrectionSignal(normalizedFields: normalizedFields),
            createdAt: createdAt
        )
        let event = try collector?.collectPersonaCorrectionSignal(signal)
        return PersonaCorrectionRecordResult(signal: signal, eventID: event?.id)
    }

    @discardableResult
    public static func recordProfileOverlay(
        appID: String,
        source: String,
        target: [String: PersonaCorrectionJSONValue],
        overlay: [String: PersonaCorrectionJSONValue],
        collector: DataCollector?,
        createdAt: Double = Date().timeIntervalSince1970
    ) throws -> PersonaCorrectionRecordResult {
        guard !target.isEmpty || !overlay.isEmpty else {
            throw PersonaCorrectionSignalRecorderError.emptyProfileOverlay
        }
        let signal = PersonaCorrectionSignalPayload(
            appID: appID,
            source: source,
            profileOverlay: PersonaProfileCorrectionSignal(
                target: target,
                overlay: overlay
            ),
            createdAt: createdAt
        )
        let event = try collector?.collectPersonaCorrectionSignal(signal)
        return PersonaCorrectionRecordResult(signal: signal, eventID: event?.id)
    }
}
