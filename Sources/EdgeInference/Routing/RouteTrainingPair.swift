// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Source of a route supervision example.
public enum RouteTrainingPairSource: String, Sendable, Codable, Equatable {
    case eval
    case userFeedback = "user_feedback"
    case synthetic
    case runtimeAudit = "runtime_audit"
    case imported
}

/// A supervised routing example used by learned/base classifiers.
///
/// This is a data contract only. It intentionally stores the expected intent
/// and tool boundary without encoding app-specific routing rules.
public struct RouteTrainingPair: Sendable, Codable, Equatable {
    public var input: UserInputContext
    public var expectedIntentTag: PersonalIntentTag
    public var selectedToolNames: [String]
    public var toolCallPlan: [ToolCallPlan]?
    public var confidence: Double
    public var source: RouteTrainingPairSource
    public var rationale: String?
    public var createdAtMs: Int64?
    public var metadata: [String: AuditValue]

    public init(
        input: UserInputContext,
        expectedIntentTag: PersonalIntentTag,
        selectedToolNames: [String] = [],
        toolCallPlan: [ToolCallPlan]? = nil,
        confidence: Double = 1.0,
        source: RouteTrainingPairSource,
        rationale: String? = nil,
        createdAtMs: Int64? = nil,
        metadata: [String: AuditValue] = [:]
    ) {
        self.input = input
        self.expectedIntentTag = expectedIntentTag
        self.selectedToolNames = selectedToolNames
        self.toolCallPlan = toolCallPlan
        self.confidence = confidence
        self.source = source
        self.rationale = rationale
        self.createdAtMs = createdAtMs
        self.metadata = metadata
    }

    public var routerLabel: RouterLabel {
        switch expectedIntentTag {
        case .exactFact, .aggregateFact:
            return .fact
        case .userProfile:
            return .persona
        case .baseChat:
            return .baseChat
        case .appAction, .mixed:
            return .mixed
        }
    }
}

/// Exact-match classifier backed by `RouteTrainingPair`.
///
/// This adapter is intended for deterministic eval replay and as a boundary
/// for learned router data. It does not perform keyword or semantic matching;
/// misses are reported explicitly so `QueryRouter` can keep its normal
/// fallback behavior. Matching canonicalizes Unicode width so full-width
/// punctuation produced by mobile keyboards (for example `？`) matches the
/// same supervised text written with ASCII punctuation (`?`). It also ignores
/// terminal sentence punctuation, so mobile text fields that omit a trailing
/// `?` still replay the same supervised route. This remains exact matching:
/// no keyword, substring, or semantic fallback is performed here.
public final class RouteTrainingPairClassifier: BaseClassifierProtocol, @unchecked Sendable {
    private let index: [String: RouteTrainingPair]
    private static let terminalSentencePunctuation = CharacterSet(charactersIn: "?!。！？.")

    public init(pairs: [RouteTrainingPair]) {
        var index: [String: RouteTrainingPair] = [:]
        for pair in pairs {
            let key = Self.normalizedText(pair.input.text)
            if index[key] == nil {
                index[key] = pair
            }
        }
        self.index = index
    }

    public var isAvailable: Bool {
        !index.isEmpty
    }

    public func classifyFewShot(_ query: String) throws -> (RouterLabel, Double) {
        let key = Self.normalizedText(query)
        guard let pair = index[key] else {
            throw BaseClassifierError.noMatch
        }
        return (pair.routerLabel, pair.confidence)
    }

    private static func normalizedText(_ text: String) -> String {
        let widthNormalized = text.applyingTransform(.fullwidthToHalfwidth, reverse: false) ?? text
        let whitespaceNormalized = widthNormalized
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        return whitespaceNormalized.trimmingCharacters(in: terminalSentencePunctuation)
    }
}
