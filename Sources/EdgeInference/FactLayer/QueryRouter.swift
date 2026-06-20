// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Router labels used by learned routing and eval replay.
///
/// This file intentionally contains no keyword routing or natural-language
/// rule parser. If no classifier can make an explicit decision, `QueryRouter`
/// fails closed to `.baseChat`.
public enum RouterLabel: String, Sendable, Equatable {
    case persona
    case fact
    case mixed
    case baseChat = "base_chat"
}

/// Default confidence threshold for learned/base classifiers.
public let DEFAULT_CONFIDENCE_THRESHOLD: Double = 0.7

/// Classification result shared by routing and audit code.
public struct RouterResult: Sendable, Equatable {
    public let label: RouterLabel
    public let confidence: Double
    public let reason: String
    public let latencyMs: Double
    public let query: String
    public let rawModelLabel: RouterLabel?

    public init(
        label: RouterLabel,
        confidence: Double,
        reason: String,
        latencyMs: Double,
        query: String = "",
        rawModelLabel: RouterLabel? = nil
    ) {
        self.label = label
        self.confidence = confidence
        self.reason = reason
        self.latencyMs = latencyMs
        self.query = query
        self.rawModelLabel = rawModelLabel
    }
}

/// Base/learned classifier contract used by `QueryRouter`.
public enum BaseClassifierError: Error, Sendable, Equatable {
    /// Classifier is available, but has no supervised/learned match for this
    /// query. `QueryRouter` must fail closed instead of guessing by rules.
    case noMatch
}

public protocol BaseClassifierProtocol: AnyObject, Sendable {
    /// false means resources are unavailable or the classifier has no usable
    /// route surface. `QueryRouter` must fail closed to base chat.
    var isAvailable: Bool { get }

    /// Returns `(label, confidence)` for an explicit learned/base decision.
    func classifyFewShot(_ query: String) throws -> (RouterLabel, Double)
}

/// Test/eval classifier that returns a fixed label or delegates to a hook.
public final class MockBaseClassifier: BaseClassifierProtocol, @unchecked Sendable {
    private let _available: Bool
    private let _hook: (@Sendable (String) throws -> (RouterLabel, Double))?
    private let _default: (RouterLabel, Double)

    public init(
        available: Bool = true,
        labelHook: (@Sendable (String) throws -> (RouterLabel, Double))? = nil,
        defaultLabel: RouterLabel = .mixed,
        defaultConfidence: Double = 0.9
    ) {
        self._available = available
        self._hook = labelHook
        self._default = (defaultLabel, defaultConfidence)
    }

    public var isAvailable: Bool { _available }

    public func classifyFewShot(_ query: String) throws -> (RouterLabel, Double) {
        if let hook = _hook {
            return try hook(query)
        }
        return _default
    }
}

/// Learned/base router facade.
///
/// The former keyword fallback has been removed because it was an anti-NI
/// path. Misses, failures, and unavailable classifiers now return `.baseChat`
/// with explicit audit reasons.
public final class QueryRouter: @unchecked Sendable {

    public enum FailureMode: String, Sendable {
        case oom = "oom"
        case timeout = "timeout"
        case modelError = "model_error"
    }

    private let base: BaseClassifierProtocol?
    private let threshold: Double

    public init(
        base: BaseClassifierProtocol? = nil,
        confidenceThreshold: Double = DEFAULT_CONFIDENCE_THRESHOLD
    ) {
        self.base = base
        self.threshold = confidenceThreshold
    }

    public convenience init(
        routeTrainingPairs: [RouteTrainingPair],
        confidenceThreshold: Double = DEFAULT_CONFIDENCE_THRESHOLD
    ) {
        self.init(
            base: RouteTrainingPairClassifier(pairs: routeTrainingPairs),
            confidenceThreshold: confidenceThreshold
        )
    }

    public func classify(
        _ query: String,
        forceConfidence: Double? = nil,
        failureMode: FailureMode? = nil
    ) -> RouterResult {
        let t0 = DispatchTime.now().uptimeNanoseconds

        func elapsedMs() -> Double {
            let dtNs = DispatchTime.now().uptimeNanoseconds &- t0
            return Double(dtNs) / 1_000_000.0
        }

        func baseChat(reason: String) -> RouterResult {
            RouterResult(
                label: .baseChat,
                confidence: 0.0,
                reason: reason,
                latencyMs: elapsedMs(),
                query: query
            )
        }

        if let mode = failureMode {
            return baseChat(reason: "legacy_rule_based_fallback_disabled_\(mode.rawValue)")
        }

        guard let base, base.isAvailable else {
            return baseChat(reason: "legacy_rule_based_fallback_disabled")
        }

        let rawLabel: RouterLabel
        let rawConfidence: Double
        do {
            let result = try base.classifyFewShot(query)
            rawLabel = result.0
            rawConfidence = result.1
        } catch BaseClassifierError.noMatch {
            return baseChat(reason: "legacy_rule_based_fallback_disabled_no_match")
        } catch {
            return baseChat(reason: "legacy_rule_based_fallback_disabled_model_error")
        }

        let confidence = forceConfidence ?? rawConfidence
        if confidence < threshold {
            return RouterResult(
                label: .mixed,
                confidence: confidence,
                reason: "low_confidence",
                latencyMs: elapsedMs(),
                query: query,
                rawModelLabel: rawLabel
            )
        }

        return RouterResult(
            label: rawLabel,
            confidence: confidence,
            reason: "few_shot_high_conf",
            latencyMs: elapsedMs(),
            query: query,
            rawModelLabel: rawLabel
        )
    }
}
