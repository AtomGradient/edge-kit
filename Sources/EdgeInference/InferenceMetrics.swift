// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Performance metrics captured after a generation call completes.
///
/// Access the latest value through `LLMEngine.lastMetrics` or
/// `VLMEngine.lastMetrics`.
///
/// ```swift
/// for try await chunk in engine.generate(messages: messages) { ... }
/// if let m = engine.lastMetrics {
///     print("TTFT: \(m.ttftMs)ms, TPS: \(m.decodeTPS)")
///     if let deferredLoadMs = m.deferredLoadMs { print("Deferred load: \(deferredLoadMs)ms") }
///     print("Memory: \(m.memoryBeforeMB)MB -> \(m.memoryAfterMB)MB (delta \(m.memoryDeltaMB)MB)")
///     print("Policy: \(m.policyReasoning)")
/// }
/// ```
public struct InferencePhaseTimings: Sendable {
    public let inputPreparationMs: Double?
    public let planningMs: Double?
    public let warmupMs: Double?
    public let sessionSetupMs: Double?
    public let promptCacheLookupMs: Double?
    public let prefillMs: Double?
    public let fullPrefillMs: Double?
    public let cacheHitSuffixPrefillMs: Double?
    public let firstTokenSelectionMs: Double?
    public let firstDecodeTokenMs: Double?
    public let decodeMs: Double?
    public let prefillTokenCount: Int?
    public let prefillMode: String?

    public init(
        inputPreparationMs: Double? = nil,
        planningMs: Double? = nil,
        warmupMs: Double? = nil,
        sessionSetupMs: Double? = nil,
        promptCacheLookupMs: Double? = nil,
        prefillMs: Double? = nil,
        fullPrefillMs: Double? = nil,
        cacheHitSuffixPrefillMs: Double? = nil,
        firstTokenSelectionMs: Double? = nil,
        firstDecodeTokenMs: Double? = nil,
        decodeMs: Double? = nil,
        prefillTokenCount: Int? = nil,
        prefillMode: String? = nil
    ) {
        self.inputPreparationMs = inputPreparationMs
        self.planningMs = planningMs
        self.warmupMs = warmupMs
        self.sessionSetupMs = sessionSetupMs
        self.promptCacheLookupMs = promptCacheLookupMs
        self.prefillMs = prefillMs
        self.fullPrefillMs = fullPrefillMs
        self.cacheHitSuffixPrefillMs = cacheHitSuffixPrefillMs
        self.firstTokenSelectionMs = firstTokenSelectionMs
        self.firstDecodeTokenMs = firstDecodeTokenMs
        self.decodeMs = decodeMs
        self.prefillTokenCount = prefillTokenCount
        self.prefillMode = prefillMode
    }
}

public struct InferenceMetrics: Sendable {

    /// Time to first token (ms)
    public let ttftMs: Double

    /// Decode tokens per second (excluding prefill)
    public let decodeTPS: Double

    /// Cold deferred model load time (ms), reported separately from TTFT.
    ///
    /// Native phased VLM text turns may load decoder weights lazily on the
    /// first generate call. `ttftMs` excludes that cold load; this field
    /// carries it so dashboards don't misread deferred mmap/load as inference
    /// latency. Nil means no deferred load happened in this turn.
    public let deferredLoadMs: Double?

    /// Best-effort phase timings for a generate turn. Nil fields mean the
    /// active backend cannot expose that phase cleanly.
    public let phaseTimings: InferencePhaseTimings?

    /// Prompt token count (prefill phase)
    public let promptTokenCount: Int

    /// Generated token count (decode phase)
    public let generationTokenCount: Int

    /// Process physical footprint before inference (MB)
    public let memoryBeforeMB: Int

    /// Process physical footprint after inference (MB)
    public let memoryAfterMB: Int

    /// Memory delta (MB) — positive means consumption increased
    public var memoryDeltaMB: Int { memoryAfterMB - memoryBeforeMB }

    /// InferencePolicy decision summary
    public let policyReasoning: String

    /// Whether prompt cache was hit (incremental prefill)
    public let promptCacheHit: Bool

    /// Number of tokens reused from prompt cache
    public let cachedTokensReused: Int

    /// Thermal state at inference time
    public let thermalState: String

    /// Current conversation turn
    public let turn: Int

    public init(
        ttftMs: Double,
        decodeTPS: Double,
        deferredLoadMs: Double?,
        phaseTimings: InferencePhaseTimings? = nil,
        promptTokenCount: Int,
        generationTokenCount: Int,
        memoryBeforeMB: Int,
        memoryAfterMB: Int,
        policyReasoning: String,
        promptCacheHit: Bool,
        cachedTokensReused: Int,
        thermalState: String,
        turn: Int
    ) {
        self.ttftMs = ttftMs
        self.decodeTPS = decodeTPS
        self.deferredLoadMs = deferredLoadMs
        self.phaseTimings = phaseTimings
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.memoryBeforeMB = memoryBeforeMB
        self.memoryAfterMB = memoryAfterMB
        self.policyReasoning = policyReasoning
        self.promptCacheHit = promptCacheHit
        self.cachedTokensReused = cachedTokensReused
        self.thermalState = thermalState
        self.turn = turn
    }
}
