// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Entry point for loading local on-device AI engines.
///
/// `loadLocal(directory:)` detects the model category and returns a typed
/// engine wrapper for LLM, VLM, TTS, or STT models.
///
/// ```swift
/// let rt = EdgeRuntime()
///
/// // LLM text chat
/// let llm = try await rt.loadLocal(directory: qwen3URL) as! LLMEngine
/// for try await chunk in llm.generate(messages: [.user("Hello")]) {
///     print(chunk.text, terminator: "")
/// }
///
/// // VLM image understanding
/// let vlm = try await rt.loadLocal(directory: gemma3URL) as! VLMEngine
/// for try await chunk in vlm.generate(messages: [.user("Describe this image")], images: [imgURL]) {
///     print(chunk.text, terminator: "")
/// }
///
/// // TTS speech synthesis
/// let tts = try await rt.loadLocal(directory: ttsURL) as! TTSEngine
/// let audio = try await tts.speak("Hello world", voice: "serena")
///
/// // STT speech recognition
/// let stt = try await rt.loadLocal(directory: asrURL) as! STTEngine
/// let result = try await stt.transcribe(audioURL: audioFileURL)
/// print(result.text)
/// ```
@MainActor
public final class EdgeRuntime {

    public init() {}

    /// Detects a local model directory and loads the matching engine type.
    ///
    /// The returned wrapper contains exactly one of `LLMEngine`, `VLMEngine`,
    /// `TTSEngine`, or `STTEngine`.
    public func loadLocal(directory: URL) async throws -> AnyEngine {
        let category = ModelCategory.detect(from: directory)

        switch category {
        case .llm:
            let engine = LLMEngine()
            try await engine.loadLocal(directory: directory)
            return AnyEngine(llm: engine)

        case .vlm:
            let engine = VLMEngine()
            try await engine.loadLocal(directory: directory)
            return AnyEngine(vlm: engine)

        case .tts:
            let engine = TTSEngine()
            try await engine.loadLocal(directory: directory)
            return AnyEngine(tts: engine)

        case .stt:
            let engine = STTEngine()
            try await engine.loadLocal(directory: directory)
            return AnyEngine(stt: engine)
        }
    }

    /// Loads the default LLM for the current device tier.
    public func loadRecommendedModel() async throws -> LLMEngine {
        let profile = DeviceProfile.current
        let tier    = ModelTierSelector.recommend(for: profile)
        let config  = ModelConfig.config(for: tier)
        let engine  = LLMEngine()
        try await engine.load(config: config)
        return engine
    }

    /// Loads a registered LLM model by local model identifier.
    public func load(_ modelID: String) async throws -> LLMEngine {
        guard let config = ModelConfig.find(modelID: modelID) else {
            throw EdgeRuntimeError.modelNotFound(modelID)
        }
        let engine = LLMEngine()
        try await engine.load(config: config)
        return engine
    }
}

/// Type-erased wrapper for one loaded engine.
public struct AnyEngine {
    /// Loaded model category.
    public let category: ModelCategory

    private let _llm: LLMEngine?
    private let _vlm: VLMEngine?
    private let _tts: TTSEngine?
    private let _stt: STTEngine?

    init(llm: LLMEngine) { self.category = .llm; self._llm = llm; self._vlm = nil; self._tts = nil; self._stt = nil }
    init(vlm: VLMEngine) { self.category = .vlm; self._llm = nil; self._vlm = vlm; self._tts = nil; self._stt = nil }
    init(tts: TTSEngine) { self.category = .tts; self._llm = nil; self._vlm = nil; self._tts = tts; self._stt = nil }
    init(stt: STTEngine) { self.category = .stt; self._llm = nil; self._vlm = nil; self._tts = nil; self._stt = stt }

    /// Loaded LLM engine, when `category == .llm`.
    public var llm: LLMEngine? { _llm }
    /// Loaded VLM engine, when `category == .vlm`.
    public var vlm: VLMEngine? { _vlm }
    /// Loaded TTS engine, when `category == .tts`.
    public var tts: TTSEngine? { _tts }
    /// Loaded STT engine, when `category == .stt`.
    public var stt: STTEngine? { _stt }
}

public enum EdgeRuntimeError: Error, LocalizedError {
    case modelNotFound(String)
    case insufficientMemory(required: Int, available: Int)
    case loadFailed(String)
    case unsupportedFeature(String)
    case thermalPause

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Model not found: \(id)"
        case .insufficientMemory(let req, let avail):
            return "Insufficient memory: required \(req)GB, available \(avail)GB"
        case .loadFailed(let reason):
            return "Load failed: \(reason)"
        case .unsupportedFeature(let reason):
            return "Unsupported feature: \(reason)"
        case .thermalPause:
            return "Generation paused: device thermal state is critical"
        }
    }
}
