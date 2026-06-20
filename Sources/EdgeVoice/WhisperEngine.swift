// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Provides a speech-to-text facade for Whisper-compatible native runtimes.
@MainActor
public final class WhisperEngine: ObservableObject {

    public enum ModelSize: String, CaseIterable, Sendable {
        case tiny    = "ggml-tiny"
        case base    = "ggml-base"
        case small   = "ggml-small"
        case medium  = "ggml-medium"

        public var filename: String { "\(rawValue).bin" }
        public var sizeBytes: Int {
            switch self {
            case .tiny:   return 75_000_000
            case .base:   return 142_000_000
            case .small:  return 466_000_000
            case .medium: return 1_500_000_000
            }
        }
    }

    @Published public private(set) var isLoaded = false
    @Published public private(set) var isTranscribing = false

    private var modelSize: ModelSize = .base

    public init() {}

    public func load(_ size: ModelSize = .base) async throws {
        modelSize = size
        isLoaded = true
    }

    /// Transcribes an audio file.
    public func transcribe(audioURL: URL, language: String = "auto") async throws -> TranscriptionResult {
        guard isLoaded else { throw WhisperError.notLoaded }
        isTranscribing = true
        defer { isTranscribing = false }

        let text = "[Transcription placeholder — whisper.xcframework integration pending]"
        return TranscriptionResult(text: text, language: language, duration: 0)
    }

    /// Starts realtime microphone transcription.
    public func startRealtime(language: String = "auto") -> AsyncStream<String> {
        AsyncStream { continuation in
            continuation.finish()
        }
    }

    public func unload() {
        isLoaded = false
    }
}

public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String
    public let duration: TimeInterval
}

public enum WhisperError: Error, LocalizedError {
    case notLoaded
    case transcriptionFailed(String)
    case unsupportedFormat

    public var errorDescription: String? {
        switch self {
        case .notLoaded:              return "Whisper model not loaded"
        case .transcriptionFailed(let r): return "Transcription failed: \(r)"
        case .unsupportedFormat:      return "Unsupported audio format"
        }
    }
}
