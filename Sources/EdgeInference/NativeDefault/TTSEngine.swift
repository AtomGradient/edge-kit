// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import EdgeEngine
import Foundation

@MainActor
public final class TTSEngine: ObservableObject {
    private static let maxTextSegmentLength = 220

    @Published public private(set) var state: EngineState = .idle
    @Published public private(set) var downloadProgress: Double = 0

    public private(set) var availableSpeakers: [String] = []
    public private(set) var ttsModelType: String = "unknown"
    public private(set) var sampleRate: Int = 24_000

    private var nativeSession: Qwen3TTSNativeSession?
    private var isLoaded = false

    public init() {}

    public func loadLocal(directory: URL, onProgress: ((Double) -> Void)? = nil) async throws {
        guard state != .loading else { return }
        state = .loading
        downloadProgress = 0
        onProgress?(0)

        do {
            let preflight = try EdgeSpeechBundlePreflightRunner.run(
                configuration: EdgeSpeechBundlePreflightConfiguration(
                    modelRootURL: directory,
                    modelFamily: .qwen3TTS
                )
            )
            let runtimeConfiguration = Self.defaultTTSMetalConfiguration()
            let session = try await Qwen3TTSNativeSession(
                modelURL: directory,
                runtimeConfiguration: runtimeConfiguration
            )
            nativeSession = session
            sampleRate = session.metadata.outputSampleRate
            ttsModelType = preflight.ttsModelType
                ?? preflight.speechTokenizerModelType
                ?? "qwen3_tts"
            availableSpeakers = session.metadata.availableSpeakers
            isLoaded = true
            downloadProgress = 1
            onProgress?(1)
            state = .ready
        } catch {
            unload()
            throw EdgeRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func speak(_ text: String, voice: String? = nil) async throws -> AudioResult {
        try await generate(text: text, speaker: voice)
    }

    public func generate(
        text: String,
        speaker: String? = nil,
        instruct: String? = nil,
        language: String = "auto",
        temperature: Float = 0.9,
        topK: Int = 50,
        maxTokens: Int = 2_048
    ) async throws -> AudioResult {
        guard isLoaded else { throw EdgeRuntimeError.loadFailed("No TTS model loaded") }
        let chunks = try synthesizeTextSegments(
            text: text,
            speaker: speaker,
            language: language,
            temperature: temperature,
            topK: topK,
            maxTokens: maxTokens
        )
        return try Self.combineAudio(chunks.map(\.audio))
    }

    public func speakStream(
        _ text: String,
        voice: String? = nil,
        instruct: String? = nil,
        streamingInterval: Float = 0.8
    ) -> AsyncThrowingStream<TTSEvent, Error> {
        speakStream(
            text,
            speaker: voice,
            instruct: instruct,
            language: "auto",
            temperature: 0.9,
            topK: 50,
            maxTokens: 2_048
        )
    }

    public func speakStream(
        _ text: String,
        speaker: String? = nil,
        instruct: String? = nil,
        language: String = "auto",
        temperature: Float = 0.9,
        topK: Int = 50,
        maxTokens: Int = 2_048
    ) -> AsyncThrowingStream<TTSEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor [weak self] in
                do {
                    guard let self else {
                        continuation.finish(throwing: EdgeRuntimeError.loadFailed("No TTS engine loaded"))
                        return
                    }
                    guard self.isLoaded else {
                        throw EdgeRuntimeError.loadFailed("No TTS model loaded")
                    }

                    let chunks = try self.synthesizeTextSegments(
                        text: text,
                        speaker: speaker,
                        language: language,
                        temperature: temperature,
                        topK: topK,
                        maxTokens: maxTokens
                    )
                    var audioResults: [AudioResult] = []
                    audioResults.reserveCapacity(chunks.count)
                    for index in chunks.indices {
                        let item = chunks[index]
                        audioResults.append(item.audio)
                        continuation.yield(.audioChunk(item.chunk))
                        continuation.yield(.progress(Int((Double(index + 1) / Double(chunks.count)) * 100)))
                    }
                    continuation.yield(.audio(try Self.combineAudio(audioResults)))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func unload() {
        state = .idle
        downloadProgress = 0
        availableSpeakers = []
        ttsModelType = "unknown"
        sampleRate = 24_000
        nativeSession = nil
        isLoaded = false
    }

    public func unloadAsync() async {
        unload()
    }

    private struct SynthesizedSegment {
        var audio: AudioResult
        var chunk: AudioChunkResult
    }

    private func synthesizeTextSegments(
        text: String,
        speaker: String?,
        language: String,
        temperature: Float,
        topK: Int,
        maxTokens: Int
    ) throws -> [SynthesizedSegment] {
        let segments = Self.textSegments(from: text)
        var synthesized: [SynthesizedSegment] = []
        synthesized.reserveCapacity(segments.count)
        for index in segments.indices {
            let (audio, chunk, _) = try synthesizeNative(
                text: segments[index],
                speaker: speaker,
                language: language,
                temperature: temperature,
                topK: topK,
                maxTokens: maxTokens,
                chunkIndex: index
            )
            synthesized.append(SynthesizedSegment(audio: audio, chunk: chunk))
        }
        return synthesized
    }

    private func synthesizeNative(
        text: String,
        speaker: String?,
        language: String,
        temperature: Float,
        topK: Int,
        maxTokens: Int,
        chunkIndex: Int = 0
    ) throws -> (AudioResult, AudioChunkResult, Int) {
        guard let nativeSession else {
            throw EdgeRuntimeError.loadFailed("Native TTS session is not initialized")
        }

        let started = Date()
        let result = try nativeSession.synthesize(
            Qwen3TTSSynthesisRequest(
                text: text,
                speaker: speaker,
                language: language,
                maxTokens: maxTokens,
                temperature: temperature,
                topK: topK,
                decodeAudio: true
            )
        )
        guard let audio = result.audio else {
            throw EdgeRuntimeError.loadFailed("Native TTS did not return decoded audio")
        }

        let audioResult = AudioResult(
            samples: audio.interleavedSamples,
            sampleRate: audio.sampleRate
        )
        let chunk = AudioChunkResult(
            samples: audioResult.samples,
            sampleRate: audioResult.sampleRate,
            chunkIndex: chunkIndex,
            generationTimeMs: max(0, Int(Date().timeIntervalSince(started) * 1_000))
        )
        return (audioResult, chunk, result.targetTokenCount)
    }

    private static func combineAudio(_ results: [AudioResult]) throws -> AudioResult {
        guard let first = results.first else {
            throw EdgeRuntimeError.loadFailed("Native TTS did not return decoded audio")
        }
        var samples: [Float] = []
        samples.reserveCapacity(results.reduce(0) { $0 + $1.samples.count })
        for result in results {
            guard result.sampleRate == first.sampleRate else {
                throw EdgeRuntimeError.loadFailed("TTS segment sample-rate mismatch")
            }
            samples.append(contentsOf: result.samples)
        }
        return AudioResult(samples: samples, sampleRate: first.sampleRate)
    }

    private static func textSegments(from text: String) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > maxTextSegmentLength else {
            return trimmed.isEmpty ? [text] : [trimmed]
        }

        let sentenceTerminators = Set("。！？!?；;")
        var segments: [String] = []
        var current = ""

        func flush() {
            let segment = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !segment.isEmpty {
                segments.append(segment)
            }
            current.removeAll(keepingCapacity: true)
        }

        for character in trimmed {
            current.append(character)
            if sentenceTerminators.contains(character), current.count >= maxTextSegmentLength / 3 {
                flush()
            } else if current.count >= maxTextSegmentLength {
                flush()
            }
        }
        flush()
        return segments.isEmpty ? [trimmed] : segments
    }

    private static func defaultTTSMetalConfiguration() -> MetalRuntimeConfiguration {
        let configuration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: 700,
            maxMBPerCommandBuffer: 256,
            contextLengthHint: 4_096,
            dynamicOpsSchedule: DynamicOpsSchedule(
                floor: 5,
                contextLow: 4_096,
                contextHigh: 12_288
            ),
            quantizedBufferCacheLimitBytes: 4_096 * 1_048_576,
            commandBufferBatchingEnabled: true,
            useVendoredCommandBufferPrefillQMM: true,
            useFusedGDNDecode: true
        ).applyingEnvironmentOverrides()
        return NativeRuntimeBridge.applyMetalConfiguration(configuration)
    }
}
