// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation
import AVFoundation

@MainActor
public final class STTEngine {
    private static let preferredSampleRate = 16_000
    private static let maxChunkDuration: TimeInterval = 60
    private static let targetChunkDuration: TimeInterval = 45

    public private(set) var isLoaded = false
    public private(set) var loadedModelName: String?

    private var nativeSession: Qwen3ASRNativeSession?

    public init() {}

    public func loadLocal(directory: URL) async throws {
        _ = try EdgeSpeechBundlePreflightRunner.run(
            configuration: EdgeSpeechBundlePreflightConfiguration(
                modelRootURL: directory,
                modelFamily: .qwen3ASR
            )
        )
        let runtimeConfiguration = Self.defaultASRMetalConfiguration()
        nativeSession = try await Qwen3ASRNativeSession(
            modelURL: directory,
            mode: .transcription,
            runtimeConfiguration: runtimeConfiguration
        )
        isLoaded = true
        loadedModelName = directory.lastPathComponent
    }

    public func unload() {
        nativeSession = nil
        isLoaded = false
        loadedModelName = nil
    }

    public func transcribe(
        audioURL: URL,
        language: String? = nil,
        maxTokens: Int = 8_192,
        temperature: Float = 0.0
    ) async throws -> TranscriptionResult {
        if let chunks = try Self.chunksForTranscription(audioURL) {
            var results: [Qwen3ASRTranscriptionResult] = []
            results.reserveCapacity(chunks.count)

            for chunk in chunks {
                let audio = try Self.loadAudioBuffer(
                    from: audioURL,
                    chunk: chunk,
                    targetSampleRate: Self.preferredSampleRate
                )
                results.append(
                    try transcribeNative(
                        audio: audio,
                        language: language,
                        maxTokens: maxTokens
                    )
                )
            }

            return Self.makeTranscriptionResult(results)
        }

        let audio = try Self.loadAudioBuffer(from: audioURL, targetSampleRate: Self.preferredSampleRate)
        return try transcribe(
            audio: audio,
            language: language,
            maxTokens: maxTokens
        )
    }

    public func transcribe(
        samples: [Float],
        sampleRate: Int = 16_000,
        language: String? = nil
    ) async throws -> TranscriptionResult {
        let audio = try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: 1,
            interleavedSamples: samples
        )
        return try transcribe(audio: audio, language: language, maxTokens: 8_192)
    }

    public func transcribeStream(
        audioURL: URL,
        language: String? = nil
    ) -> AsyncThrowingStream<STTStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor [weak self] in
                do {
                    guard let self else {
                        continuation.finish(throwing: EdgeRuntimeError.loadFailed("No STT engine loaded"))
                        return
                    }
                    guard self.isLoaded else {
                        throw EdgeRuntimeError.loadFailed("No STT model loaded")
                    }
                    guard let nativeSession = self.nativeSession else {
                        throw EdgeRuntimeError.loadFailed("Native ASR session is not initialized")
                    }

                    let nativeResults: [Qwen3ASRTranscriptionResult]
                    if let chunks = try Self.chunksForTranscription(audioURL) {
                        var results: [Qwen3ASRTranscriptionResult] = []
                        results.reserveCapacity(chunks.count)

                        for index in chunks.indices {
                            let chunk = chunks[index]
                            let audio = try Self.loadAudioBuffer(
                                from: audioURL,
                                chunk: chunk,
                                targetSampleRate: Self.preferredSampleRate
                            )
                            let nativeResult = try self.transcribeNative(
                                audio: audio,
                                language: language,
                                maxTokens: 8_192
                            )
                            let trimmed = nativeResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                let suffix = index < chunks.index(before: chunks.endIndex) ? " " : ""
                                continuation.yield(.token(trimmed + suffix))
                            }
                            results.append(nativeResult)
                        }
                        nativeResults = results
                    } else {
                        let audio = try Self.loadAudioBuffer(
                            from: audioURL,
                            targetSampleRate: Self.preferredSampleRate
                        )
                        let nativeResult = try nativeSession.transcribe(
                            Qwen3ASRTranscriptionRequest(
                                audio: audio,
                                language: language ?? "English",
                                maxTokens: 8_192
                            ),
                            onTextDelta: { delta in
                                continuation.yield(.token(delta))
                            }
                        )
                        nativeResults = [nativeResult]
                    }

                    let result = Self.makeTranscriptionResult(nativeResults)
                    continuation.yield(
                        .info(
                            STTStreamInfo(
                                promptTokenCount: result.promptTokens,
                                generationTokenCount: result.generationTokens,
                                tokensPerSecond: result.generationTps
                            )
                        )
                    )
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func transcribe(
        audio: EdgeAudioBuffer,
        language: String?,
        maxTokens: Int
    ) throws -> TranscriptionResult {
        let normalizedAudio = try audio.resampled(to: Self.preferredSampleRate)
        if let chunks = try Self.chunksForTranscription(normalizedAudio) {
            var results: [Qwen3ASRTranscriptionResult] = []
            results.reserveCapacity(chunks.count)
            for chunk in chunks {
                results.append(
                    try transcribeNative(
                        audio: chunk,
                        language: language,
                        maxTokens: maxTokens
                    )
                )
            }
            return Self.makeTranscriptionResult(results)
        }

        return try Self.makeTranscriptionResult(
            transcribeNative(
                audio: normalizedAudio,
                language: language,
                maxTokens: maxTokens
            )
        )
    }

    private func transcribeNative(
        audio: EdgeAudioBuffer,
        language: String?,
        maxTokens: Int
    ) throws -> Qwen3ASRTranscriptionResult {
        guard isLoaded else { throw EdgeRuntimeError.loadFailed("No STT model loaded") }
        guard let nativeSession else {
            throw EdgeRuntimeError.loadFailed("Native ASR session is not initialized")
        }
        let nativeResult = try nativeSession.transcribe(
            Qwen3ASRTranscriptionRequest(
                audio: audio,
                language: language ?? "English",
                maxTokens: maxTokens
            )
        )
        return nativeResult
    }

    private static func makeTranscriptionResult(
        _ result: Qwen3ASRTranscriptionResult
    ) -> TranscriptionResult {
        let totalTime = result.prefillSeconds + result.decodeSeconds
        let totalTokens = result.promptTokenCount + result.generatedTokenIDs.count
        return TranscriptionResult(
            text: result.text,
            segments: nil,
            language: result.language,
            promptTokens: result.promptTokenCount,
            generationTokens: result.generatedTokenIDs.count,
            totalTokens: totalTokens,
            promptTps: tokensPerSecond(result.promptTokenCount, result.prefillSeconds),
            generationTps: tokensPerSecond(result.generatedTokenIDs.count, result.decodeSeconds),
            totalTime: totalTime
        )
    }

    private static func makeTranscriptionResult(
        _ results: [Qwen3ASRTranscriptionResult]
    ) -> TranscriptionResult {
        guard !results.isEmpty else {
            return TranscriptionResult(
                text: "",
                segments: nil,
                language: nil,
                promptTokens: 0,
                generationTokens: 0,
                totalTokens: 0,
                promptTps: 0,
                generationTps: 0,
                totalTime: 0
            )
        }

        if results.count == 1, let result = results.first {
            return makeTranscriptionResult(result)
        }

        let text = results
            .map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        let promptTokens = results.reduce(0) { $0 + $1.promptTokenCount }
        let generationTokens = results.reduce(0) { $0 + $1.generatedTokenIDs.count }
        let prefillSeconds = results.reduce(0) { $0 + $1.prefillSeconds }
        let decodeSeconds = results.reduce(0) { $0 + $1.decodeSeconds }
        let totalTime = prefillSeconds + decodeSeconds

        return TranscriptionResult(
            text: text,
            segments: nil,
            language: results.first?.language,
            promptTokens: promptTokens,
            generationTokens: generationTokens,
            totalTokens: promptTokens + generationTokens,
            promptTps: tokensPerSecond(promptTokens, prefillSeconds),
            generationTps: tokensPerSecond(generationTokens, decodeSeconds),
            totalTime: totalTime
        )
    }

    private static func tokensPerSecond(_ tokens: Int, _ seconds: TimeInterval) -> Double {
        seconds > 0 ? Double(tokens) / seconds : 0
    }

    private static func defaultASRMetalConfiguration() -> MetalRuntimeConfiguration {
        let configuration = NativeRuntimeBridge.metalConfiguration(
            maxOpsPerCommandBuffer: 700,
            maxMBPerCommandBuffer: 256,
            contextLengthHint: 8_192 + 1_024,
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

    private static func chunksForTranscription(_ url: URL) throws -> [AudioChunkInfo]? {
        guard AudioChunker.needsChunking(url, maxDuration: maxChunkDuration) else {
            return nil
        }
        return try AudioChunker.chunkAudio(
            url,
            targetDuration: targetChunkDuration,
            maxDuration: maxChunkDuration
        )
    }

    private static func chunksForTranscription(_ audio: EdgeAudioBuffer) throws -> [EdgeAudioBuffer]? {
        guard audio.durationSeconds > maxChunkDuration else {
            return nil
        }

        let monoSamples = audio.monoSamples()
        let chunkFrames = max(1, Int(targetChunkDuration * Double(audio.sampleRate)))
        var chunks: [EdgeAudioBuffer] = []
        chunks.reserveCapacity(Int((audio.durationSeconds / targetChunkDuration).rounded(.up)))

        var start = 0
        while start < monoSamples.count {
            let end = min(start + chunkFrames, monoSamples.count)
            let samples = Array(monoSamples[start..<end])
            chunks.append(
                try EdgeAudioBuffer(
                    sampleRate: audio.sampleRate,
                    channelCount: 1,
                    interleavedSamples: samples
                )
            )
            start = end
        }

        return chunks
    }

    private static func loadAudioBuffer(from url: URL, targetSampleRate: Int) throws -> EdgeAudioBuffer {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw EdgeAudioError.emptySamples
        }
        try file.read(into: buffer)
        guard let channelData = buffer.floatChannelData else {
            throw EdgeAudioError.emptySamples
        }

        let channels = Int(format.channelCount)
        let frames = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: frames)
        for frame in 0..<frames {
            var sum = Float.zero
            for channel in 0..<channels {
                sum += channelData[channel][frame]
            }
            samples[frame] = sum / Float(channels)
        }

        let audio = try EdgeAudioBuffer(
            sampleRate: Int(format.sampleRate),
            channelCount: 1,
            interleavedSamples: samples
        )
        return try audio.resampled(to: targetSampleRate)
    }

    private static func loadAudioBuffer(
        from url: URL,
        chunk: AudioChunkInfo,
        targetSampleRate: Int
    ) throws -> EdgeAudioBuffer {
        let (sampleRate, samples) = try AudioChunker.extractChunkSamples(
            from: url,
            chunk: chunk,
            targetSampleRate: targetSampleRate
        )
        return try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: 1,
            interleavedSamples: samples
        )
    }
}

public extension STTEngine {
    func prepareNativeASRRequest(
        audio: EdgeAudioBuffer,
        language: String? = nil,
        maxTokens: Int = 8_192
    ) throws -> EdgeASRRequest {
        let plan = try NativeRuntimeBridge.makeQwen3ASRPlan()
        let normalizedAudio = try audio.resampled(to: plan.preferredSampleRate)
        return EdgeASRRequest(
            audio: normalizedAudio,
            languageHint: language,
            maxTokens: maxTokens,
            featureConfiguration: try NativeRuntimeBridge.qwenASRFeatureConfiguration()
        )
    }

    func prepareNativeASRRequest(
        samples: [Float],
        sampleRate: Int = 16_000,
        language: String? = nil,
        maxTokens: Int = 8_192
    ) throws -> EdgeASRRequest {
        let audio = try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: 1,
            interleavedSamples: samples
        )
        return try prepareNativeASRRequest(
            audio: audio,
            language: language,
            maxTokens: maxTokens
        )
    }

    func prepareNativeASRRequest(
        wavData: Data,
        language: String? = nil,
        maxTokens: Int = 8_192
    ) throws -> EdgeASRRequest {
        try prepareNativeASRRequest(
            audio: EdgeWAVFile.decode(wavData),
            language: language,
            maxTokens: maxTokens
        )
    }

    func prepareNativeASRRequest(
        wavURL: URL,
        language: String? = nil,
        maxTokens: Int = 8_192
    ) throws -> EdgeASRRequest {
        try prepareNativeASRRequest(
            wavData: Data(contentsOf: wavURL),
            language: language,
            maxTokens: maxTokens
        )
    }

    func extractNativeASRFeatures(
        samples: [Float],
        sampleRate: Int = 16_000
    ) throws -> EdgeLogMelSpectrogram {
        try prepareNativeASRRequest(samples: samples, sampleRate: sampleRate)
            .logMelFeatures()
    }
}
