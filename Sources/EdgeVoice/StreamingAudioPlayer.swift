// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import AVFoundation
import Combine
import Foundation

public struct StreamingAudioChunk: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let channelCount: Int
    public let chunkIndex: Int
    public let generationTimeMs: Int

    public init(
        samples: [Float],
        sampleRate: Int,
        channelCount: Int = 1,
        chunkIndex: Int,
        generationTimeMs: Int
    ) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.channelCount = max(1, channelCount)
        self.chunkIndex = chunkIndex
        self.generationTimeMs = max(0, generationTimeMs)
    }

    public var audioDuration: TimeInterval {
        guard sampleRate > 0, channelCount > 0 else { return 0 }
        return Double(samples.count / channelCount) / Double(sampleRate)
    }

    public var instantaneousRTF: Double {
        guard audioDuration > 0 else { return 0 }
        return Double(generationTimeMs) / 1_000.0 / audioDuration
    }
}

public enum StreamingAudioPlayerError: Error, LocalizedError {
    case invalidFormat(sampleRate: Int, channelCount: Int)
    case emptySamples
    case channelFrameMismatch(sampleCount: Int, channelCount: Int)
    case engineStartFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidFormat(let sampleRate, let channelCount):
            return "Invalid audio format: sampleRate=\(sampleRate), channelCount=\(channelCount)"
        case .emptySamples:
            return "Audio chunk is empty"
        case .channelFrameMismatch(let sampleCount, let channelCount):
            return "Audio sample count \(sampleCount) is not divisible by channel count \(channelCount)"
        case .engineStartFailed(let reason):
            return "Audio engine start failed: \(reason)"
        }
    }
}

@MainActor
public final class StreamingAudioPlayer: ObservableObject {
    @Published public private(set) var isPlaying = false

    public private(set) var scheduledDuration: TimeInterval = 0
    public private(set) var steadyStateRTF: Double = 0

    public var adaptiveRTFThreshold: Double

    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var format: AVAudioFormat?
    private var scheduledBufferCount = 0
    private var lastBufferFinished = true
    private var isFirstBuffer = true

    private var pendingSamples: [Float] = []
    private var pendingDuration: TimeInterval = 0
    private var streamingDecision: Bool?

    public init(adaptiveRTFThreshold: Double = 0.95) {
        self.adaptiveRTFThreshold = adaptiveRTFThreshold
    }

    public func start(sampleRate: Int, channelCount: Int = 1) throws {
        guard sampleRate > 0, channelCount > 0 else {
            throw StreamingAudioPlayerError.invalidFormat(
                sampleRate: sampleRate,
                channelCount: channelCount
            )
        }

        stop()

#if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
#endif

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        guard let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channelCount)
        ) else {
            throw StreamingAudioPlayerError.invalidFormat(
                sampleRate: sampleRate,
                channelCount: channelCount
            )
        }

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        do {
            try engine.start()
        } catch {
            throw StreamingAudioPlayerError.engineStartFailed(error.localizedDescription)
        }
        player.play()

        self.engine = engine
        self.playerNode = player
        self.format = format
        scheduledBufferCount = 0
        lastBufferFinished = true
        isFirstBuffer = true
        scheduledDuration = 0
        pendingSamples = []
        pendingDuration = 0
        streamingDecision = nil
        steadyStateRTF = 0
        isPlaying = false
    }

    public func scheduleChunk(_ chunk: StreamingAudioChunk) throws {
        if format == nil {
            try start(sampleRate: chunk.sampleRate, channelCount: chunk.channelCount)
        } else {
            try validateCurrentFormat(
                sampleRate: chunk.sampleRate,
                channelCount: chunk.channelCount
            )
        }
        try validate(samples: chunk.samples, channelCount: chunk.channelCount)

        if chunk.chunkIndex >= 1 {
            steadyStateRTF = chunk.instantaneousRTF
        }

        if let decision = streamingDecision {
            if decision {
                try scheduleBuffer(chunk.samples, channelCount: chunk.channelCount)
            } else {
                pendingSamples.append(contentsOf: chunk.samples)
                pendingDuration += chunk.audioDuration
            }
            return
        }

        if chunk.chunkIndex == 0 {
            pendingSamples.append(contentsOf: chunk.samples)
            pendingDuration += chunk.audioDuration
        } else if steadyStateRTF < adaptiveRTFThreshold {
            streamingDecision = true
            try flushPending(channelCount: chunk.channelCount)
            try scheduleBuffer(chunk.samples, channelCount: chunk.channelCount)
        } else {
            streamingDecision = false
            pendingSamples.append(contentsOf: chunk.samples)
            pendingDuration += chunk.audioDuration
        }
    }

    public func scheduleChunk(
        samples: [Float],
        sampleRate: Int,
        channelCount: Int = 1
    ) throws {
        if format == nil {
            try start(sampleRate: sampleRate, channelCount: channelCount)
        } else {
            try validateCurrentFormat(sampleRate: sampleRate, channelCount: channelCount)
        }
        try scheduleBuffer(samples, channelCount: channelCount)
    }

    public func markComplete() throws {
        let channelCount = Int(format?.channelCount ?? 1)
        try flushPending(channelCount: channelCount)
    }

    public func waitForPlaybackEnd(pollInterval: TimeInterval = 0.1) async {
        guard scheduledBufferCount > 0 else { return }
        while !lastBufferFinished {
            try? await Task.sleep(nanoseconds: UInt64(max(0.01, pollInterval) * 1_000_000_000))
        }
    }

    public func stop() {
        playerNode?.stop()
        engine?.stop()
        playerNode = nil
        engine = nil
        format = nil
        scheduledBufferCount = 0
        lastBufferFinished = true
        isFirstBuffer = true
        pendingSamples = []
        pendingDuration = 0
        streamingDecision = nil
        scheduledDuration = 0
        steadyStateRTF = 0
        isPlaying = false
#if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false)
#endif
    }

    private func flushPending(channelCount: Int) throws {
        guard !pendingSamples.isEmpty else { return }
        try scheduleBuffer(pendingSamples, channelCount: channelCount)
        pendingSamples.removeAll(keepingCapacity: true)
        pendingDuration = 0
    }

    private func scheduleBuffer(_ samples: [Float], channelCount: Int) throws {
        guard let player = playerNode, let format else { return }
        try validateCurrentFormat(
            sampleRate: Int(format.sampleRate),
            channelCount: channelCount
        )
        try validate(samples: samples, channelCount: channelCount)

        let frameCount = samples.count / channelCount
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            throw StreamingAudioPlayerError.invalidFormat(
                sampleRate: Int(format.sampleRate),
                channelCount: channelCount
            )
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        fill(buffer: buffer, samples: samples, channelCount: channelCount)

        scheduledBufferCount += 1
        lastBufferFinished = false
        isPlaying = true
        let bufferIndex = scheduledBufferCount
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                if bufferIndex == self.scheduledBufferCount {
                    self.lastBufferFinished = true
                    self.isPlaying = false
                }
            }
        }

        scheduledDuration += Double(frameCount) / format.sampleRate
    }

    private func fill(
        buffer: AVAudioPCMBuffer,
        samples: [Float],
        channelCount: Int
    ) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let fadeInFrames = isFirstBuffer
            ? min(Int(buffer.format.sampleRate * 0.01), frameCount)
            : 0

        for channel in 0..<channelCount {
            let destination = channelData[channel]
            for frame in 0..<frameCount {
                let sample = samples[frame * channelCount + channel]
                if frame < fadeInFrames {
                    destination[frame] = sample * (Float(frame) / Float(max(1, fadeInFrames)))
                } else {
                    destination[frame] = sample
                }
            }
        }
        isFirstBuffer = false
    }

    private func validate(samples: [Float], channelCount: Int) throws {
        guard !samples.isEmpty else { throw StreamingAudioPlayerError.emptySamples }
        guard channelCount > 0 else {
            throw StreamingAudioPlayerError.invalidFormat(sampleRate: 0, channelCount: channelCount)
        }
        guard samples.count.isMultiple(of: channelCount) else {
            throw StreamingAudioPlayerError.channelFrameMismatch(
                sampleCount: samples.count,
                channelCount: channelCount
            )
        }
    }

    private func validateCurrentFormat(sampleRate: Int, channelCount: Int) throws {
        guard let format else { return }
        let expectedSampleRate = Int(format.sampleRate.rounded())
        let expectedChannelCount = Int(format.channelCount)
        guard sampleRate == expectedSampleRate, channelCount == expectedChannelCount else {
            throw StreamingAudioPlayerError.invalidFormat(
                sampleRate: sampleRate,
                channelCount: channelCount
            )
        }
    }
}
