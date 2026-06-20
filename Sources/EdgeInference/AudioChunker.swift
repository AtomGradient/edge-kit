// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import AVFoundation
import Accelerate

/// Audio chunk timing metadata.
public struct AudioChunkInfo: Sendable {
    /// Zero-based chunk index.
    public let index: Int
    /// Start time in seconds.
    public let startTime: TimeInterval
    /// End time in seconds.
    public let endTime: TimeInterval
    /// Duration in seconds.
    public let duration: TimeInterval
}

/// Audio chunker that splits long recordings near quiet regions.
///
/// The chunker streams audio blocks, downsamples them for energy analysis, and keeps memory usage bounded.
public enum AudioChunker {

    private static let analysisSR = 8000
    private static let windowMs = 100
    private static let windowSamples = 800

    /// Returns whether an audio file is longer than the chunking threshold.
    public static func needsChunking(_ url: URL, maxDuration: TimeInterval = 600) -> Bool {
        guard let file = try? AVAudioFile(forReading: url) else { return false }
        let duration = Double(file.length) / file.processingFormat.sampleRate
        return duration > maxDuration
    }

    /// Computes chunk boundaries for a long audio file.
    public static func chunkAudio(
        _ url: URL,
        targetDuration: TimeInterval = 120,
        maxDuration: TimeInterval = 600,
        minDuration: TimeInterval = 10
    ) throws -> [AudioChunkInfo]? {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let sampleRate = format.sampleRate
        let totalFrames = file.length
        let totalDuration = Double(totalFrames) / sampleRate

        if totalDuration <= maxDuration {
            return nil
        }

        let decimationRatio = max(1, Int(sampleRate) / analysisSR)
        let blockFrames = AVAudioFrameCount(sampleRate * 10)
        let channelCount = Int(format.channelCount)

        var rmsValues: [Float] = []
        var residualSamples: [Float] = []

        while file.framePosition < totalFrames {
            let framesToRead = min(blockFrames, AVAudioFrameCount(totalFrames - file.framePosition))
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesToRead) else { break }
            try file.read(into: buffer, frameCount: framesToRead)

            guard let channelData = buffer.floatChannelData else { break }
            let count = Int(buffer.frameLength)

            var mono = [Float](repeating: 0, count: count)
            if channelCount == 1 {
                mono = Array(UnsafeBufferPointer(start: channelData[0], count: count))
            } else {
                for ch in 0..<channelCount {
                    let ptr = channelData[ch]
                    for i in 0..<count { mono[i] += ptr[i] }
                }
                let scale = 1.0 / Float(channelCount)
                vDSP.multiply(scale, mono, result: &mono)
            }

            var decimated = [Float]()
            decimated.reserveCapacity(count / decimationRatio + 1)
            for i in stride(from: 0, to: count, by: decimationRatio) {
                decimated.append(mono[i])
            }

            var combined = residualSamples + decimated

            while combined.count >= windowSamples {
                let window = Array(combined.prefix(windowSamples))
                combined = Array(combined.dropFirst(windowSamples))
                var sumSq: Float = 0
                vDSP_svesq(window, 1, &sumSq, vDSP_Length(windowSamples))
                rmsValues.append(sqrt(sumSq / Float(windowSamples)))
            }
            residualSamples = combined
        }

        let nWindows = rmsValues.count
        guard nWindows > 0 else { return nil }

        var rmsSmooth = [Float](repeating: 0, count: nWindows)
        let smoothLen = 5
        for i in 0..<nWindows {
            let lo = max(0, i - smoothLen / 2)
            let hi = min(nWindows, i + smoothLen / 2 + 1)
            var sum: Float = 0
            for j in lo..<hi { sum += rmsValues[j] }
            rmsSmooth[i] = sum / Float(hi - lo)
        }

        var sorted = rmsSmooth
        sorted.sort()
        let median = sorted[nWindows / 2]
        let quietThreshold = median * 0.3

        struct QuietPoint { let time: Double; let energy: Float }
        var quietPoints: [QuietPoint] = []
        var regionStart: Int? = nil

        for i in 0..<nWindows {
            if rmsSmooth[i] < quietThreshold {
                if regionStart == nil { regionStart = i }
            } else {
                if let start = regionStart, i - start >= 2 {
                    let mid = (start + i) / 2
                    quietPoints.append(QuietPoint(
                        time: Double(mid) * Double(windowMs) / 1000.0,
                        energy: rmsSmooth[mid]
                    ))
                }
                regionStart = nil
            }
        }

        var chunks: [AudioChunkInfo] = []
        var chunkStart = 0.0
        var idx = 0

        var targetEnd = targetDuration
        while targetEnd < totalDuration {
            let searchStart = max(chunkStart + minDuration, targetEnd - 15)
            let searchEnd = min(targetEnd + 15, totalDuration)

            let candidates = quietPoints.filter { $0.time >= searchStart && $0.time <= searchEnd }

            let bestTime: Double
            if let best = candidates.min(by: { $0.energy < $1.energy }) {
                bestTime = best.time
            } else {
                let wStart = max(0, Int(searchStart * 1000.0 / Double(windowMs)))
                let wEnd = min(nWindows, Int(searchEnd * 1000.0 / Double(windowMs)))
                if wStart < wEnd {
                    var minIdx = wStart
                    for j in wStart..<wEnd {
                        if rmsSmooth[j] < rmsSmooth[minIdx] { minIdx = j }
                    }
                    bestTime = Double(minIdx) * Double(windowMs) / 1000.0
                } else {
                    bestTime = targetEnd
                }
            }

            let clampedTime = min(bestTime, totalDuration)
            if clampedTime > chunkStart {
                chunks.append(AudioChunkInfo(
                    index: idx, startTime: chunkStart,
                    endTime: clampedTime, duration: clampedTime - chunkStart
                ))
                idx += 1
            }
            chunkStart = clampedTime
            targetEnd = chunkStart + targetDuration
        }

        if chunkStart < totalDuration - 1 {
            chunks.append(AudioChunkInfo(
                index: idx, startTime: chunkStart,
                endTime: totalDuration, duration: totalDuration - chunkStart
            ))
        }

        return chunks
    }

    /// Extracts samples for a single chunk without loading the full file.
    public static func extractChunkSamples(
        from url: URL,
        chunk: AudioChunkInfo,
        targetSampleRate: Int = 16000
    ) throws -> (Int, [Float]) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let sampleRate = format.sampleRate

        let startFrame = AVAudioFramePosition(chunk.startTime * sampleRate)
        let frameCount = AVAudioFrameCount(chunk.duration * sampleRate)

        file.framePosition = startFrame
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw EdgeRuntimeError.loadFailed("Cannot create chunk buffer")
        }
        try file.read(into: buffer, frameCount: frameCount)

        guard let channelData = buffer.floatChannelData else {
            throw EdgeRuntimeError.loadFailed("Cannot read chunk data")
        }

        let count = Int(buffer.frameLength)
        var mono = [Float](repeating: 0, count: count)
        let channels = Int(format.channelCount)

        if channels == 1 {
            mono = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        } else {
            for ch in 0..<channels {
                let ptr = channelData[ch]
                for i in 0..<count { mono[i] += ptr[i] }
            }
            let scale = 1.0 / Float(channels)
            vDSP.multiply(scale, mono, result: &mono)
        }

        let sourceSR = Int(sampleRate)
        if sourceSR != targetSampleRate {
            mono = try resample(mono, from: sourceSR, to: targetSampleRate)
        }

        return (targetSampleRate, mono)
    }

    private static func resample(_ samples: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        guard sourceRate > 0, targetRate > 0, !samples.isEmpty else { return samples }
        guard sourceRate != targetRate else { return samples }
        if sourceRate > targetRate, sourceRate % targetRate == 0 {
            return resampleByIntegerDecimation(samples, factor: sourceRate / targetRate)
        }
        return try resampleWithAudioConverter(samples, from: sourceRate, to: targetRate)
    }

    private static func resampleByIntegerDecimation(_ samples: [Float], factor: Int) -> [Float] {
        guard factor > 1, !samples.isEmpty else { return samples }

        let filterLength = 64
        let leftPadding = filterLength / 2
        let outputCount = max(1, Int((Double(samples.count) / Double(factor)).rounded()))
        let requiredInputCount = (outputCount - 1) * factor + filterLength
        var padded = [Float](repeating: samples[0], count: leftPadding)
        padded.append(contentsOf: samples)
        if padded.count < requiredInputCount {
            padded.append(contentsOf: repeatElement(samples.last ?? 0, count: requiredInputCount - padded.count))
        }

        var output = [Float](repeating: 0, count: outputCount)
        let filter = lowPassFilter(decimationFactor: factor, tapCount: filterLength)
        padded.withUnsafeBufferPointer { input in
            filter.withUnsafeBufferPointer { taps in
                output.withUnsafeMutableBufferPointer { result in
                    guard let inputBase = input.baseAddress,
                          let tapsBase = taps.baseAddress,
                          let resultBase = result.baseAddress else { return }
                    vDSP_desamp(
                        inputBase,
                        vDSP_Stride(factor),
                        tapsBase,
                        resultBase,
                        vDSP_Length(outputCount),
                        vDSP_Length(filter.count)
                    )
                }
            }
        }
        return output
    }

    private static func lowPassFilter(decimationFactor: Int, tapCount: Int) -> [Float] {
        let count = max(8, tapCount)
        let cutoff = 0.45 / Float(decimationFactor)
        let middle = Float(count - 1) / 2
        var taps = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let x = Float(i) - middle
            let sinc: Float
            if abs(x) < .ulpOfOne {
                sinc = 2 * cutoff
            } else {
                sinc = sin(2 * Float.pi * cutoff * x) / (Float.pi * x)
            }
            let window = 0.54 - 0.46 * cos(2 * Float.pi * Float(i) / Float(count - 1))
            taps[i] = sinc * window
        }

        let sum = taps.reduce(0, +)
        if sum != 0 {
            vDSP.divide(taps, sum, result: &taps)
        }
        return taps
    }

    private static func resampleWithAudioConverter(
        _ samples: [Float],
        from sourceRate: Int,
        to targetRate: Int
    ) throws -> [Float] {
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sourceRate),
            channels: 1,
            interleaved: false
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetRate),
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw EdgeRuntimeError.loadFailed("Cannot create audio resampler")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw EdgeRuntimeError.loadFailed("Cannot create resampler input buffer")
        }
        inputBuffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { pointer in
            guard let source = pointer.baseAddress,
                  let destination = inputBuffer.floatChannelData?[0] else { return }
            memcpy(destination, source, samples.count * MemoryLayout<Float>.size)
        }

        let ratio = Double(targetRate) / Double(sourceRate)
        let outputCapacity = max(1, Int(ceil(Double(samples.count) * ratio)) + 128)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(outputCapacity)
        ) else {
            throw EdgeRuntimeError.loadFailed("Cannot create resampler output buffer")
        }

        var hasProvidedInput = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
            if hasProvidedInput {
                outputStatus.pointee = .endOfStream
                return nil
            }
            hasProvidedInput = true
            outputStatus.pointee = .haveData
            return inputBuffer
        }

        if let conversionError {
            throw EdgeRuntimeError.loadFailed("Audio resampling failed: \(conversionError.localizedDescription)")
        }
        guard status == .haveData || status == .inputRanDry || status == .endOfStream else {
            throw EdgeRuntimeError.loadFailed("Audio resampling failed with status \(status.rawValue)")
        }

        let count = Int(outputBuffer.frameLength)
        guard count > 0, let channel = outputBuffer.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: count))
    }
}
