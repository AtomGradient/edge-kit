// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import XCTest
@testable import EdgeInference

final class AudioChunkerResampleTests: XCTestCase {
    func testExtractChunkSamplesUsesIntegerDecimationResampler() throws {
        let url = try writeTone(sampleRate: 48_000, duration: 0.25, frequency: 1_000)
        defer { try? FileManager.default.removeItem(at: url) }

        let chunk = AudioChunkInfo(index: 0, startTime: 0, endTime: 0.25, duration: 0.25)
        let (sampleRate, samples) = try AudioChunker.extractChunkSamples(
            from: url,
            chunk: chunk,
            targetSampleRate: 16_000
        )

        XCTAssertEqual(sampleRate, 16_000)
        XCTAssertEqual(samples.count, 4_000)
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(rms(samples), 0.01)
    }

    func testExtractChunkSamplesUsesAudioConverterForNonIntegerRatios() throws {
        let url = try writeTone(sampleRate: 44_100, duration: 0.2, frequency: 440)
        defer { try? FileManager.default.removeItem(at: url) }

        let chunk = AudioChunkInfo(index: 0, startTime: 0, endTime: 0.2, duration: 0.2)
        let (sampleRate, samples) = try AudioChunker.extractChunkSamples(
            from: url,
            chunk: chunk,
            targetSampleRate: 16_000
        )

        XCTAssertEqual(sampleRate, 16_000)
        XCTAssertTrue((3_180...3_220).contains(samples.count))
        XCTAssertTrue(samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(rms(samples), 0.01)
    }

    func testIntegerDecimationSuppressesAboveOutputNyquistTone() throws {
        let url = try writeTone(sampleRate: 48_000, duration: 0.25, frequency: 12_000)
        defer { try? FileManager.default.removeItem(at: url) }

        let chunk = AudioChunkInfo(index: 0, startTime: 0, endTime: 0.25, duration: 0.25)
        let (_, samples) = try AudioChunker.extractChunkSamples(
            from: url,
            chunk: chunk,
            targetSampleRate: 16_000
        )

        XCTAssertLessThan(rms(samples), 0.01)
    }

    func testAdjacentIntegerDecimatedChunksMatchWholeExtractionAwayFromBoundaries() throws {
        let url = try writeTone(sampleRate: 48_000, duration: 1.0, frequency: 1_000)
        defer { try? FileManager.default.removeItem(at: url) }

        let wholeChunk = AudioChunkInfo(index: 0, startTime: 0, endTime: 1.0, duration: 1.0)
        let (_, whole) = try AudioChunker.extractChunkSamples(
            from: url,
            chunk: wholeChunk,
            targetSampleRate: 16_000
        )

        var stitched: [Float] = []
        for index in 0..<4 {
            let start = Double(index) * 0.25
            let chunk = AudioChunkInfo(
                index: index,
                startTime: start,
                endTime: start + 0.25,
                duration: 0.25
            )
            let (_, samples) = try AudioChunker.extractChunkSamples(
                from: url,
                chunk: chunk,
                targetSampleRate: 16_000
            )
            stitched.append(contentsOf: samples)
        }

        XCTAssertEqual(stitched.count, whole.count)
        XCTAssertLessThan(
            rmsDifference(stitched, whole, ignoringEdges: 96, boundaries: [4_000, 8_000, 12_000]),
            0.001
        )
    }

    private func writeTone(sampleRate: Int, duration: Double, frequency: Double) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-chunker-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        let frameCount = Int((Double(sampleRate) * duration).rounded())
        let samples = (0..<frameCount).map { index in
            Float(0.4 * sin(2 * Double.pi * frequency * Double(index) / Double(sampleRate)))
        }
        let audio = try EdgeAudioBuffer(
            sampleRate: sampleRate,
            channelCount: 1,
            interleavedSamples: samples
        )

        try EdgeWAVFile.encodePCM16(audio).write(to: url)
        return url
    }

    private func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sum = samples.reduce(Float(0)) { partial, sample in
            partial + sample * sample
        }
        return sqrt(sum / Float(samples.count))
    }

    private func rmsDifference(
        _ lhs: [Float],
        _ rhs: [Float],
        ignoringEdges edgeSamples: Int,
        boundaries: [Int]
    ) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > edgeSamples * 2 else { return .infinity }
        var sum = Float(0)
        var compared = 0
        for index in edgeSamples..<(count - edgeSamples) {
            if boundaries.contains(where: { abs(index - $0) <= edgeSamples }) {
                continue
            }
            let delta = lhs[index] - rhs[index]
            sum += delta * delta
            compared += 1
        }
        guard compared > 0 else { return .infinity }
        return sqrt(sum / Float(compared))
    }
}
