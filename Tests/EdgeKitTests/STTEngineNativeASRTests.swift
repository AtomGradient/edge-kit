// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation
import XCTest
@testable import EdgeInference

final class STTEngineNativeASRTests: XCTestCase {
    @MainActor
    func testPublicBuildCanPrepareNativeASRFeaturesWithoutLegacyAudioRuntime() throws {
        let engine = STTEngine()
        let sourceRate = 8_000
        let samples = (0..<640).map { index in
            Float(sin(2.0 * Double.pi * 440.0 * Double(index) / Double(sourceRate)))
        }

        let request = try engine.prepareNativeASRRequest(
            samples: samples,
            sampleRate: sourceRate,
            language: "zh",
            maxTokens: 256
        )
        let features = try request.logMelFeatures()

        XCTAssertEqual(request.audio.sampleRate, 16_000)
        XCTAssertEqual(request.audio.channelCount, 1)
        XCTAssertEqual(request.languageHint, "zh")
        XCTAssertEqual(request.maxTokens, 256)
        XCTAssertEqual(features.configuration.sampleRate, 16_000)
        XCTAssertEqual(features.configuration.melBinCount, 128)
        XCTAssertFalse(features.frames.isEmpty)
        XCTAssertEqual(features.frames.first?.count, 128)
        XCTAssertTrue(features.frames.flatMap { $0 }.allSatisfy { $0.isFinite })
    }

    @MainActor
    func testPublicBuildCanPrepareNativeASRFeaturesFromWAV() throws {
        let engine = STTEngine()
        let audio = try EdgeAudioBuffer(
            sampleRate: 16_000,
            channelCount: 1,
            interleavedSamples: Array(repeating: 0.1, count: 480)
        )
        let wavData = try EdgeWAVFile.encodePCM16(audio)

        let request = try engine.prepareNativeASRRequest(wavData: wavData)

        XCTAssertEqual(request.audio.sampleRate, 16_000)
        XCTAssertEqual(request.audio.frameCount, 480)
        XCTAssertNoThrow(try request.logMelFeatures())
    }
}
