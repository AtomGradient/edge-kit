// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeVoice
import XCTest

final class StreamingAudioPlayerTests: XCTestCase {
    func testStreamingAudioChunkDerivesDurationAndRTF() {
        let chunk = StreamingAudioChunk(
            samples: Array(repeating: 0.25, count: 24_000),
            sampleRate: 24_000,
            chunkIndex: 1,
            generationTimeMs: 500
        )

        XCTAssertEqual(chunk.audioDuration, 1.0, accuracy: 0.0001)
        XCTAssertEqual(chunk.instantaneousRTF, 0.5, accuracy: 0.0001)
    }

    func testStreamingAudioChunkAccountsForChannelCount() {
        let chunk = StreamingAudioChunk(
            samples: Array(repeating: 0.25, count: 48_000),
            sampleRate: 24_000,
            channelCount: 2,
            chunkIndex: 1,
            generationTimeMs: 1_000
        )

        XCTAssertEqual(chunk.audioDuration, 1.0, accuracy: 0.0001)
        XCTAssertEqual(chunk.instantaneousRTF, 1.0, accuracy: 0.0001)
    }
}
