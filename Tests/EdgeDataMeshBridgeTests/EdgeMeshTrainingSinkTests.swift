// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeDataMeshBridge
import EdgeMesh
import XCTest

final class EdgeMeshTrainingSinkTests: XCTestCase {
    func testCollectTrainingSampleMapsKnownTags() throws {
        let (sink, url) = try makeSink()
        defer { try? FileManager.default.removeItem(at: url) }

        try sink.collectTrainingSample(
            appId: "com.test",
            eventType: "transaction",
            payload: Data("{}".utf8),
            tags: ["trainingSample", "retrievableFact"]
        )

        let events = try sink.store.query()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.tags, [.trainingSample, .retrievableFact])
    }

    func testCollectTrainingSampleIgnoresUnknownTags() throws {
        let (sink, url) = try makeSink()
        defer { try? FileManager.default.removeItem(at: url) }

        try sink.collectTrainingSample(
            appId: "com.test",
            eventType: "correction",
            payload: Data("{}".utf8),
            tags: ["trainingSample", "futureTag"]
        )

        let events = try sink.store.query()
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.tags, [.trainingSample])
    }

    func testEventStoreCountReflectsCollectedSamples() throws {
        let (sink, url) = try makeSink()
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(sink.eventStoreCount(), 0)
        try sink.collectTrainingSample(
            appId: "com.test",
            eventType: "x",
            payload: Data("1".utf8),
            tags: ["trainingSample"]
        )
        XCTAssertEqual(sink.eventStoreCount(), 1)
    }

    private func makeSink() throws -> (EdgeMeshTrainingSink, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("edge-mesh-training-sink-\(UUID().uuidString)")
            .appendingPathExtension("sqlite")
        return (try EdgeMeshTrainingSink(storeURL: url), url)
    }
}
