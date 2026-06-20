// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class EdgeMeshDataCollectorTests: XCTestCase {

    private func makeTempStore() throws -> (EventStore, URL) {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("eventstore_\(UUID()).sqlite")
        let store = try EventStore(url: tmp)
        return (store, tmp)
    }

    func testEventStore_InsertAndQuery() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let event = DataEvent(
            appId: "com.edgestudio.dailyn",
            eventType: "transaction",
            payload: Data("{\"amount\":8.6}".utf8),
            tags: [.trainingSample, .retrievableFact, .aggregatable]
        )
        try store.insert(event)

        XCTAssertEqual(try store.count(), 1)

        let results = try store.query()
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results.first?.id, event.id)
        XCTAssertEqual(results.first?.eventType, "transaction")
        XCTAssertEqual(results.first?.tags, event.tags)
    }

    func testEventStore_IdempotentInsert() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let e = DataEvent(appId: "app", eventType: "x", payload: Data(), tags: [])
        try store.insert(e)
        try store.insert(e)
        XCTAssertEqual(try store.count(), 1)
    }

    func testEventStore_QueryByTag() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let trainingEvent = DataEvent(appId: "a", eventType: "t1", payload: Data(), tags: [.trainingSample])
        let factEvent = DataEvent(appId: "a", eventType: "t2", payload: Data(), tags: [.retrievableFact])
        let bothEvent = DataEvent(appId: "a", eventType: "t3", payload: Data(), tags: [.trainingSample, .retrievableFact])

        try store.insert(trainingEvent)
        try store.insert(factEvent)
        try store.insert(bothEvent)

        let training = try store.query(tags: [.trainingSample])
        XCTAssertEqual(training.count, 2)
        XCTAssertTrue(training.contains { $0.id == trainingEvent.id })
        XCTAssertTrue(training.contains { $0.id == bothEvent.id })

        let facts = try store.query(tags: [.retrievableFact])
        XCTAssertEqual(facts.count, 2)
    }

    func testEventStore_QueryByAppId() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        try store.insert(DataEvent(appId: "com.app.one", eventType: "x", payload: Data(), tags: []))
        try store.insert(DataEvent(appId: "com.app.two", eventType: "x", payload: Data(), tags: []))

        let one = try store.query(appId: "com.app.one")
        XCTAssertEqual(one.count, 1)
        XCTAssertEqual(one.first?.appId, "com.app.one")
    }

    func testEventStore_QueryOrderedByTimestamp() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let older = DataEvent(timestamp: Date(timeIntervalSince1970: 1000), appId: "a", eventType: "x", payload: Data(), tags: [])
        let newer = DataEvent(timestamp: Date(timeIntervalSince1970: 2000), appId: "a", eventType: "x", payload: Data(), tags: [])

        try store.insert(newer)
        try store.insert(older)

        let all = try store.query()
        XCTAssertEqual(all.count, 2)
        XCTAssertEqual(all[0].id, older.id)
        XCTAssertEqual(all[1].id, newer.id)
    }

    func testEventStore_InsertBatch() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let events = (0..<5).map { i in
            DataEvent(
                appId: "com.app",
                eventType: "batch",
                payload: Data("item_\(i)".utf8),
                tags: [.trainingSample]
            )
        }
        let received = try store.insertBatch(events)
        XCTAssertEqual(received.count, 5)
        XCTAssertEqual(try store.count(), 5)
    }

    func testEventStore_UnsyncedAndMarkSynced() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let e1 = DataEvent(appId: "a", eventType: "x", payload: Data(), tags: [])
        let e2 = DataEvent(appId: "a", eventType: "x", payload: Data(), tags: [])
        try store.insert(e1)
        try store.insert(e2)

        let peerId = "mac-studio-abc"

        let unsynced = try store.unsyncedEvents(for: peerId)
        XCTAssertEqual(unsynced.count, 2)

        try store.markSynced(eventIds: [e1.id], peerId: peerId)
        let remaining = try store.unsyncedEvents(for: peerId)
        XCTAssertEqual(remaining.count, 1)
        XCTAssertEqual(remaining.first?.id, e2.id)

        let otherPeerUnsynced = try store.unsyncedEvents(for: "other-peer")
        XCTAssertEqual(otherPeerUnsynced.count, 2)
    }

    func testEventStore_Stats() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        try store.insert(DataEvent(appId: "a", eventType: "tx", payload: Data(count: 10), tags: [.trainingSample]))
        try store.insert(DataEvent(appId: "a", eventType: "tx", payload: Data(count: 20), tags: [.retrievableFact]))
        try store.insert(DataEvent(appId: "a", eventType: "chat", payload: Data(count: 5), tags: [.trainingSample, .conversation]))

        let stats = try store.stats()
        XCTAssertEqual(stats.totalEvents, 3)
        XCTAssertEqual(stats.totalBytes, 35)
        XCTAssertEqual(stats.perTypeCounts["tx"], 2)
        XCTAssertEqual(stats.perTypeCounts["chat"], 1)
        XCTAssertEqual(stats.perTagCounts["trainingSample"], 2)
        XCTAssertEqual(stats.perTagCounts["retrievableFact"], 1)
        XCTAssertEqual(stats.perTagCounts["conversation"], 1)
    }

    func testEventStore_Purge() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }

        let old = DataEvent(timestamp: Date(timeIntervalSince1970: 1000), appId: "a", eventType: "x", payload: Data(), tags: [])
        let new = DataEvent(timestamp: Date(timeIntervalSince1970: 10000), appId: "a", eventType: "x", payload: Data(), tags: [])
        try store.insert(old)
        try store.insert(new)

        let deleted = try store.purge(olderThan: Date(timeIntervalSince1970: 5000))
        XCTAssertEqual(deleted, 1)
        XCTAssertEqual(try store.count(), 1)
    }

    func testDataCollector_AppendAndQuery() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        _ = try collector.appendEvent(
            appId: "com.edgestudio.dailyn",
            eventType: "transaction",
            payload: Data("{\"amount\":10}".utf8),
            tags: [.trainingSample, .retrievableFact]
        )
        XCTAssertEqual(try collector.count(), 1)
        let results = try collector.query(tags: [.retrievableFact])
        XCTAssertEqual(results.count, 1)
    }

    func testDataCollector_T11_CollectConversation() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        let messages = [
            ChatMessage(role: "user", content: "Hello"),
            ChatMessage(role: "assistant", content: "Hi!")
        ]
        let event = try collector.collectConversation(
            messages: messages,
            systemPrompt: "You are helpful.",
            appId: "com.test.app"
        )
        XCTAssertEqual(event.eventType, "conversation")
        XCTAssertEqual(event.appId, "com.test.app")
        XCTAssertTrue(event.tags.contains(.trainingSample))
        XCTAssertTrue(event.tags.contains(.conversation))

        let pair = try JSONDecoder().decode(TrainingPair.self, from: event.payload)
        XCTAssertEqual(pair.messages.count, 2)
        XCTAssertEqual(pair.systemPrompt, "You are helpful.")
    }

    func testDataCollector_T11_CollectCorrection() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        let event = try collector.collectCorrection(
            context: [ChatMessage(role: "user", content: "Q")],
            original: "bad answer",
            corrected: "good answer",
            systemPrompt: nil,
            appId: "com.test.app"
        )
        XCTAssertEqual(event.eventType, "user_correction")
        XCTAssertTrue(event.tags.contains(.userCorrection))
        XCTAssertTrue(event.tags.contains(.preference))

        let pair = try JSONDecoder().decode(TrainingPair.self, from: event.payload)
        XCTAssertEqual(pair.preferredResponse, "good answer")
        XCTAssertEqual(pair.rejectedResponse, "bad answer")
    }

    func testDataCollector_CollectPersonaCorrectionSignal() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)
        let signal = PersonaCorrectionSignalPayload(
            appID: "com.test.app",
            source: "receipt_review",
            factCorrection: PersonaFactCorrectionSignal(
                normalizedFields: ["task_label": .string("dinner")]
            )
        )

        let event = try collector.collectPersonaCorrectionSignal(signal)

        XCTAssertEqual(event.eventType, PersonaCorrectionSignalPayload.eventType)
        XCTAssertEqual(event.appId, "com.test.app")
        XCTAssertTrue(event.tags.contains(.userCorrection))
        XCTAssertTrue(event.tags.contains(.trainingSample))
        let decoded = try JSONDecoder().decode(
            PersonaCorrectionSignalPayload.self,
            from: event.payload
        )
        XCTAssertEqual(decoded.schemaVersion, PersonaCorrectionSignalPayload.schemaVersion)
        XCTAssertEqual(decoded.correctionFingerprint, signal.correctionFingerprint)
        XCTAssertEqual(
            decoded.factCorrection?.normalizedFields["task_label"],
            .string("dinner")
        )
    }

    func testPersonaCorrectionSignalRecorder_RecordFactCorrection() throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        let result = try PersonaCorrectionSignalRecorder.recordFactCorrection(
            appID: "com.test.app",
            source: "tool_update",
            normalizedFields: ["category": .string("餐饮")],
            collector: collector,
            createdAt: 12
        )

        XCTAssertNotNil(result.eventID)
        XCTAssertEqual(result.signal.factCorrection?.normalizedFields["category"], .string("餐饮"))
        let events = try collector.query(tags: [.userCorrection], appId: "com.test.app")
        XCTAssertEqual(events.count, 1)
        XCTAssertEqual(events.first?.id, result.eventID)
        XCTAssertEqual(events.first?.eventType, PersonaCorrectionSignalPayload.eventType)
    }

    func testPersonaCorrectionSignalRecorder_RejectsEmptyFactCorrection() throws {
        XCTAssertThrowsError(
            try PersonaCorrectionSignalRecorder.recordFactCorrection(
                appID: "com.test.app",
                source: "tool_update",
                normalizedFields: [:],
                collector: nil
            )
        ) { error in
            XCTAssertEqual(
                error as? PersonaCorrectionSignalRecorderError,
                .emptyFactCorrection
            )
        }
    }

    func testPersonaCorrectionFieldNormalizer_StringFieldsFiltersAndTruncates() {
        let normalized = PersonaCorrectionFieldNormalizer.normalizeStringFields(
            [
                "category": "  餐饮  ",
                "description": "abcdefghijklmnopqrstuvwxyz",
                "amount": "100",
                "empty": "   ",
            ],
            policy: PersonaCorrectionFieldNormalizationPolicy(
                allowedFields: Set(["category", "description", "empty"]),
                maxStringLength: 5
            )
        )

        XCTAssertEqual(normalized["category"], .string("餐饮"))
        XCTAssertEqual(normalized["description"], .string("abcde"))
        XCTAssertNil(normalized["amount"])
        XCTAssertNil(normalized["empty"])
    }

    func testPersonaCorrectionSignalFingerprintIncludesEventTime() {
        let first = PersonaCorrectionSignalPayload(
            appID: "com.test.app",
            source: "receipt_review",
            factCorrection: PersonaFactCorrectionSignal(
                normalizedFields: ["task_label": .string("dinner")]
            ),
            createdAt: 1
        )
        let second = PersonaCorrectionSignalPayload(
            appID: "com.test.app",
            source: "receipt_review",
            factCorrection: PersonaFactCorrectionSignal(
                normalizedFields: ["task_label": .string("dinner")]
            ),
            createdAt: 2
        )

        XCTAssertNotEqual(first.correctionFingerprint, second.correctionFingerprint)
    }

    func testDataCollector_FlushHonorsServerAck() async throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        var ids: [UUID] = []
        for i in 0..<3 {
            let e = try collector.appendEvent(
                appId: "com.test", eventType: "x", payload: Data("e\(i)".utf8), tags: [.trainingSample]
            )
            ids.append(e.id)
        }

        let peerId = "brain-1"
        let uploaded = try await collector.flush(to: peerId) { batch in
            XCTAssertEqual(batch.events.count, 3)
            return EventUploadAck(receivedIds: Array(batch.events.prefix(2).map(\.id)))
        }
        XCTAssertEqual(uploaded, 2)

        let stillUnsynced = try store.unsyncedEvents(for: peerId)
        XCTAssertEqual(stillUnsynced.count, 1)
        XCTAssertEqual(stillUnsynced.first?.id, ids[2])
    }

    func testEventSyncClient_EncodesUploadAndHonorsAck() async throws {
        let (store, url) = try makeTempStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let collector = DataCollector(store: store)

        for i in 0..<3 {
            _ = try collector.appendEvent(
                appId: "com.test",
                eventType: "x",
                payload: Data("e\(i)".utf8),
                tags: [.trainingSample]
            )
        }

        var frameHandler: ((Data) -> Void)?
        let client = EventSyncClient(
            collector: collector,
            sendFrame: { data in
                let batch = try Self.decodeEventUploadBatch(from: data)
                XCTAssertEqual(batch.events.count, 3)
                let ack = try Self.makeEventUploadAckFrame(
                    receivedIds: Array(batch.events.prefix(2).map(\.id))
                )
                frameHandler?(ack)
            },
            registerFrameHandler: { handler in
                frameHandler = handler
            },
            configuration: .init(ackTimeout: 1)
        )

        let synced = try await client.flush(to: "brain-1")
        XCTAssertEqual(synced, 2)
        let remaining = try store.unsyncedEvents(for: "brain-1")
        XCTAssertEqual(remaining.count, 1)
    }

    func testEventSyncClient_IgnoresNonAckFrames() throws {
        let event = DataEvent(
            appId: "a",
            eventType: "x",
            payload: Data("p".utf8),
            tags: [.trainingSample]
        )
        let batch = EventUploadBatch(events: [event])
        let data = try EventSyncClient.makeEventUploadFrame(batch: batch)
        XCTAssertNil(EventSyncClient.decodeEventUploadAckFrame(data))
    }

    func testEventUploadBatch_Codable() throws {
        let events = [
            DataEvent(appId: "a", eventType: "x", payload: Data("p1".utf8), tags: [.trainingSample]),
            DataEvent(appId: "b", eventType: "y", payload: Data("p2".utf8), tags: [.retrievableFact, .aggregatable])
        ]
        let batch = EventUploadBatch(events: events)
        let encoded = try JSONEncoder().encode(batch)
        let decoded = try JSONDecoder().decode(EventUploadBatch.self, from: encoded)
        XCTAssertEqual(decoded.events.count, 2)
        XCTAssertEqual(decoded.events[0].tags, events[0].tags)
        XCTAssertEqual(decoded.events[1].payload, events[1].payload)
    }

    private static func decodeEventUploadBatch(from data: Data) throws -> EventUploadBatch {
        struct Envelope: Decodable {
            let op: String
            let payload: EventUploadBatch
        }

        let envelope = try JSONDecoder().decode(Envelope.self, from: data)
        XCTAssertEqual(envelope.op, "event_upload")
        return envelope.payload
    }

    private static func makeEventUploadAckFrame(receivedIds: [UUID]) throws -> Data {
        struct Envelope: Encodable {
            let op: String
            let payload: EventUploadAck
        }

        return try JSONEncoder().encode(
            Envelope(
                op: "event_upload_ack",
                payload: EventUploadAck(receivedIds: receivedIds)
            )
        )
    }
}
