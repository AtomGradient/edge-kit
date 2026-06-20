// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// High-level append-and-sync API for on-device data collection.
///
/// Apps interact with this type. It's a thin facade over `EventStore` plus the
/// optional `MeshEngine` wiring for uploading batches to a paired brain.
///
/// ```swift
/// let collector = DataCollector(store: try EventStore(url: EventStore.defaultURL()))
/// try collector.append(DataEvent(
///     appId: "com.example.app",
///     eventType: "transaction",
///     payload: try JSONEncoder().encode(tx),
///     tags: [.trainingSample, .retrievableFact, .aggregatable]
/// ))
///
/// let uploaded = try await collector.flush(to: brainPeerId, via: meshConnection)
/// ```
@available(iOS 17.0, macOS 14.0, *)
public final class DataCollector: @unchecked Sendable {

    private let store: EventStore

    public init(store: EventStore) {
        self.store = store
    }

    /// Persist a new event. Safe to call from any thread / actor.
    public func append(_ event: DataEvent) throws {
        try store.insert(event)
    }

    /// Convenience: construct + append in one call.
    public func appendEvent(
        appId: String,
        eventType: String,
        payload: Data,
        tags: Set<EventTag>,
        timestamp: Date = Date()
    ) throws -> DataEvent {
        let event = DataEvent(
            id: UUID(),
            timestamp: timestamp,
            appId: appId,
            eventType: eventType,
            payload: payload,
            tags: tags
        )
        try store.insert(event)
        return event
    }

    public func query(tags: Set<EventTag>? = nil, appId: String? = nil, since: Date? = nil, limit: Int = 1000) throws -> [DataEvent] {
        try store.query(tags: tags, appId: appId, since: since, limit: limit)
    }

    public func count() throws -> Int { try store.count() }
    public func stats() throws -> DataCollectorStats { try store.stats() }

    /// Drain up to `batchSize` unsynced events to the given peer via the supplied
    /// `send` closure (typically `MeshConnection.sendJSON`). On success, mark
    /// those events as acknowledged. Returns the number of events uploaded.
    ///
    /// The caller provides the send and ack-wait functions so this type stays
    /// decoupled from the concrete `MeshConnection`. The brain is expected to
    /// respond with `EventUploadAck` in the same session.
    public func flush(
        to peerId: String,
        batchSize: Int = 100,
        send: (EventUploadBatch) async throws -> EventUploadAck
    ) async throws -> Int {
        let pending = try store.unsyncedEvents(for: peerId, limit: batchSize)
        guard !pending.isEmpty else { return 0 }
        let batch = EventUploadBatch(events: pending)
        let ack = try await send(batch)
        try store.markSynced(eventIds: ack.receivedIds, peerId: peerId)
        return ack.receivedIds.count
    }

    @discardableResult
    public func purge(olderThan cutoff: Date) throws -> Int {
        try store.purge(olderThan: cutoff)
    }

    public func deleteAll() throws { try store.deleteAll() }
}

/// Minimal chat message type used by training-pair convenience helpers.
public struct ChatMessage: Codable, Equatable {
    /// Chat role, such as `system`, `user`, or `assistant`.
    public let role: String
    public let content: String
    public let timestamp: Date

    public init(role: String, content: String, timestamp: Date = Date()) {
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}

/// Training-pair payload stored as `DataEvent.payload`.
public struct TrainingPair: Codable, Equatable {
    public let systemPrompt: String?
    public let messages: [ChatMessage]
    public let preferredResponse: String?
    public let rejectedResponse: String?
    public let qualityScore: Double?
    public let userTags: [String]

    public init(
        systemPrompt: String?,
        messages: [ChatMessage],
        preferredResponse: String? = nil,
        rejectedResponse: String? = nil,
        qualityScore: Double? = nil,
        userTags: [String] = []
    ) {
        self.systemPrompt = systemPrompt
        self.messages = messages
        self.preferredResponse = preferredResponse
        self.rejectedResponse = rejectedResponse
        self.qualityScore = qualityScore
        self.userTags = userTags
    }
}

@available(iOS 17.0, macOS 14.0, *)
extension DataCollector {

    /// Persist a finished conversation as a training sample.
    @discardableResult
    public func collectConversation(
        messages: [ChatMessage],
        systemPrompt: String?,
        userTags: [String] = [],
        appId: String? = nil
    ) throws -> DataEvent {
        let pair = TrainingPair(
            systemPrompt: systemPrompt,
            messages: messages,
            preferredResponse: nil,
            rejectedResponse: nil,
            qualityScore: nil,
            userTags: userTags
        )
        let payload = try JSONEncoder().encode(pair)
        return try appendEvent(
            appId: appId ?? (Bundle.main.bundleIdentifier ?? "unknown"),
            eventType: "conversation",
            payload: payload,
            tags: [.trainingSample, .conversation]
        )
    }

    /// Persist a user correction as a strong supervised signal.
    @discardableResult
    public func collectCorrection(
        context: [ChatMessage],
        original: String,
        corrected: String,
        systemPrompt: String? = nil,
        appId: String? = nil
    ) throws -> DataEvent {
        let pair = TrainingPair(
            systemPrompt: systemPrompt,
            messages: context,
            preferredResponse: corrected,
            rejectedResponse: original,
            qualityScore: 1.0,
            userTags: []
        )
        let payload = try JSONEncoder().encode(pair)
        return try appendEvent(
            appId: appId ?? (Bundle.main.bundleIdentifier ?? "unknown"),
            eventType: "user_correction",
            payload: payload,
            tags: [.trainingSample, .userCorrection, .preference]
        )
    }

    /// Persist an A/B preference.
    @discardableResult
    public func collectPreference(
        context: [ChatMessage],
        preferred: String,
        rejected: String,
        systemPrompt: String? = nil,
        appId: String? = nil
    ) throws -> DataEvent {
        let pair = TrainingPair(
            systemPrompt: systemPrompt,
            messages: context,
            preferredResponse: preferred,
            rejectedResponse: rejected,
            qualityScore: nil,
            userTags: []
        )
        let payload = try JSONEncoder().encode(pair)
        return try appendEvent(
            appId: appId ?? (Bundle.main.bundleIdentifier ?? "unknown"),
            eventType: "preference",
            payload: payload,
            tags: [.trainingSample, .preference]
        )
    }
}
