// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// A single unit of on-device data that may flow into downstream consumers.
///
/// The event carries producer identity, type, opaque payload bytes, and consumer
/// eligibility tags. The collector stores and syncs events without interpreting
/// the payload schema.
public struct DataEvent: Codable, Identifiable, Equatable {

    public let id: UUID
    public let timestamp: Date
    public let appId: String
    public let eventType: String
    public let payload: Data
    public let tags: Set<EventTag>

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        appId: String,
        eventType: String,
        payload: Data,
        tags: Set<EventTag>
    ) {
        self.id = id
        self.timestamp = timestamp
        self.appId = appId
        self.eventType = eventType
        self.payload = payload
        self.tags = tags
    }
}

/// Tags declare which downstream consumer(s) a given event may feed.
///
/// Adding a new consumer? Add a new case here. The `DataEvent` structure is
/// intentionally forward-compatible — unknown tags on the wire are ignored
/// by older consumers but preserved in storage.
public enum EventTag: String, Codable, CaseIterable, Sendable {
    /// Eligible for persona learning data pipeline.
    case trainingSample

    /// Eligible for the episodic fact layer (precise fact retrieval).
    case retrievableFact

    /// Eligible for statistics / aggregation (totals, counts, time-series).
    case aggregatable

    /// A complete conversation turn — implies `trainingSample` typically.
    case conversation

    /// User explicitly corrected the model's output. The strongest supervised signal.
    case userCorrection

    /// User chose one response over another (A/B). Feeds DPO / preference learning.
    case preference

    /// User-supplied domain knowledge (uploaded files, notes).
    case domainKnowledge
}

/// Statistics for monitoring the local collector (surfaced in Settings → EdgeMesh).
public struct DataCollectorStats: Codable, Equatable {
    public let totalEvents: Int
    public let totalBytes: Int
    public let perTypeCounts: [String: Int]
    public let perTagCounts: [String: Int]
    public let oldestTimestamp: Date?
    public let newestTimestamp: Date?

    public init(
        totalEvents: Int,
        totalBytes: Int,
        perTypeCounts: [String: Int],
        perTagCounts: [String: Int],
        oldestTimestamp: Date?,
        newestTimestamp: Date?
    ) {
        self.totalEvents = totalEvents
        self.totalBytes = totalBytes
        self.perTypeCounts = perTypeCounts
        self.perTagCounts = perTagCounts
        self.oldestTimestamp = oldestTimestamp
        self.newestTimestamp = newestTimestamp
    }
}

extension Set where Element == EventTag {
    public func sortedRawValues() -> [String] {
        self.map(\.rawValue).sorted()
    }

    public static func fromRawValues(_ values: [String]) -> Set<EventTag> {
        Set(values.compactMap(EventTag.init(rawValue:)))
    }
}

/// Payload for the `event_upload` wire op.
public struct EventUploadBatch: Codable, Equatable {
    public let events: [DataEvent]
    public init(events: [DataEvent]) { self.events = events }
}

/// Server-side acknowledgement for `event_upload`. Lists which event IDs were
/// successfully persisted (duplicates are still reported, since the client uses
/// them to mark `synced_peers`).
public struct EventUploadAck: Codable, Equatable {
    public let receivedIds: [UUID]
    public init(receivedIds: [UUID]) { self.receivedIds = receivedIds }
}
