// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import SQLite3

/// SQLite-backed append-only store for `DataEvent`s.
///
/// Schema is intentionally simple — tags are stored as a JSON array of raw
/// strings, with `LIKE '%tag%'` queries for filtering. If scale demands it
/// later, FTS5 or per-tag indexed tables can be added without API changes.
///
/// Sync state: each row tracks which peers have successfully received this
/// event via `synced_peers` (JSON array). `flush(to peer:)` reads events not
/// yet sent to that peer, uploads them, then records the ack.
///
/// Thread-safety: serialized through an internal queue. Safe to share across
/// async contexts and SwiftUI views.
public final class EventStore: @unchecked Sendable {

    private let dbURL: URL
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "com.edgemesh.eventstore")

    public init(url: URL) throws {
        self.dbURL = url
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try open()
        try migrate()
    }

    deinit {
        if let handle = db {
            sqlite3_close_v2(handle)
        }
    }

    private func open() throws {
        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let rc = sqlite3_open_v2(dbURL.path, &handle, flags, nil)
        guard rc == SQLITE_OK, let handle = handle else {
            if let h = handle { sqlite3_close_v2(h) }
            throw MeshError.trustStoreError("EventStore: sqlite3_open_v2 rc=\(rc)")
        }
        sqlite3_exec(handle, "PRAGMA journal_mode=WAL;", nil, nil, nil)
        sqlite3_exec(handle, "PRAGMA synchronous=FULL;", nil, nil, nil)
        self.db = handle
    }

    private func migrate() throws {
        let ddl = """
        CREATE TABLE IF NOT EXISTS events (
            id            TEXT PRIMARY KEY,
            timestamp     INTEGER NOT NULL,
            app_id        TEXT NOT NULL,
            event_type    TEXT NOT NULL,
            payload       BLOB NOT NULL,
            tags          TEXT NOT NULL,
            synced_peers  TEXT NOT NULL DEFAULT '[]',
            created_at    INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_app ON events(app_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        """
        try exec(ddl)
    }

    /// Insert a new event. Idempotent by `id` (INSERT OR IGNORE) so duplicate
    /// sync attempts are safe.
    public func insert(_ event: DataEvent) throws {
        try queue.sync {
            let sql = """
            INSERT OR IGNORE INTO events
                (id, timestamp, app_id, event_type, payload, tags, synced_peers, created_at)
            VALUES (?, ?, ?, ?, ?, ?, '[]', ?);
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare insert failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, event.id.uuidString, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(stmt, 2, Int64(event.timestamp.timeIntervalSince1970 * 1000))
            sqlite3_bind_text(stmt, 3, event.appId, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 4, event.eventType, -1, SQLITE_TRANSIENT)
            event.payload.withUnsafeBytes { raw in
                _ = sqlite3_bind_blob(stmt, 5, raw.baseAddress, Int32(event.payload.count), SQLITE_TRANSIENT)
            }
            let tagsJSON = try? JSONEncoder().encode(event.tags.sortedRawValues())
            let tagsStr = tagsJSON.flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
            sqlite3_bind_text(stmt, 6, tagsStr, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(stmt, 7, Int64(Date().timeIntervalSince1970 * 1000))
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MeshError.trustStoreError("EventStore: step insert failed: \(errmsg())")
            }
        }
    }

    /// Bulk insert used on the server side when receiving an upload batch.
    /// Wraps inserts in a single transaction for throughput.
    public func insertBatch(_ events: [DataEvent]) throws -> [UUID] {
        var received: [UUID] = []
        try queue.sync {
            try exec("BEGIN IMMEDIATE;")
            do {
                for event in events {
                    try insertInternal(event)
                    received.append(event.id)
                }
                try exec("COMMIT;")
            } catch {
                try? exec("ROLLBACK;")
                throw error
            }
        }
        return received
    }

    private func insertInternal(_ event: DataEvent) throws {
        let sql = """
        INSERT OR IGNORE INTO events
            (id, timestamp, app_id, event_type, payload, tags, synced_peers, created_at)
        VALUES (?, ?, ?, ?, ?, ?, '[]', ?);
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw MeshError.trustStoreError("EventStore: prepare insertInternal failed: \(errmsg())")
        }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, event.id.uuidString, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(stmt, 2, Int64(event.timestamp.timeIntervalSince1970 * 1000))
        sqlite3_bind_text(stmt, 3, event.appId, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(stmt, 4, event.eventType, -1, SQLITE_TRANSIENT)
        event.payload.withUnsafeBytes { raw in
            _ = sqlite3_bind_blob(stmt, 5, raw.baseAddress, Int32(event.payload.count), SQLITE_TRANSIENT)
        }
        let tagsJSON = try? JSONEncoder().encode(event.tags.sortedRawValues())
        let tagsStr = tagsJSON.flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
        sqlite3_bind_text(stmt, 6, tagsStr, -1, SQLITE_TRANSIENT)
        sqlite3_bind_int64(stmt, 7, Int64(Date().timeIntervalSince1970 * 1000))
        guard sqlite3_step(stmt) == SQLITE_DONE else {
            throw MeshError.trustStoreError("EventStore: step insertInternal failed: \(errmsg())")
        }
    }

    /// Query events. `tags == nil` means "any tag"; otherwise events that carry
    /// at least one of the requested tags are returned. `since` is exclusive.
    public func query(
        tags: Set<EventTag>? = nil,
        appId: String? = nil,
        since: Date? = nil,
        limit: Int = 1000
    ) throws -> [DataEvent] {
        try queue.sync {
            var sql = "SELECT id, timestamp, app_id, event_type, payload, tags FROM events WHERE 1=1"
            var params: [(Int32) -> Int32] = []

            if let appId = appId {
                sql += " AND app_id = ?"
                params.append { stmt in
                    sqlite3_bind_text(OpaquePointer(bitPattern: 0), stmt, appId, -1, SQLITE_TRANSIENT)
                }
            }
            if let tags = tags, !tags.isEmpty {
                let tagClauses = tags.map { _ in "tags LIKE ?" }.joined(separator: " OR ")
                sql += " AND (\(tagClauses))"
            }
            if let since = since {
                sql += " AND timestamp > ?"
                _ = since
            }
            sql += " ORDER BY timestamp ASC LIMIT ?;"

            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare query failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }

            var idx: Int32 = 1
            if let appId = appId {
                sqlite3_bind_text(stmt, idx, appId, -1, SQLITE_TRANSIENT); idx += 1
            }
            if let tags = tags, !tags.isEmpty {
                for tag in tags {
                    let pattern = "%\"\(tag.rawValue)\"%"
                    sqlite3_bind_text(stmt, idx, pattern, -1, SQLITE_TRANSIENT)
                    idx += 1
                }
            }
            if let since = since {
                sqlite3_bind_int64(stmt, idx, Int64(since.timeIntervalSince1970 * 1000))
                idx += 1
            }
            sqlite3_bind_int(stmt, idx, Int32(limit))
            _ = params

            var out: [DataEvent] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let event = Self.readEvent(from: stmt) {
                    out.append(event)
                }
            }
            return out
        }
    }

    public func count() throws -> Int {
        try queue.sync {
            let sql = "SELECT COUNT(*) FROM events;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare count failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
            return Int(sqlite3_column_int64(stmt, 0))
        }
    }

    public func stats() throws -> DataCollectorStats {
        try queue.sync {
            var total = 0
            var totalBytes = 0
            var oldest: Date?
            var newest: Date?
            var perType: [String: Int] = [:]
            var perTag: [String: Int] = [:]

            let sql = "SELECT timestamp, event_type, tags, LENGTH(payload) FROM events;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare stats failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                total += 1
                let ts = Date(timeIntervalSince1970: Double(sqlite3_column_int64(stmt, 0)) / 1000.0)
                if let o = oldest { oldest = min(o, ts) } else { oldest = ts }
                if let n = newest { newest = max(n, ts) } else { newest = ts }
                if let typeCStr = sqlite3_column_text(stmt, 1) {
                    let type = String(cString: typeCStr)
                    perType[type, default: 0] += 1
                }
                if let tagsCStr = sqlite3_column_text(stmt, 2) {
                    let str = String(cString: tagsCStr)
                    if let data = str.data(using: .utf8),
                       let arr = try? JSONDecoder().decode([String].self, from: data) {
                        for tag in arr { perTag[tag, default: 0] += 1 }
                    }
                }
                totalBytes += Int(sqlite3_column_int(stmt, 3))
            }
            return DataCollectorStats(
                totalEvents: total,
                totalBytes: totalBytes,
                perTypeCounts: perType,
                perTagCounts: perTag,
                oldestTimestamp: oldest,
                newestTimestamp: newest
            )
        }
    }

    /// Return events not yet acknowledged by the given peer.
    public func unsyncedEvents(for peerId: String, limit: Int = 100) throws -> [DataEvent] {
        try queue.sync {
            let sql = "SELECT id, timestamp, app_id, event_type, payload, tags FROM events WHERE synced_peers NOT LIKE ? ORDER BY timestamp ASC LIMIT ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare unsynced failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            let pattern = "%\"\(peerId)\"%"
            sqlite3_bind_text(stmt, 1, pattern, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int(stmt, 2, Int32(limit))

            var out: [DataEvent] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let event = Self.readEvent(from: stmt) {
                    out.append(event)
                }
            }
            return out
        }
    }

    /// Mark events as synced to a peer after the server acks them.
    public func markSynced(eventIds: [UUID], peerId: String) throws {
        guard !eventIds.isEmpty else { return }
        try queue.sync {
            try exec("BEGIN IMMEDIATE;")
            do {
                for eid in eventIds {
                    let sql = """
                    UPDATE events
                    SET synced_peers = json_insert(synced_peers, '$[#]', ?)
                    WHERE id = ? AND synced_peers NOT LIKE ?;
                    """
                    var stmt: OpaquePointer?
                    guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                        throw MeshError.trustStoreError("EventStore: prepare markSynced failed: \(errmsg())")
                    }
                    defer { sqlite3_finalize(stmt) }
                    let pattern = "%\"\(peerId)\"%"
                    sqlite3_bind_text(stmt, 1, peerId, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_text(stmt, 2, eid.uuidString, -1, SQLITE_TRANSIENT)
                    sqlite3_bind_text(stmt, 3, pattern, -1, SQLITE_TRANSIENT)
                    guard sqlite3_step(stmt) == SQLITE_DONE else {
                        throw MeshError.trustStoreError("EventStore: step markSynced failed: \(errmsg())")
                    }
                }
                try exec("COMMIT;")
            } catch {
                try? exec("ROLLBACK;")
                throw error
            }
        }
    }

    /// Delete events older than `cutoff`. Returns number of rows deleted.
    @discardableResult
    public func purge(olderThan cutoff: Date) throws -> Int {
        try queue.sync {
            let sql = "DELETE FROM events WHERE timestamp < ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("EventStore: prepare purge failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int64(stmt, 1, Int64(cutoff.timeIntervalSince1970 * 1000))
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MeshError.trustStoreError("EventStore: step purge failed: \(errmsg())")
            }
            return Int(sqlite3_changes(db))
        }
    }

    public func deleteAll() throws {
        try queue.sync { try exec("DELETE FROM events;") }
    }

    private func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            if let e = err { sqlite3_free(e) }
            throw MeshError.trustStoreError("EventStore: exec rc=\(rc) \(msg)")
        }
    }

    private func errmsg() -> String {
        guard let msg = sqlite3_errmsg(db) else { return "nil" }
        return String(cString: msg)
    }

    private static func readEvent(from stmt: OpaquePointer?) -> DataEvent? {
        guard let idCStr = sqlite3_column_text(stmt, 0),
              let appCStr = sqlite3_column_text(stmt, 2),
              let typeCStr = sqlite3_column_text(stmt, 3),
              let tagsCStr = sqlite3_column_text(stmt, 5)
        else { return nil }
        guard let uuid = UUID(uuidString: String(cString: idCStr)) else { return nil }

        let tsMs = sqlite3_column_int64(stmt, 1)
        let payloadBytes = sqlite3_column_bytes(stmt, 4)
        let payloadPtr = sqlite3_column_blob(stmt, 4)
        let payload: Data = {
            guard payloadBytes > 0, let ptr = payloadPtr else { return Data() }
            return Data(bytes: ptr, count: Int(payloadBytes))
        }()

        let tagsStr = String(cString: tagsCStr)
        let tags: Set<EventTag> = {
            guard let data = tagsStr.data(using: .utf8),
                  let arr = try? JSONDecoder().decode([String].self, from: data)
            else { return [] }
            return Set.fromRawValues(arr)
        }()

        return DataEvent(
            id: uuid,
            timestamp: Date(timeIntervalSince1970: Double(tsMs) / 1000.0),
            appId: String(cString: appCStr),
            eventType: String(cString: typeCStr),
            payload: payload,
            tags: tags
        )
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(
    OpaquePointer(bitPattern: -1),
    to: sqlite3_destructor_type.self
)

extension EventStore {
    public static func defaultURL() -> URL {
        let fm = FileManager.default
        #if os(iOS)
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        let bundleId = Bundle.main.bundleIdentifier ?? "EdgeRuntime"
        return base.appendingPathComponent(bundleId, isDirectory: true)
                   .appendingPathComponent("edgemesh_events.sqlite")
        #else
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        return base.appendingPathComponent("EdgeStudio", isDirectory: true)
                   .appendingPathComponent("edgemesh_events.sqlite")
        #endif
    }
}
