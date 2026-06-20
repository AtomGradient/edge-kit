// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import SQLite3

/// Persistent list of peers this device has paired with.
///
/// Each entry pins a peer by the SHA-256 fingerprint of its self-signed certificate.
/// mTLS handshakes verify that the server's leaf certificate hashes to the stored fingerprint —
/// that's the entire trust model (no CA, no revocation list beyond this table).
///
/// Storage:
/// - iOS: `~/Library/Application Support/<bundleId>/edgemesh_trust.sqlite`
/// - macOS: `~/Library/Application Support/EdgeStudio/edgemesh_trust.sqlite`
///
/// Thread-safety: serialized through an internal `DispatchQueue`.
public final class TrustStore: @unchecked Sendable {

    public struct TrustedPeer: Equatable {
        public let peerId: String
        public let displayName: String
        /// 64-character lowercase hex certificate fingerprint.
        public let fingerprint: String
        /// Peer role: `brain`, `sensor`, or `peer`.
        public let role: String
        public let pairedAt: Date
        public var lastSeenAt: Date?
        public var revoked: Bool

        public init(peerId: String, displayName: String, fingerprint: String, role: String, pairedAt: Date, lastSeenAt: Date?, revoked: Bool) {
            self.peerId = peerId
            self.displayName = displayName
            self.fingerprint = fingerprint
            self.role = role
            self.pairedAt = pairedAt
            self.lastSeenAt = lastSeenAt
            self.revoked = revoked
        }
    }

    private let dbURL: URL
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "com.edgemesh.truststore")

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
            throw MeshError.trustStoreError("sqlite3_open_v2 rc=\(rc)")
        }
        sqlite3_exec(handle, "PRAGMA journal_mode=WAL;", nil, nil, nil)
        sqlite3_exec(handle, "PRAGMA synchronous=FULL;", nil, nil, nil)
        self.db = handle
    }

    private func migrate() throws {
        let ddl = """
        CREATE TABLE IF NOT EXISTS trusted_peers (
            peer_id       TEXT PRIMARY KEY,
            display_name  TEXT NOT NULL,
            fingerprint   TEXT NOT NULL,
            role          TEXT NOT NULL,
            paired_at     INTEGER NOT NULL,
            last_seen_at  INTEGER,
            revoked       INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_trusted_peers_fingerprint ON trusted_peers(fingerprint);
        """
        try exec(ddl)
    }

    /// Insert or replace a trusted peer. Used when a pairing completes successfully.
    public func upsert(_ peer: TrustedPeer) throws {
        try queue.sync {
            let sql = """
            INSERT INTO trusted_peers (peer_id, display_name, fingerprint, role, paired_at, last_seen_at, revoked)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(peer_id) DO UPDATE SET
                display_name = excluded.display_name,
                fingerprint  = excluded.fingerprint,
                role         = excluded.role,
                paired_at    = excluded.paired_at,
                last_seen_at = excluded.last_seen_at,
                revoked      = excluded.revoked;
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare upsert failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, peer.peerId, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 2, peer.displayName, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 3, peer.fingerprint, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 4, peer.role, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(stmt, 5, Int64(peer.pairedAt.timeIntervalSince1970 * 1000))
            if let seen = peer.lastSeenAt {
                sqlite3_bind_int64(stmt, 6, Int64(seen.timeIntervalSince1970 * 1000))
            } else {
                sqlite3_bind_null(stmt, 6)
            }
            sqlite3_bind_int(stmt, 7, peer.revoked ? 1 : 0)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MeshError.trustStoreError("step upsert failed: \(errmsg())")
            }
        }
    }

    public func lookup(peerId: String) throws -> TrustedPeer? {
        try queue.sync {
            let sql = "SELECT peer_id, display_name, fingerprint, role, paired_at, last_seen_at, revoked FROM trusted_peers WHERE peer_id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare lookup failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, peerId, -1, SQLITE_TRANSIENT)
            let rc = sqlite3_step(stmt)
            if rc == SQLITE_ROW {
                return Self.readPeer(from: stmt)
            } else if rc == SQLITE_DONE {
                return nil
            } else {
                throw MeshError.trustStoreError("step lookup failed: \(errmsg())")
            }
        }
    }

    public func lookup(fingerprint: String) throws -> TrustedPeer? {
        try queue.sync {
            let sql = "SELECT peer_id, display_name, fingerprint, role, paired_at, last_seen_at, revoked FROM trusted_peers WHERE fingerprint = ? LIMIT 1;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare lookup_fp failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, fingerprint.lowercased(), -1, SQLITE_TRANSIENT)
            let rc = sqlite3_step(stmt)
            if rc == SQLITE_ROW {
                return Self.readPeer(from: stmt)
            } else if rc == SQLITE_DONE {
                return nil
            } else {
                throw MeshError.trustStoreError("step lookup_fp failed: \(errmsg())")
            }
        }
    }

    public func listAll() throws -> [TrustedPeer] {
        try queue.sync {
            let sql = "SELECT peer_id, display_name, fingerprint, role, paired_at, last_seen_at, revoked FROM trusted_peers ORDER BY paired_at DESC;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare list failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            var results: [TrustedPeer] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                if let p = Self.readPeer(from: stmt) { results.append(p) }
            }
            return results
        }
    }

    public func revoke(peerId: String) throws {
        try queue.sync {
            let sql = "UPDATE trusted_peers SET revoked = 1 WHERE peer_id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare revoke failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, peerId, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MeshError.trustStoreError("step revoke failed: \(errmsg())")
            }
        }
    }

    public func delete(peerId: String) throws {
        try queue.sync {
            let sql = "DELETE FROM trusted_peers WHERE peer_id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare delete failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, peerId, -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw MeshError.trustStoreError("step delete failed: \(errmsg())")
            }
        }
    }

    public func touchLastSeen(peerId: String, at date: Date = Date()) throws {
        try queue.sync {
            let sql = "UPDATE trusted_peers SET last_seen_at = ? WHERE peer_id = ?;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw MeshError.trustStoreError("prepare touch failed: \(errmsg())")
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_int64(stmt, 1, Int64(date.timeIntervalSince1970 * 1000))
            sqlite3_bind_text(stmt, 2, peerId, -1, SQLITE_TRANSIENT)
            sqlite3_step(stmt)
        }
    }

    /// Returns `true` iff the presented fingerprint matches a trusted, non-revoked peer.
    /// Used inside the mTLS `verify_block`.
    public func verify(fingerprint: String) throws -> TrustedPeer? {
        guard let peer = try lookup(fingerprint: fingerprint) else { return nil }
        guard !peer.revoked else { throw MeshError.peerRevoked(peer.peerId) }
        return peer
    }

    private func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(db, sql, nil, nil, &err)
        if rc != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            if let e = err { sqlite3_free(e) }
            throw MeshError.trustStoreError("exec rc=\(rc) \(msg)")
        }
    }

    private func errmsg() -> String {
        guard let msg = sqlite3_errmsg(db) else { return "nil" }
        return String(cString: msg)
    }

    private static func readPeer(from stmt: OpaquePointer?) -> TrustedPeer? {
        guard let idCStr = sqlite3_column_text(stmt, 0),
              let nameCStr = sqlite3_column_text(stmt, 1),
              let fpCStr = sqlite3_column_text(stmt, 2),
              let roleCStr = sqlite3_column_text(stmt, 3)
        else { return nil }

        let pairedMs = sqlite3_column_int64(stmt, 4)
        let lastSeenMs: Int64? = sqlite3_column_type(stmt, 5) == SQLITE_NULL ? nil : sqlite3_column_int64(stmt, 5)
        let revoked = sqlite3_column_int(stmt, 6) != 0

        return TrustedPeer(
            peerId: String(cString: idCStr),
            displayName: String(cString: nameCStr),
            fingerprint: String(cString: fpCStr),
            role: String(cString: roleCStr),
            pairedAt: Date(timeIntervalSince1970: Double(pairedMs) / 1000.0),
            lastSeenAt: lastSeenMs.map { Date(timeIntervalSince1970: Double($0) / 1000.0) },
            revoked: revoked
        )
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(
    OpaquePointer(bitPattern: -1),
    to: sqlite3_destructor_type.self
)

extension TrustStore {
    /// Default trust store URL for this platform / bundle.
    /// Never returns nil: falls back to the temp directory if Application Support is unavailable
    /// (should only happen on simulator edge cases or misconfigured sandboxes).
    public static func defaultURL() -> URL {
        let fm = FileManager.default
        #if os(iOS)
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        let bundleId = Bundle.main.bundleIdentifier ?? "EdgeRuntime"
        return base.appendingPathComponent(bundleId, isDirectory: true)
                   .appendingPathComponent("edgemesh_trust.sqlite")
        #else
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        return base.appendingPathComponent("EdgeStudio", isDirectory: true)
                   .appendingPathComponent("edgemesh_trust.sqlite")
        #endif
    }
}
