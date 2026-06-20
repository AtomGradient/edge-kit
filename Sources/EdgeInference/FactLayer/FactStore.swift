// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(
    OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self
)

/// SQL schema version for the fact table.
public let FACT_TABLE_SCHEMA_VERSION: Int = 1

public enum FactStoreError: Error, Equatable {
    case openFailed(path: String, code: Int32, message: String)
    case readOnlyWrite
    case prepareFailed(sql: String, message: String)
    case stepFailed(sql: String, code: Int32, message: String)
    case execFailed(sql: String, message: String)
    case fileNotFound(path: String)
    case h1SourceTypeViolation(String)
    case fieldNotNumeric(field: String, schema: String)
    case aggregationFieldUnsupported(String)
    case unsupportedAggregationField(String)
}

public final class FactStore: @unchecked Sendable {
    public static let createSchemaSQL: String = """

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                schema_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                sensitivity TEXT NOT NULL DEFAULT 'private',
                ttl_seconds INTEGER,
                derived_from TEXT,
                source_type TEXT NOT NULL DEFAULT 'user_device',

                idx_amount REAL,
                idx_merchant TEXT,
                idx_category TEXT,
                idx_time INTEGER,
                idx_location TEXT
            );

            CREATE INDEX IF NOT EXISTS ix_facts_schema ON facts(schema_name);
            CREATE INDEX IF NOT EXISTS ix_facts_time ON facts(idx_time);
            CREATE INDEX IF NOT EXISTS ix_facts_merchant ON facts(idx_merchant);
            CREATE INDEX IF NOT EXISTS ix_facts_category ON facts(idx_category);

"""

    public let path: URL
    public let isReadOnly: Bool

    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "edge.factstore.serial")
    private let registry: FactSchemaRegistry

    /// Opens or creates a fact store.
    ///
    /// - Parameters:
    ///   - path: SQLite database file path.
    ///   - readOnly: Opens the store in read-only mode.
    ///   - registry: Schema registry used for payload validation.
    public init(
        path: URL,
        readOnly: Bool = false,
        registry: FactSchemaRegistry = .shared
    ) throws {
        self.path = path
        self.isReadOnly = readOnly
        self.registry = registry

        if readOnly {
            guard FileManager.default.fileExists(atPath: path.path) else {
                throw FactStoreError.fileNotFound(path: path.path)
            }
            let uri = "file:\(path.path)?mode=ro"
            var handle: OpaquePointer?
            let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_URI
            let rc = sqlite3_open_v2(uri, &handle, flags, nil)
            guard rc == SQLITE_OK, let handle = handle else {
                let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
                if let handle = handle { sqlite3_close(handle) }
                throw FactStoreError.openFailed(path: path.path, code: rc, message: msg)
            }
            self.db = handle
        } else {
            try FileManager.default.createDirectory(
                at: path.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            var handle: OpaquePointer?
            let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
            let rc = sqlite3_open_v2(path.path, &handle, flags, nil)
            guard rc == SQLITE_OK, let handle = handle else {
                let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
                if let handle = handle { sqlite3_close(handle) }
                throw FactStoreError.openFailed(path: path.path, code: rc, message: msg)
            }
            self.db = handle
        }

        try exec("PRAGMA foreign_keys = OFF")
        try exec("PRAGMA journal_mode = WAL")

        if !readOnly {
            try initSchema()
        }
    }

    deinit {
        if let db = db { sqlite3_close(db) }
    }

    public func close() {
        queue.sync {
            if let db = db {
                sqlite3_close(db)
                self.db = nil
            }
        }
    }

    private func initSchema() throws {
        try exec(Self.createSchemaSQL)
        try queue.sync {
            let sql = "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)"
            var stmt: OpaquePointer?
            guard let db = self.db, sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, "schema_version", -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 2, String(FACT_TABLE_SCHEMA_VERSION), -1, SQLITE_TRANSIENT)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw FactStoreError.stepFailed(sql: sql, code: sqlite3_errcode(db), message: errmsg())
            }
        }
    }

    /// Writes one fact and returns its identifier.
    @discardableResult
    public func record(_ fact: FactRecord) throws -> String {
        if isReadOnly { throw FactStoreError.readOnlyWrite }

        guard ALLOWED_FACT_SOURCE_TYPES.contains(fact.sourceType) else {
            throw FactStoreError.h1SourceTypeViolation(fact.sourceType)
        }

        let schema = try registry.get(fact.schemaName)
        let payloadAny = fact.payload.mapValues { $0.asAnyForValidation ?? NSNull() as Any }
        let payloadForValidation = payloadAny.filter { !($0.value is NSNull) }
        try schema.validatePayload(payloadForValidation)

        let idxAmount = fact.payload["amount"]?.doubleValue
        let idxMerchant = fact.payload["merchant"]?.stringValue
        let idxCategory = fact.payload["category"]?.stringValue
        let idxTime = fact.payload["time"]?.intValue
        let idxLocation = fact.payload["location"]?.stringValue

        let payloadJSON = try fact.payloadAsJSONString()

        let sql = """
            INSERT OR REPLACE INTO facts (
                id, schema_name, payload_json, created_at,
                sensitivity, ttl_seconds, derived_from, source_type,
                idx_amount, idx_merchant, idx_category, idx_time, idx_location
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        try queue.sync {
            guard let db = db else { throw FactStoreError.readOnlyWrite }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, fact.id, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 2, fact.schemaName, -1, SQLITE_TRANSIENT)
            sqlite3_bind_text(stmt, 3, payloadJSON, -1, SQLITE_TRANSIENT)
            sqlite3_bind_int64(stmt, 4, fact.createdAt)
            sqlite3_bind_text(stmt, 5, fact.sensitivity, -1, SQLITE_TRANSIENT)
            bindOptionalInt64(stmt, index: 6, value: fact.ttlSeconds)
            bindOptionalText(stmt, index: 7, value: fact.derivedFrom)
            sqlite3_bind_text(stmt, 8, fact.sourceType, -1, SQLITE_TRANSIENT)
            bindOptionalDouble(stmt, index: 9, value: idxAmount)
            bindOptionalText(stmt, index: 10, value: idxMerchant)
            bindOptionalText(stmt, index: 11, value: idxCategory)
            bindOptionalInt64(stmt, index: 12, value: idxTime)
            bindOptionalText(stmt, index: 13, value: idxLocation)

            let rc = sqlite3_step(stmt)
            guard rc == SQLITE_DONE else {
                throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
            }
        }
        return fact.id
    }

    /// Writes facts atomically and returns `(written, failed)`.
    @discardableResult
    public func recordMany(_ facts: [FactRecord]) throws -> (written: Int, failed: Int) {
        if isReadOnly { throw FactStoreError.readOnlyWrite }
        if facts.isEmpty { return (0, 0) }

        do {
            try exec("BEGIN")
            for f in facts {
                do {
                    _ = try record(f)
                } catch {
                    try? exec("ROLLBACK")
                    return (0, facts.count)
                }
            }
            try exec("COMMIT")
            return (facts.count, 0)
        } catch {
            try? exec("ROLLBACK")
            return (0, facts.count)
        }
    }

    public func get(_ factID: String) throws -> FactRecord? {
        let sql = "SELECT * FROM facts WHERE id = ?"
        return try queue.sync { [self] in
            guard let db = db else { return nil }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, factID, -1, SQLITE_TRANSIENT)
            let rc = sqlite3_step(stmt)
            if rc == SQLITE_ROW {
                return try rowToRecord(stmt)
            } else if rc == SQLITE_DONE {
                return nil
            } else {
                throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
            }
        }
    }

    /// Structured query filter for facts.
    public struct QueryFilter {
        public var schema: String?
        public var amount: Double?
        public var amountGte: Double?
        public var amountLte: Double?
        public var merchant: String?
        public var merchantLike: String?
        public var category: String?
        public var timeGte: Int64?
        public var timeLte: Int64?
        public var location: String?
        public var limit: Int = 100

        public init(
            schema: String? = nil,
            amount: Double? = nil,
            amountGte: Double? = nil,
            amountLte: Double? = nil,
            merchant: String? = nil,
            merchantLike: String? = nil,
            category: String? = nil,
            timeGte: Int64? = nil,
            timeLte: Int64? = nil,
            location: String? = nil,
            limit: Int = 100
        ) {
            self.schema = schema
            self.amount = amount
            self.amountGte = amountGte
            self.amountLte = amountLte
            self.merchant = merchant
            self.merchantLike = merchantLike
            self.category = category
            self.timeGte = timeGte
            self.timeLte = timeLte
            self.location = location
            self.limit = limit
        }
    }

    /// Queries facts with AND-combined filters ordered by descending indexed time.
    public func query(_ filter: QueryFilter = QueryFilter()) throws -> [FactRecord] {
        var conditions: [String] = []
        var binders: [(OpaquePointer?, Int32) -> Void] = []

        func bindText(_ s: String) {
            binders.append { stmt, i in sqlite3_bind_text(stmt, i, s, -1, SQLITE_TRANSIENT) }
        }
        func bindDouble(_ d: Double) {
            binders.append { stmt, i in sqlite3_bind_double(stmt, i, d) }
        }
        func bindInt64(_ v: Int64) {
            binders.append { stmt, i in sqlite3_bind_int64(stmt, i, v) }
        }

        if let s = filter.schema { conditions.append("schema_name = ?"); bindText(s) }
        if let a = filter.amount { conditions.append("idx_amount = ?"); bindDouble(a) }
        if let a = filter.amountGte { conditions.append("idx_amount >= ?"); bindDouble(a) }
        if let a = filter.amountLte { conditions.append("idx_amount <= ?"); bindDouble(a) }
        if let m = filter.merchant { conditions.append("idx_merchant = ?"); bindText(m) }
        if let m = filter.merchantLike { conditions.append("idx_merchant LIKE ?"); bindText("%\(m)%") }
        if let c = filter.category { conditions.append("idx_category = ?"); bindText(c) }
        if let t = filter.timeGte { conditions.append("idx_time >= ?"); bindInt64(t) }
        if let t = filter.timeLte { conditions.append("idx_time <= ?"); bindInt64(t) }
        if let l = filter.location { conditions.append("idx_location = ?"); bindText(l) }

        let whereClause = conditions.isEmpty ? "1" : conditions.joined(separator: " AND ")
        let sql = "SELECT * FROM facts WHERE \(whereClause) ORDER BY idx_time DESC LIMIT ?"
        bindInt64(Int64(filter.limit))

        return try queue.sync { [self] in
            guard let db = db else { return [] }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            for (offset, binder) in binders.enumerated() {
                binder(stmt, Int32(offset + 1))
            }
            var out: [FactRecord] = []
            while true {
                let rc = sqlite3_step(stmt)
                if rc == SQLITE_ROW {
                    out.append(try rowToRecord(stmt))
                } else if rc == SQLITE_DONE {
                    break
                } else {
                    throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
                }
            }
            return out
        }
    }

    public func count(schema: String? = nil) throws -> Int {
        let sql: String
        if schema != nil {
            sql = "SELECT COUNT(*) FROM facts WHERE schema_name = ?"
        } else {
            sql = "SELECT COUNT(*) FROM facts"
        }
        return try queue.sync { [self] in
            guard let db = db else { return 0 }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            if let schema = schema {
                sqlite3_bind_text(stmt, 1, schema, -1, SQLITE_TRANSIENT)
            }
            let rc = sqlite3_step(stmt)
            guard rc == SQLITE_ROW else {
                throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
            }
            return Int(sqlite3_column_int64(stmt, 0))
        }
    }

    /// Aggregates a numeric field by sum.
    public func aggregateSum(
        schema: String,
        field: String = "amount",
        category: String? = nil,
        merchant: String? = nil,
        merchantLike: String? = nil,
        timeGte: Int64? = nil,
        timeLte: Int64? = nil
    ) throws -> Double {
        let schemaDef = try registry.get(schema)
        guard let fd = schemaDef.fields[field], (fd.type == .int || fd.type == .float) else {
            throw FactStoreError.fieldNotNumeric(field: field, schema: schema)
        }

        let colMap: [String: String] = ["amount": "idx_amount"]
        guard let col = colMap[field] else {
            throw FactStoreError.unsupportedAggregationField(field)
        }

        var conditions: [String] = ["schema_name = ?"]
        var binders: [(OpaquePointer?, Int32) -> Void] = [
            { stmt, i in sqlite3_bind_text(stmt, i, schema, -1, SQLITE_TRANSIENT) }
        ]
        if let c = category {
            conditions.append("idx_category = ?")
            binders.append { stmt, i in sqlite3_bind_text(stmt, i, c, -1, SQLITE_TRANSIENT) }
        }
        if let m = merchant {
            conditions.append("idx_merchant = ?")
            binders.append { stmt, i in sqlite3_bind_text(stmt, i, m, -1, SQLITE_TRANSIENT) }
        }
        if let m = merchantLike {
            conditions.append("idx_merchant LIKE ?")
            let like = "%\(m)%"
            binders.append { stmt, i in sqlite3_bind_text(stmt, i, like, -1, SQLITE_TRANSIENT) }
        }
        if let t = timeGte {
            conditions.append("idx_time >= ?")
            binders.append { stmt, i in sqlite3_bind_int64(stmt, i, t) }
        }
        if let t = timeLte {
            conditions.append("idx_time <= ?")
            binders.append { stmt, i in sqlite3_bind_int64(stmt, i, t) }
        }

        let sql = "SELECT COALESCE(SUM(\(col)), 0) FROM facts WHERE \(conditions.joined(separator: " AND "))"

        return try queue.sync { [self] in
            guard let db = db else { return 0 }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            for (offset, binder) in binders.enumerated() {
                binder(stmt, Int32(offset + 1))
            }
            let rc = sqlite3_step(stmt)
            guard rc == SQLITE_ROW else {
                throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
            }
            return sqlite3_column_double(stmt, 0)
        }
    }

    @discardableResult
    public func delete(_ factID: String) throws -> Bool {
        if isReadOnly { throw FactStoreError.readOnlyWrite }
        let sql = "DELETE FROM facts WHERE id = ?"
        return try queue.sync { [self] in
            guard let db = db else { return false }
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw FactStoreError.prepareFailed(sql: sql, message: errmsg())
            }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, factID, -1, SQLITE_TRANSIENT)
            let rc = sqlite3_step(stmt)
            guard rc == SQLITE_DONE else {
                throw FactStoreError.stepFailed(sql: sql, code: rc, message: errmsg())
            }
            return sqlite3_changes(db) > 0
        }
    }

    private func exec(_ sql: String) throws {
        try queue.sync {
            guard let db = db else { return }
            var err: UnsafeMutablePointer<CChar>?
            let rc = sqlite3_exec(db, sql, nil, nil, &err)
            if rc != SQLITE_OK {
                let msg = err.flatMap { String(cString: $0) } ?? "unknown"
                if let err = err { sqlite3_free(err) }
                throw FactStoreError.execFailed(sql: sql, message: msg)
            }
        }
    }

    private func errmsg() -> String {
        guard let db = db else { return "no db" }
        return String(cString: sqlite3_errmsg(db))
    }

    private func rowToRecord(_ stmt: OpaquePointer?) throws -> FactRecord {
        let id = String(cString: sqlite3_column_text(stmt, 0))
        let schemaName = String(cString: sqlite3_column_text(stmt, 1))
        let payloadJSON = String(cString: sqlite3_column_text(stmt, 2))
        let createdAt = sqlite3_column_int64(stmt, 3)
        let sensitivity = String(cString: sqlite3_column_text(stmt, 4))
        let ttlSeconds: Int64? = sqlite3_column_type(stmt, 5) == SQLITE_NULL
            ? nil : sqlite3_column_int64(stmt, 5)
        let derivedFrom: String? = sqlite3_column_type(stmt, 6) == SQLITE_NULL
            ? nil : String(cString: sqlite3_column_text(stmt, 6))
        let sourceType = String(cString: sqlite3_column_text(stmt, 7))

        let payload = try FactRecord.parsePayloadJSON(payloadJSON)
        return FactRecord(
            id: id,
            schemaName: schemaName,
            payload: payload,
            createdAt: createdAt,
            sensitivity: sensitivity,
            ttlSeconds: ttlSeconds,
            derivedFrom: derivedFrom,
            sourceType: sourceType
        )
    }
}

private func bindOptionalText(_ stmt: OpaquePointer?, index: Int32, value: String?) {
    if let v = value {
        sqlite3_bind_text(stmt, index, v, -1, SQLITE_TRANSIENT)
    } else {
        sqlite3_bind_null(stmt, index)
    }
}

private func bindOptionalInt64(_ stmt: OpaquePointer?, index: Int32, value: Int64?) {
    if let v = value {
        sqlite3_bind_int64(stmt, index, v)
    } else {
        sqlite3_bind_null(stmt, index)
    }
}

private func bindOptionalDouble(_ stmt: OpaquePointer?, index: Int32, value: Double?) {
    if let v = value {
        sqlite3_bind_double(stmt, index, v)
    } else {
        sqlite3_bind_null(stmt, index)
    }
}
