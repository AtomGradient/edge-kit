// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import GRDB
import Foundation

public enum V2APrimitiveTables {
    public static let migrationId = "v2a_create_primitive_tables"

    public static func register(_ migrator: inout DatabaseMigrator) {
        migrator.registerMigration(migrationId) { db in
            try db.create(table: "events") { t in
                t.primaryKey("id", .text)
                t.column("ts_ms", .integer).notNull()
                t.column("session_id", .text)
                t.column("namespace", .text).notNull()
                    .defaults(to: Edge.defaultNamespace)
                t.column("user_input", .blob)
                t.column("ai_output", .blob)
                t.column("feedback", .text)
                t.column("feedback_detail", .blob)
                t.column("derived_facts", .text)
                t.column("derived_artifacts", .text)
                t.column("sensitivity", .integer).notNull().defaults(to: 2)
                t.column("created_at", .datetime).notNull()
            }
            try db.create(index: "idx_events_ts", on: "events", columns: ["ts_ms"])
            try db.create(index: "idx_events_session", on: "events", columns: ["session_id"])

            try db.create(table: "facts") { t in
                t.primaryKey("id", .text)
                t.column("ts_ms", .integer).notNull()
                t.column("schema", .text).notNull()
                t.column("namespace", .text).notNull()
                    .defaults(to: Edge.defaultNamespace)
                t.column("payload", .blob).notNull()
                t.column("derived_from_event_id", .text)
                t.column("sensitivity", .integer).notNull().defaults(to: 2)
                t.column("status", .text).notNull().defaults(to: "active")
                t.column("created_at", .datetime).notNull()
            }
            try db.create(index: "idx_facts_schema", on: "facts", columns: ["schema"])
            try db.create(index: "idx_facts_ts", on: "facts", columns: ["ts_ms"])
            try db.create(index: "idx_facts_status", on: "facts", columns: ["status"])

            try db.create(table: "traces") { t in
                t.primaryKey("id", .text)
                t.column("ts_ms", .integer).notNull()
                t.column("namespace", .text).notNull()
                    .defaults(to: Edge.defaultNamespace)
                t.column("action", .text).notNull()
                t.column("target", .text)
                t.column("context", .blob)
                t.column("sensitivity", .integer).notNull().defaults(to: 2)
                t.column("created_at", .datetime).notNull()
            }
            try db.create(index: "idx_traces_action", on: "traces", columns: ["action"])
            try db.create(index: "idx_traces_ts", on: "traces", columns: ["ts_ms"])

            try db.create(table: "artifacts") { t in
                t.primaryKey("content_hash", .text)
                t.column("namespace", .text).notNull()
                    .defaults(to: Edge.defaultNamespace)
                t.column("type", .text).notNull()
                t.column("size_bytes", .integer).notNull()
                t.column("stored_path", .text).notNull()
                t.column("metadata", .blob)
                t.column("sensitivity", .integer).notNull().defaults(to: 2)
                t.column("ref_count", .integer).notNull().defaults(to: 0)
                t.column("created_at", .datetime).notNull()
            }
            try db.create(index: "idx_artifacts_type", on: "artifacts", columns: ["type"])
            try db.create(index: "idx_artifacts_refcount", on: "artifacts", columns: ["ref_count"])

            NSLog("[DatabaseMigration] v2a_create_primitive_tables: created events, facts, traces, and artifacts tables")
        }
    }

    /// Verifies tables and indexes created by this migration.
    public static func verifySchema(_ db: Database) throws -> (tables: [String], indexes: [String]) {
        let expectedTables = ["events", "facts", "traces", "artifacts"]
        var foundTables: [String] = []
        for name in expectedTables {
            if try db.tableExists(name) {
                foundTables.append(name)
            }
        }

        let expectedIndexes = [
            "idx_events_ts", "idx_events_session",
            "idx_facts_schema", "idx_facts_ts", "idx_facts_status",
            "idx_traces_action", "idx_traces_ts",
            "idx_artifacts_type", "idx_artifacts_refcount"
        ]
        var foundIndexes: [String] = []
        let allIndexes = try String.fetchAll(db, sql: "SELECT name FROM sqlite_master WHERE type='index'")
        let indexSet = Set(allIndexes)
        for name in expectedIndexes {
            if indexSet.contains(name) {
                foundIndexes.append(name)
            }
        }

        return (foundTables, foundIndexes)
    }
}
