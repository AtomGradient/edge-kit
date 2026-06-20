// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import GRDB
import Foundation

public enum V5RetryFields {
    public static let migrationId = "v5_retry_fields"

    public static func register(_ migrator: inout DatabaseMigrator) {
        migrator.registerMigration(migrationId) { db in
            try db.alter(table: "facts") { t in
                t.add(column: "classification_retry_count", .integer)
                    .notNull()
                    .defaults(to: 0)
                t.add(column: "classification_last_attempt_at", .integer)
            }

            try db.create(index: "idx_facts_retry_count",
                          on: "facts",
                          columns: ["classification_retry_count"])
            try db.create(index: "idx_facts_last_attempt_at",
                          on: "facts",
                          columns: ["classification_last_attempt_at"])

            NSLog("[DatabaseMigration] v5_retry_fields: added classification retry columns and indexes")
        }
    }

    /// Verifies columns and indexes created by this migration.
    public static func verifySchema(_ db: Database) throws -> (newColumns: [String], newIndexes: [String]) {
        let expectedColumns = [
            "classification_retry_count",
            "classification_last_attempt_at"
        ]
        let columns = try db.columns(in: "facts")
        let columnNames = Set(columns.map { $0.name })
        var foundColumns: [String] = []
        for name in expectedColumns where columnNames.contains(name) {
            foundColumns.append(name)
        }

        let expectedIndexes = [
            "idx_facts_retry_count",
            "idx_facts_last_attempt_at"
        ]
        let allIndexes = try String.fetchAll(db, sql: "SELECT name FROM sqlite_master WHERE type='index'")
        let indexSet = Set(allIndexes)
        var foundIndexes: [String] = []
        for name in expectedIndexes where indexSet.contains(name) {
            foundIndexes.append(name)
        }

        return (foundColumns, foundIndexes)
    }
}
