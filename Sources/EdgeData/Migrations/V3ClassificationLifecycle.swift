// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import GRDB
import Foundation

public enum V3ClassificationLifecycle {
    public static let migrationId = "v3_classification_lifecycle"

    public static func register(_ migrator: inout DatabaseMigrator) {
        migrator.registerMigration(migrationId) { db in
            try db.alter(table: "facts") { t in
                t.add(column: "classification_confidence", .double)
                t.add(column: "classification_model_ver", .text)
                t.add(column: "classified_at", .integer)
                t.add(column: "classification_corrected_at", .integer)
            }

            try db.create(index: "idx_facts_classification_confidence",
                          on: "facts",
                          columns: ["classification_confidence"])
            try db.create(index: "idx_facts_classified_at",
                          on: "facts",
                          columns: ["classified_at"])

            NSLog("[DatabaseMigration] v3_classification_lifecycle: added classification lifecycle columns and indexes")
        }
    }

    /// Verifies columns and indexes created by this migration.
    public static func verifySchema(_ db: Database) throws -> (newColumns: [String], newIndexes: [String]) {
        let expectedColumns = [
            "classification_confidence",
            "classification_model_ver",
            "classified_at",
            "classification_corrected_at"
        ]
        let columns = try db.columns(in: "facts")
        let columnNames = Set(columns.map { $0.name })
        var foundColumns: [String] = []
        for name in expectedColumns {
            if columnNames.contains(name) {
                foundColumns.append(name)
            }
        }

        let expectedIndexes = [
            "idx_facts_classification_confidence",
            "idx_facts_classified_at"
        ]
        let allIndexes = try String.fetchAll(db, sql: "SELECT name FROM sqlite_master WHERE type='index'")
        let indexSet = Set(allIndexes)
        var foundIndexes: [String] = []
        for name in expectedIndexes {
            if indexSet.contains(name) {
                foundIndexes.append(name)
            }
        }

        return (foundColumns, foundIndexes)
    }
}
