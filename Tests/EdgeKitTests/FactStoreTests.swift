// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class FactStoreTests: XCTestCase {

    var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("factstore-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    private func makeStore(readOnly: Bool = false) throws -> FactStore {
        let path = tmpDir.appendingPathComponent("data.sqlite3")
        return try FactStore(path: path, readOnly: readOnly)
    }

    private func sampleExpense(
        amount: Double = 50.5,
        merchant: String = "必胜客",
        category: String = "餐饮",
        time: Int64 = 1_742_000_000_000,
        location: String? = "北京",
        description: String? = "午餐"
    ) -> FactRecord {
        var payload: [String: FactValue] = [
            "amount": .double(amount),
            "merchant": .string(merchant),
            "category": .string(category),
            "time": .int(time),
        ]
        if let l = location { payload["location"] = .string(l) }
        if let d = description { payload["description"] = .string(d) }
        return FactRecord.new(schemaName: "finance.expense", payload: payload)
    }

    func test_createSchemaSQL_containsRequiredArtifacts() {
        let sql = FactStore.createSchemaSQL
        XCTAssertTrue(sql.contains("CREATE TABLE IF NOT EXISTS facts"), "facts 表声明缺失")
        XCTAssertTrue(sql.contains("CREATE TABLE IF NOT EXISTS meta"), "meta 表声明缺失")
        for col in ["id TEXT PRIMARY KEY",
                    "schema_name TEXT NOT NULL",
                    "payload_json TEXT NOT NULL",
                    "created_at INTEGER NOT NULL",
                    "sensitivity TEXT NOT NULL DEFAULT 'private'",
                    "ttl_seconds INTEGER",
                    "derived_from TEXT",
                    "source_type TEXT NOT NULL DEFAULT 'user_device'",
                    "idx_amount REAL",
                    "idx_merchant TEXT",
                    "idx_category TEXT",
                    "idx_time INTEGER",
                    "idx_location TEXT"] {
            XCTAssertTrue(sql.contains(col), "列声明缺失: \(col)")
        }
        for idx in ["ix_facts_schema", "ix_facts_time", "ix_facts_merchant", "ix_facts_category"] {
            XCTAssertTrue(sql.contains(idx), "索引声明缺失: \(idx)")
        }
    }

    func test_record_get_roundTrip() throws {
        let store = try makeStore()
        let fact = sampleExpense()
        _ = try store.record(fact)

        let got = try store.get(fact.id)
        XCTAssertNotNil(got)
        XCTAssertEqual(got?.id, fact.id)
        XCTAssertEqual(got?.schemaName, "finance.expense")
        XCTAssertEqual(got?.sourceType, "user_device")
        XCTAssertEqual(got?.payload["amount"]?.doubleValue, 50.5)
        XCTAssertEqual(got?.payload["merchant"]?.stringValue, "必胜客")
        XCTAssertEqual(got?.payload["category"]?.stringValue, "餐饮")
        XCTAssertEqual(got?.payload["time"]?.intValue, 1_742_000_000_000)
    }

    func test_get_missingReturnsNil() throws {
        let store = try makeStore()
        XCTAssertNil(try store.get("nonexistent-id"))
    }

    func test_query_byMerchant() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(merchant: "必胜客", time: 100))
        _ = try store.record(sampleExpense(merchant: "邻几便利", time: 200))
        _ = try store.record(sampleExpense(merchant: "必胜客", time: 300))

        let results = try store.query(.init(merchant: "必胜客"))
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.first?.payload["time"]?.intValue, 300)
    }

    func test_query_byAmountRange() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(amount: 10.0, time: 1))
        _ = try store.record(sampleExpense(amount: 50.0, time: 2))
        _ = try store.record(sampleExpense(amount: 200.0, time: 3))

        let mid = try store.query(.init(amountGte: 30, amountLte: 100))
        XCTAssertEqual(mid.count, 1)
        XCTAssertEqual(mid.first?.payload["amount"]?.doubleValue, 50.0)
    }

    func test_query_byTimeRangeAndCategory() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(category: "餐饮", time: 100))
        _ = try store.record(sampleExpense(category: "交通", time: 200))
        _ = try store.record(sampleExpense(category: "餐饮", time: 300))

        let r = try store.query(.init(category: "餐饮", timeGte: 150))
        XCTAssertEqual(r.count, 1)
        XCTAssertEqual(r.first?.payload["time"]?.intValue, 300)
    }

    func test_query_merchantLike() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(merchant: "邻几便利店", time: 1))
        _ = try store.record(sampleExpense(merchant: "邻家餐厅", time: 2))

        let r = try store.query(.init(merchantLike: "邻几"))
        XCTAssertEqual(r.count, 1)
        XCTAssertEqual(r.first?.payload["merchant"]?.stringValue, "邻几便利店")
    }

    func test_query_limit() throws {
        let store = try makeStore()
        for i in 0..<10 {
            _ = try store.record(sampleExpense(time: Int64(i)))
        }
        let r = try store.query(.init(limit: 3))
        XCTAssertEqual(r.count, 3)
    }

    func test_count_totalAndByScheme() throws {
        let store = try makeStore()
        for _ in 0..<5 {
            _ = try store.record(sampleExpense())
        }
        XCTAssertEqual(try store.count(), 5)
        XCTAssertEqual(try store.count(schema: "finance.expense"), 5)
        XCTAssertEqual(try store.count(schema: "finance.nonexistent"), 0)
    }

    func test_aggregateSum_totalAmount() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(amount: 10.5))
        _ = try store.record(sampleExpense(amount: 20.0))
        _ = try store.record(sampleExpense(amount: 30.0))

        let sum = try store.aggregateSum(schema: "finance.expense")
        XCTAssertEqual(sum, 60.5, accuracy: 1e-9)
    }

    func test_aggregateSum_byCategoryAndTimeRange() throws {
        let store = try makeStore()
        _ = try store.record(sampleExpense(amount: 100, category: "餐饮", time: 100))
        _ = try store.record(sampleExpense(amount: 50, category: "餐饮", time: 200))
        _ = try store.record(sampleExpense(amount: 999, category: "交通", time: 300))

        let sum = try store.aggregateSum(
            schema: "finance.expense",
            category: "餐饮",
            timeGte: 150
        )
        XCTAssertEqual(sum, 50, accuracy: 1e-9)
    }

    func test_aggregateSum_nonNumericFieldThrows() throws {
        let store = try makeStore()
        XCTAssertThrowsError(
            try store.aggregateSum(schema: "finance.expense", field: "merchant")
        ) { error in
            guard case FactStoreError.fieldNotNumeric = error else {
                return XCTFail("expected fieldNotNumeric, got \(error)")
            }
        }
    }

    func test_delete_existingReturnsTrue() throws {
        let store = try makeStore()
        let fact = sampleExpense()
        _ = try store.record(fact)
        XCTAssertTrue(try store.delete(fact.id))
        XCTAssertNil(try store.get(fact.id))
    }

    func test_delete_missingReturnsFalse() throws {
        let store = try makeStore()
        XCTAssertFalse(try store.delete("nonexistent-id"))
    }

    func test_H1_rejectsExternalSourceType() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: [
                "amount": .double(10),
                "time": .int(1),
            ],
            sourceType: "external_doc"
        )
        XCTAssertThrowsError(try store.record(fact)) { error in
            guard case FactStoreError.h1SourceTypeViolation(let got) = error else {
                return XCTFail("expected h1SourceTypeViolation, got \(error)")
            }
            XCTAssertEqual(got, "external_doc")
        }
    }

    func test_H1_rejectsSharedUserSourceType() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: ["amount": .double(10), "time": .int(1)],
            sourceType: "shared_user"
        )
        XCTAssertThrowsError(try store.record(fact))
    }

    func test_schemaValidation_rejectsUnknownField() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: [
                "amount": .double(10),
                "time": .int(1),
                "bogus_field": .string("x"),
            ]
        )
        XCTAssertThrowsError(try store.record(fact)) { error in
            guard case FactSchemaError.unknownFields = error else {
                return XCTFail("expected unknownFields, got \(error)")
            }
        }
    }

    func test_schemaValidation_rejectsMissingRequired() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: [
                "time": .int(1),
            ]
        )
        XCTAssertThrowsError(try store.record(fact)) { error in
            guard case FactSchemaError.requiredFieldNamed(let name) = error else {
                return XCTFail("expected requiredFieldNamed, got \(error)")
            }
            XCTAssertEqual(name, "amount")
        }
    }

    func test_schemaValidation_rejectsWrongType() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: [
                "amount": .string("not a number"),
                "time": .int(1),
            ]
        )
        XCTAssertThrowsError(try store.record(fact)) { error in
            guard case FactSchemaError.fieldTypeMismatch(let field, _, _) = error else {
                return XCTFail("expected fieldTypeMismatch, got \(error)")
            }
            XCTAssertEqual(field, "amount")
        }
    }

    func test_schemaValidation_rejectsUnregisteredSchema() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "unknown.thing",
            payload: ["amount": .double(1)]
        )
        XCTAssertThrowsError(try store.record(fact)) { error in
            guard case FactSchemaError.schemaNotRegistered = error else {
                return XCTFail("expected schemaNotRegistered, got \(error)")
            }
        }
    }

    func test_readOnly_rejectsWrites() throws {
        let path = tmpDir.appendingPathComponent("data.sqlite3")
        let writeStore = try FactStore(path: path)
        _ = try writeStore.record(sampleExpense())
        writeStore.close()

        let ro = try FactStore(path: path, readOnly: true)
        XCTAssertThrowsError(try ro.record(sampleExpense())) { error in
            guard case FactStoreError.readOnlyWrite = error else {
                return XCTFail("expected readOnlyWrite, got \(error)")
            }
        }
        XCTAssertThrowsError(try ro.delete("any")) { error in
            guard case FactStoreError.readOnlyWrite = error else {
                return XCTFail("expected readOnlyWrite, got \(error)")
            }
        }
        let results = try ro.query()
        XCTAssertEqual(results.count, 1)
    }

    func test_readOnly_missingFileThrows() {
        let missing = tmpDir.appendingPathComponent("nonexistent.sqlite3")
        XCTAssertThrowsError(try FactStore(path: missing, readOnly: true)) { error in
            guard case FactStoreError.fileNotFound = error else {
                return XCTFail("expected fileNotFound, got \(error)")
            }
        }
    }

    func test_recordMany_allSucceed() throws {
        let store = try makeStore()
        let facts = (0..<5).map { i in sampleExpense(time: Int64(i)) }
        let (written, failed) = try store.recordMany(facts)
        XCTAssertEqual(written, 5)
        XCTAssertEqual(failed, 0)
        XCTAssertEqual(try store.count(), 5)
    }

    func test_recordMany_oneFailureRollsBackAll() throws {
        let store = try makeStore()
        let good = sampleExpense(time: 100)
        let bad = FactRecord.new(
            schemaName: "finance.expense",
            payload: ["amount": .double(1), "time": .int(2)],
            sourceType: "external_doc"
        )
        let good2 = sampleExpense(time: 300)

        let (written, failed) = try store.recordMany([good, bad, good2])
        XCTAssertEqual(written, 0)
        XCTAssertEqual(failed, 3)
        XCTAssertEqual(try store.count(), 0)
    }

    func test_record_sameIdReplaces() throws {
        let store = try makeStore()
        let fact = sampleExpense(amount: 10)
        _ = try store.record(fact)

        let updated = FactRecord(
            id: fact.id,
            schemaName: fact.schemaName,
            payload: [
                "amount": .double(999),
                "time": .int(999),
            ],
            createdAt: fact.createdAt,
            sensitivity: fact.sensitivity,
            ttlSeconds: fact.ttlSeconds,
            derivedFrom: fact.derivedFrom,
            sourceType: fact.sourceType
        )
        _ = try store.record(updated)

        XCTAssertEqual(try store.count(), 1)
        let got = try store.get(fact.id)
        XCTAssertEqual(got?.payload["amount"]?.doubleValue, 999)
    }

    func test_payloadJSON_roundTrip() throws {
        let store = try makeStore()
        let fact = FactRecord.new(
            schemaName: "finance.expense",
            payload: [
                "amount": .double(50.5),
                "merchant": .string("必胜客"),
                "category": .string("餐饮"),
                "time": .int(1_742_000_000_000),
                "location": .null,
            ]
        )
        _ = try store.record(fact)
        let got = try store.get(fact.id)!
        XCTAssertEqual(got.payload["amount"]?.doubleValue, 50.5)
        XCTAssertEqual(got.payload["merchant"]?.stringValue, "必胜客")
        XCTAssertEqual(got.payload["location"]?.isNull, true)
    }

    func test_indexColumnsExtracted() throws {
        let store = try makeStore()
        let fact = sampleExpense(
            amount: 42.0,
            merchant: "xx",
            category: "cat",
            time: 999,
            location: "loc"
        )
        _ = try store.record(fact)

        XCTAssertEqual(try store.query(.init(amount: 42.0)).count, 1)
        XCTAssertEqual(try store.query(.init(merchant: "xx")).count, 1)
        XCTAssertEqual(try store.query(.init(category: "cat")).count, 1)
        XCTAssertEqual(try store.query(.init(timeGte: 999, timeLte: 999)).count, 1)
        XCTAssertEqual(try store.query(.init(location: "loc")).count, 1)
    }
}
