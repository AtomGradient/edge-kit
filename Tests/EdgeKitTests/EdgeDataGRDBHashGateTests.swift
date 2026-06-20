// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation
import GRDB
import XCTest
@testable import EdgeData

final class EdgeDataGRDBHashGateTests: XCTestCase {
    private static let fixtureFactID = "edge-hash-fixture-001"
    private static let expectedPayloadJSON = #"{"amount":12.5,"empty_string":"","floats_edge":{"exact_eighth":0.125,"exact_half":0.5,"literal_int_in_float_slot":1.0,"negative_zero":-0.0,"nonexact_tenth":0.1,"nonexact_third":0.3333333333333333,"scientific_huge":1e+20,"scientific_tiny":1e-100},"is_business":false,"large_numbers":{"int64_max":9223372036854775807,"int64_min":-9223372036854775808},"line_items":[{"name":"美式","price":9.5,"qty":1},{"name":"三明治","price":12,"qty":1}],"merchant":"邻几便利店","metadata":{"category":"餐饮","confidence":0.875,"empty":{},"source":"edge"},"note":"午餐 coffee","optional":null,"sorted_unicode_keys":{"éclair":1,"Δelta":2,"咖啡":3,"𠮷":4},"tags":["午餐","coffee","edge"],"text_escape":"line\nquote\"slash/backslash\\tab\tbackspace\bformfeed\fcarriage\r","time":1742000000000,"unicode":"多语种Δ咖啡𠮷"}"#
    private static let expectedPayloadSHA256 = "f541972c2cb13e86acbd17e7b190a898d96e99e166b522bded90206a514d7fac"
    private static let expectedReceiptSHA256 = "a8a34659a6ed0819a3e248df9708cbd09ee3a060b36893972c6e691de191abde"
    private static let expectedManifestJSON = #"{"empty_files":[],"encoder":{"base_model_id":"qwen3.5-4b-base","hidden_size":2048,"kind":"base_model_last_hidden","layer_index":-1,"pooling":"mean_excluding_special","tokenizer_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"fallback_chain":["matrix","evidence_matcher","base_router"],"intent_vocab":["base_chat","exact_fact","aggregate_fact","app_action","user_profile","mixed"],"matrices":{"confidence":{"dtype":"float32","file":"route_confidence_matrix.safetensors","shape":[2048,3],"tensor":"confidence_weights"},"intent":{"bias_tensor":"intent_bias","dtype":"float16","file":"route_intent_matrix.safetensors","shape":[2048,6],"tensor":"intent_weights"}},"min_runtime_version":"0.9.0","router_type":"matrix_v0","schema_version":"edgestudio.route_router_manifest.v0","thresholds":{"disabled":null,"exact_match":0.1,"shadow_accept":0.3333333333333333},"training_run_id":"rrr-fixture-001"}"#
    private static let expectedManifestSHA256 = "fa68b212dfbdd78f7d3a38255a68794f579be572de2cb341f02e5484c7983e41"

    var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("edgedata-grdb-hash-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    func testGRDBRecordRawPayloadMatchesPythonCanonicalHash() throws {
        let dbQueue = try makeMigratedDatabase()
        Edge.bootstrap(dbQueue: dbQueue)

        let factID = try Edge.recordRaw(
            fact: RawFact(
                namespace: "com.atomgradient.hashgate:default",
                rawPayload: [
                    "amount": 12.5,
                    "empty_string": "",
                    "floats_edge": [
                        "exact_eighth": 0.125,
                        "exact_half": 0.5,
                        "literal_int_in_float_slot": 1.0,
                        "negative_zero": -0.0,
                        "nonexact_tenth": 0.1,
                        "nonexact_third": 1.0 / 3.0,
                        "scientific_huge": 1e20,
                        "scientific_tiny": 1e-100,
                    ] as [String: Any],
                    "merchant": "邻几便利店",
                    "time": 1_742_000_000_000,
                    "note": "午餐 coffee",
                    "is_business": false,
                    "line_items": [
                        [
                            "name": "美式",
                            "price": 9.5,
                            "qty": 1,
                        ] as [String: Any],
                        [
                            "name": "三明治",
                            "price": 12,
                            "qty": 1,
                        ] as [String: Any],
                    ],
                    "large_numbers": [
                        "int64_max": Int64.max,
                        "int64_min": Int64.min,
                    ] as [String: Any],
                    "metadata": [
                        "category": "餐饮",
                        "confidence": 0.875,
                        "empty": [String: Any](),
                        "source": "edge",
                    ] as [String: Any],
                    "optional": NSNull(),
                    "sorted_unicode_keys": [
                        "éclair": 1,
                        "Δelta": 2,
                        "咖啡": 3,
                        "𠮷": 4,
                    ] as [String: Any],
                    "tags": ["午餐", "coffee", "edge"],
                    "text_escape": "line\nquote\"slash/backslash\\tab\tbackspace\u{08}formfeed\u{0C}carriage\r",
                    "unicode": "多语种Δ咖啡𠮷",
                ],
                candidateSchemas: ["finance.expense"],
                sensitivity: .meshOk
            ),
            customFactId: Self.fixtureFactID
        )

        XCTAssertEqual(factID, Self.fixtureFactID)

        let snapshot = try dbQueue.read { db -> Snapshot in
            guard let row = try Row.fetchOne(
                db,
                sql: """
                SELECT id, schema, namespace, payload, sensitivity, status
                FROM facts
                WHERE id = ?
                """,
                arguments: [Self.fixtureFactID]
            ) else {
                throw HashGateError.missingFixtureRow
            }

            let payload: Data = row["payload"]
            return Snapshot(
                id: row["id"],
                schema: row["schema"],
                namespace: row["namespace"],
                payload: payload,
                sensitivity: row["sensitivity"],
                status: row["status"]
            )
        }

        let payloadJSON = try XCTUnwrap(String(data: snapshot.payload, encoding: .utf8))
        XCTAssertEqual(payloadJSON, Self.expectedPayloadJSON)
        XCTAssertEqual(sha256Hex(snapshot.payload), Self.expectedPayloadSHA256)
        XCTAssertEqual(snapshot.schema, "raw_hashgate_record")
        XCTAssertEqual(snapshot.namespace, "com.atomgradient.hashgate:default")
        XCTAssertEqual(snapshot.sensitivity, Sensitivity.meshOk.rawValue)
        XCTAssertEqual(snapshot.status, FactStatus.rawUnclassified.rawValue)

        let receipt: [String: Any] = [
            "id": snapshot.id,
            "namespace": snapshot.namespace,
            "payload_sha256": sha256Hex(snapshot.payload),
            "payload_utf8": payloadJSON,
            "schema": snapshot.schema,
            "sensitivity": snapshot.sensitivity,
            "status": snapshot.status,
        ]
        let receiptData = try Edge.canonicalJSONData(receipt)
        XCTAssertEqual(sha256Hex(receiptData), Self.expectedReceiptSHA256)
    }

    func testManifestShapeMatchesPythonCanonicalHash() throws {
        let manifest: [String: Any] = [
            "schema_version": "edgestudio.route_router_manifest.v0",
            "router_type": "matrix_v0",
            "training_run_id": "rrr-fixture-001",
            "min_runtime_version": "0.9.0",
            "encoder": [
                "kind": "base_model_last_hidden",
                "base_model_id": "qwen3.5-4b-base",
                "hidden_size": 2_048,
                "layer_index": -1,
                "pooling": "mean_excluding_special",
                "tokenizer_sha256": String(repeating: "a", count: 64),
            ] as [String: Any],
            "matrices": [
                "intent": [
                    "file": "route_intent_matrix.safetensors",
                    "tensor": "intent_weights",
                    "bias_tensor": "intent_bias",
                    "dtype": "float16",
                    "shape": [2_048, 6],
                ] as [String: Any],
                "confidence": [
                    "file": "route_confidence_matrix.safetensors",
                    "tensor": "confidence_weights",
                    "dtype": "float32",
                    "shape": [2_048, 3],
                ] as [String: Any],
            ] as [String: Any],
            "fallback_chain": ["matrix", "evidence_matcher", "base_router"],
            "intent_vocab": ["base_chat", "exact_fact", "aggregate_fact", "app_action", "user_profile", "mixed"],
            "thresholds": [
                "exact_match": 0.1,
                "shadow_accept": 1.0 / 3.0,
                "disabled": NSNull(),
            ] as [String: Any],
            "empty_files": [],
        ]

        let manifestData = try Edge.canonicalJSONData(manifest)
        XCTAssertEqual(String(data: manifestData, encoding: .utf8), Self.expectedManifestJSON)
        XCTAssertEqual(sha256Hex(manifestData), Self.expectedManifestSHA256)
    }

    func testPrimitiveTableDefaultNamespaceIsAppNeutral() throws {
        let dbQueue = try makeMigratedDatabase()
        let emptyPayload = Data(#"{}"#.utf8)
        let nowMs: Int64 = 1_762_000_000_000
        let createdAt = Date(timeIntervalSince1970: 1_762_000_000)

        try dbQueue.write { db in
            try db.execute(
                sql: "INSERT INTO events (id, ts_ms, created_at) VALUES (?, ?, ?)",
                arguments: ["event-default-namespace", nowMs, createdAt]
            )
            try db.execute(
                sql: "INSERT INTO facts (id, ts_ms, schema, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                arguments: ["fact-default-namespace", nowMs, "test.default.fact", emptyPayload, createdAt]
            )
            try db.execute(
                sql: "INSERT INTO traces (id, ts_ms, action, created_at) VALUES (?, ?, ?, ?)",
                arguments: ["trace-default-namespace", nowMs, "test_action", createdAt]
            )
            try db.execute(
                sql: "INSERT INTO artifacts (content_hash, type, size_bytes, stored_path, created_at) VALUES (?, ?, ?, ?, ?)",
                arguments: [String(repeating: "a", count: 64), "text", 2, "artifact.txt", createdAt]
            )
        }

        let namespaces = try dbQueue.read { db in
            try String.fetchAll(
                db,
                sql: """
                SELECT namespace FROM events
                UNION ALL SELECT namespace FROM facts
                UNION ALL SELECT namespace FROM traces
                UNION ALL SELECT namespace FROM artifacts
                """
            )
        }

        XCTAssertEqual(namespaces.count, 4)
        XCTAssertEqual(Set(namespaces), [Edge.defaultNamespace])
    }

    func testApplyClassificationKeepsFactNamespaceInEventAndTrainingSink() throws {
        let dbQueue = try makeMigratedDatabase()
        let sink = RecordingTrainingSink()
        Edge.bootstrap(dbQueue: dbQueue, trainingDataSink: sink)

        let namespace = "com.example.scaffold:developer"
        let factID = try Edge.recordRaw(
            fact: RawFact(
                namespace: namespace,
                rawPayload: ["ocr": "coffee receipt"],
                candidateSchemas: ["finance.expense"],
                sensitivity: .trainingOk
            ),
            customFactId: "edge-namespace-fixture-001"
        )

        try Edge.applyClassification(
            factId: factID,
            schema: "finance.expense",
            payload: ["amount": 18.8, "category": "food"],
            confidence: 0.91,
            modelVer: "test-classifier@namespace",
            reasoning: "unit test"
        )

        let snapshot = try dbQueue.read { db -> (factNamespace: String, eventNamespace: String, eventSensitivity: Int) in
            guard let factNamespace = try String.fetchOne(
                db,
                sql: "SELECT namespace FROM facts WHERE id = ?",
                arguments: [factID]
            ) else {
                throw HashGateError.missingFixtureRow
            }
            guard let event = try Row.fetchOne(
                db,
                sql: "SELECT namespace, sensitivity FROM events WHERE session_id = ? AND derived_facts = ?",
                arguments: ["classification", "[\"\(factID)\"]"]
            ) else {
                throw HashGateError.missingFixtureRow
            }
            return (
                factNamespace,
                event["namespace"],
                event["sensitivity"]
            )
        }

        XCTAssertEqual(snapshot.factNamespace, namespace)
        XCTAssertEqual(snapshot.eventNamespace, namespace)
        XCTAssertEqual(snapshot.eventSensitivity, Sensitivity.trainingOk.rawValue)

        let samples = sink.collectedSamples
        XCTAssertEqual(samples.count, 1)
        let sample = try XCTUnwrap(samples.first)
        XCTAssertEqual(sample.appId, namespace)
        XCTAssertEqual(sample.eventType, "classification")

        let payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: sample.payload) as? [String: Any]
        )
        XCTAssertEqual(payload["namespace"] as? String, namespace)
        XCTAssertEqual(payload["schema"] as? String, "finance.expense")
    }

    func testPreCanonicalJSONSerializationBytesRequireMigrationBeforeHashComparison() throws {
        let payload: [String: Any] = [
            "float_one": 1.0,
            "negative_zero": -0.0,
            "tenth": 0.1,
            "third": 1.0 / 3.0,
        ]

        let legacyData = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        let canonicalData = try Edge.canonicalJSONData(payload)

        XCTAssertEqual(
            String(data: legacyData, encoding: .utf8),
            #"{"float_one":1,"negative_zero":-0,"tenth":0.10000000000000001,"third":0.33333333333333331}"#
        )
        XCTAssertEqual(
            String(data: canonicalData, encoding: .utf8),
            #"{"float_one":1.0,"negative_zero":-0.0,"tenth":0.1,"third":0.3333333333333333}"#
        )
        XCTAssertNotEqual(legacyData, canonicalData)
        XCTAssertNotEqual(sha256Hex(legacyData), sha256Hex(canonicalData))
    }

    func testCanonicalJSONRejectsSwiftOnlyDecimalValues() {
        XCTAssertThrowsError(try Edge.canonicalJSONData(["amount": Decimal(12.5)]))
    }

    private func makeMigratedDatabase() throws -> DatabaseQueue {
        let dbURL = tmpDir.appendingPathComponent("edge_data.sqlite")
        let dbQueue = try DatabaseQueue(path: dbURL.path)
        var migrator = DatabaseMigrator()
        V2APrimitiveTables.register(&migrator)
        V3ClassificationLifecycle.register(&migrator)
        V5RetryFields.register(&migrator)
        try migrator.migrate(dbQueue)
        return dbQueue
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private struct Snapshot {
        let id: String
        let schema: String
        let namespace: String
        let payload: Data
        let sensitivity: Int
        let status: String
    }

    private enum HashGateError: Error {
        case missingFixtureRow
    }

    private final class RecordingTrainingSink: EdgeTrainingDataSink, @unchecked Sendable {
        struct Sample {
            let appId: String
            let eventType: String
            let payload: Data
            let tags: [String]
        }

        private let lock = NSLock()
        private var storage: [Sample] = []

        var collectedSamples: [Sample] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        func collectTrainingSample(
            appId: String,
            eventType: String,
            payload: Data,
            tags: [String]
        ) throws {
            lock.lock()
            defer { lock.unlock() }
            storage.append(Sample(appId: appId, eventType: eventType, payload: payload, tags: tags))
        }
    }
}
