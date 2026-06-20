// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import GRDB
import XCTest
@testable import EdgeData

final class StructuredExtractionTests: XCTestCase {
    var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("structured-extraction-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    func testApplyStructuredExtractionPersistsClassifiedFactAndPostsNotification() throws {
        let dbQueue = try makeMigratedDatabase()
        Edge.bootstrap(dbQueue: dbQueue)
        let schemaName = "test.structured.transaction.\(UUID().uuidString)"
        Edge.registerSchema(makeTransactionSchema(name: schemaName))

        let factId = try Edge.recordRaw(
            fact: RawFact(
                namespace: "com.atomgradient.structured:default",
                rawPayload: ["ocr": "coffee receipt"],
                candidateSchemas: [schemaName],
                sensitivity: .meshOk
            ),
            customFactId: "structured-extraction-fact-001"
        )

        let notification = expectation(description: "edgeClassified notification")
        let token = NotificationCenter.default.addObserver(
            forName: .edgeClassified,
            object: nil,
            queue: nil
        ) { note in
            if note.userInfo?["factId"] as? String == factId {
                notification.fulfill()
            }
        }
        defer { NotificationCenter.default.removeObserver(token) }

        let receipt = try Edge.applyStructuredExtraction(
            factId: factId,
            schema: schemaName,
            payload: [
                "amount": 12.5,
                "category": "food",
                "description": "coffee"
            ],
            confidence: 0.86,
            modelVersion: "test-model@structured-extraction",
            reasoning: "extractor returned JSON"
        )

        wait(for: [notification], timeout: 1.0)
        XCTAssertEqual(receipt.factId, factId)
        XCTAssertEqual(receipt.schema, schemaName)
        XCTAssertEqual(receipt.status, .classified)
        XCTAssertEqual(receipt.confidence, 0.86)
        XCTAssertEqual(receipt.modelVersion, "test-model@structured-extraction")
        XCTAssertTrue(receipt.didPostClassificationNotification)

        let facts = try Edge.queryFacts(namespace: "com.atomgradient.structured:default", schema: schemaName)
        XCTAssertEqual(facts.count, 1)
        let fact = try XCTUnwrap(facts.first)
        XCTAssertEqual(fact.status, .classified)
        XCTAssertEqual(fact.classificationConfidence, 0.86)
        XCTAssertEqual(fact.classificationModelVer, "test-model@structured-extraction")
        XCTAssertEqual(fact.payload["description"] as? String, "coffee")
    }

    func testApplyStructuredExtractionRejectsUnregisteredSchemaBeforeWriting() throws {
        let dbQueue = try makeMigratedDatabase()
        Edge.bootstrap(dbQueue: dbQueue)
        let registeredSchema = "test.structured.registered.\(UUID().uuidString)"
        Edge.registerSchema(makeTransactionSchema(name: registeredSchema))

        let factId = try Edge.recordRaw(
            fact: RawFact(
                namespace: "com.atomgradient.structured:default",
                rawPayload: ["ocr": "coffee receipt"],
                candidateSchemas: [registeredSchema],
                sensitivity: .meshOk
            ),
            customFactId: "structured-extraction-fact-002"
        )

        XCTAssertThrowsError(
            try Edge.applyStructuredExtraction(
                factId: factId,
                schema: "test.structured.unregistered.\(UUID().uuidString)",
                payload: ["description": "coffee"],
                confidence: 0.8,
                modelVersion: "test-model",
                reasoning: nil
            )
        ) { error in
            guard case EdgeError.schemaNotRegistered = error else {
                return XCTFail("expected schemaNotRegistered, got \(error)")
            }
        }

        let rawFacts = try Edge.queryFacts(namespace: "com.atomgradient.structured:default", status: .rawUnclassified)
        XCTAssertEqual(rawFacts.map(\.id), [factId])
    }

    func testApplyStructuredExtractionRejectsOutOfRangeConfidence() throws {
        let dbQueue = try makeMigratedDatabase()
        Edge.bootstrap(dbQueue: dbQueue)
        let schemaName = "test.structured.confidence.\(UUID().uuidString)"
        Edge.registerSchema(makeTransactionSchema(name: schemaName))

        let factId = try Edge.recordRaw(
            fact: RawFact(
                namespace: "com.atomgradient.structured:default",
                rawPayload: ["ocr": "coffee receipt"],
                candidateSchemas: [schemaName],
                sensitivity: .meshOk
            ),
            customFactId: "structured-extraction-fact-003"
        )

        XCTAssertThrowsError(
            try Edge.applyStructuredExtraction(
                factId: factId,
                schema: schemaName,
                payload: ["description": "coffee"],
                confidence: 1.5,
                modelVersion: "test-model",
                reasoning: nil
            )
        ) { error in
            XCTAssertEqual(error as? StructuredExtractionError, .invalidConfidence(1.5))
        }
    }

    private func makeTransactionSchema(name: String) -> SchemaDef {
        SchemaDef(
            name: name,
            fields: [
                FieldDef(name: "amount", type: .numeric, required: true),
                FieldDef(name: "category", type: .categorical(["food", "other"])),
                FieldDef(name: "description", type: .text),
            ],
            semanticLabels: SemanticLabels(primaryValue: "amount", primaryEntity: "description")
        )
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
}
