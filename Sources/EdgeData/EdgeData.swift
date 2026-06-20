// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import GRDB
import Foundation

/// Data handling policy for facts and events.
public enum Sensitivity: Int, Codable {
    /// Keep the record on the local device.
    case localOnly = 1
    /// Allow sync inside the user's EdgeMesh.
    case meshOk = 2
    /// Allow use as a training or learning sample.
    case trainingOk = 3
}

/// Lifecycle state for a fact.
public enum FactStatus: String, Codable {
    /// Raw record awaiting classification.
    case rawUnclassified = "raw_unclassified"
    /// Fact has a resolved schema and payload.
    case classified = "classified"
    /// Classification retries were exhausted.
    case classificationFailed = "classification_failed"
    /// User input is required before the fact can be resolved.
    case manualClassificationRequired = "manual_classification_required"
    /// Fact has been deleted.
    case deleted = "deleted"
    /// Fact expired by retention policy.
    case decayed = "decayed"
}

/// Supported payload field types for a schema.
public enum FieldType {
    /// Numeric value.
    case numeric
    /// Free-form text value.
    case text
    /// Value constrained to one of the supplied strings.
    case categorical([String])
    /// Named entity value.
    case entity
    /// Millisecond UTC timestamp.
    case timestamp
    /// Geohash value.
    case geohash
}

/// Schema field declaration.
public struct FieldDef {
    /// Field name in the payload.
    public let name: String
    /// Field value type.
    public let type: FieldType
    /// Optional field-level sensitivity override.
    public let sensitivity: Sensitivity?
    /// Whether the field must be present.
    public let required: Bool

    public init(name: String, type: FieldType, sensitivity: Sensitivity? = nil, required: Bool = false) {
        self.name = name
        self.type = type
        self.sensitivity = sensitivity
        self.required = required
    }
}

/// Schema-level hints used by query and presentation layers.
public struct SemanticLabels {
    /// Primary timestamp field name.
    public let primaryTimestamp: String?
    /// Primary numeric or value field name.
    public let primaryValue: String?
    /// Primary entity field name.
    public let primaryEntity: String?

    public init(primaryTimestamp: String? = nil,
                primaryValue: String? = nil,
                primaryEntity: String? = nil) {
        self.primaryTimestamp = primaryTimestamp
        self.primaryValue = primaryValue
        self.primaryEntity = primaryEntity
    }
}

/// Schema declaration for classified facts.
public struct SchemaDef {
    /// Dotted schema name such as `finance.expense`.
    public let name: String
    /// Fields accepted by this schema.
    public let fields: [FieldDef]
    /// Schema-level semantic hints.
    public let semanticLabels: SemanticLabels

    public init(name: String, fields: [FieldDef], semanticLabels: SemanticLabels) {
        self.name = name
        self.fields = fields
        self.semanticLabels = semanticLabels
    }
}

/// Raw fact submitted before schema classification.
public struct RawFact {
    /// Application namespace for the record.
    public let namespace: String
    /// JSON-encodable payload.
    public let rawPayload: [String: Any]
    /// Candidate schemas the classifier may choose from.
    public let candidateSchemas: [String]
    /// Handling policy for the record.
    public let sensitivity: Sensitivity

    public init(namespace: String,
                rawPayload: [String: Any],
                candidateSchemas: [String],
                sensitivity: Sensitivity = .meshOk) {
        self.namespace = namespace
        self.rawPayload = rawPayload
        self.candidateSchemas = candidateSchemas
        self.sensitivity = sensitivity
    }
}

/// Classified fact returned by EdgeData queries or inserted explicitly.
public struct Fact: Identifiable {
    public let id: String
    public let namespace: String
    public let schema: String
    public let payload: [String: Any]
    public let derivedFromEventId: String?
    public let sensitivity: Sensitivity
    public let status: FactStatus
    public let classificationConfidence: Double?
    public let classificationModelVer: String?
    public let classifiedAt: Int64?
    public let classificationCorrectedAt: Int64?
    public let tsMs: Int64
    public let createdAt: Date

    public init(id: String,
                namespace: String,
                schema: String,
                payload: [String: Any],
                derivedFromEventId: String? = nil,
                sensitivity: Sensitivity = .meshOk,
                tsMs: Int64? = nil) {
        self.id = id
        self.namespace = namespace
        self.schema = schema
        self.payload = payload
        self.derivedFromEventId = derivedFromEventId
        self.sensitivity = sensitivity
        self.status = .classified
        self.classificationConfidence = 1.0
        self.classificationModelVer = "explicit"
        self.classifiedAt = Int64(Date().timeIntervalSince1970 * 1000)
        self.classificationCorrectedAt = nil
        self.tsMs = tsMs ?? Int64(Date().timeIntervalSince1970 * 1000)
        self.createdAt = Date()
    }

    fileprivate init(rowValues: RowValues) {
        self.id = rowValues.id
        self.namespace = rowValues.namespace
        self.schema = rowValues.schema
        self.payload = rowValues.payload
        self.derivedFromEventId = rowValues.derivedFromEventId
        self.sensitivity = Sensitivity(rawValue: rowValues.sensitivity) ?? .meshOk
        self.status = FactStatus(rawValue: rowValues.status) ?? .classified
        self.classificationConfidence = rowValues.classificationConfidence
        self.classificationModelVer = rowValues.classificationModelVer
        self.classifiedAt = rowValues.classifiedAt
        self.classificationCorrectedAt = rowValues.classificationCorrectedAt
        self.tsMs = rowValues.tsMs
        self.createdAt = rowValues.createdAt
    }

    fileprivate struct RowValues {
        let id: String
        let namespace: String
        let schema: String
        let payload: [String: Any]
        let derivedFromEventId: String?
        let sensitivity: Int
        let status: String
        let classificationConfidence: Double?
        let classificationModelVer: String?
        let classifiedAt: Int64?
        let classificationCorrectedAt: Int64?
        let tsMs: Int64
        let createdAt: Date
    }
}

public extension Notification.Name {
    /// Posted after a fact is classified.
    static let edgeClassified = Notification.Name("Edge.classification.completed")

    /// Posted after fact classification fails.
    static let edgeClassificationFailed = Notification.Name("Edge.classification.failed")
}

/// Errors raised by EdgeData APIs.
public enum EdgeError: Error, LocalizedError {
    case notBootstrapped
    case schemaNotRegistered(String)
    case payloadEncodingFailed(String)
    case dbError(Error)

    public var errorDescription: String? {
        switch self {
        case .notBootstrapped: return "Edge.bootstrap(dbQueue:) not called"
        case .schemaNotRegistered(let name): return "Schema not registered: \(name)"
        case .payloadEncodingFailed(let reason): return "Payload encoding failed: \(reason)"
        case .dbError(let err): return "DB error: \(err.localizedDescription)"
        }
    }
}

/// Fact status filter for EdgeData queries.
public enum QueryStatus {
    /// Return only classified facts.
    case classifiedOnly
    /// Return raw facts awaiting classification.
    case rawUnclassified
    /// Return facts that need failure handling or manual correction.
    case classificationFailed
    /// Return facts in any status.
    case all
}

/// App-provided sink for forwarding EdgeData events into a training or learning pipeline.
public protocol EdgeTrainingDataSink: Sendable {
    /// Collects one serialized sample with string tags supplied by EdgeData.
    func collectTrainingSample(
        appId: String,
        eventType: String,
        payload: Data,
        tags: [String]
    ) throws
}

public enum Edge {

    public static let defaultNamespace = "com.edgestudio.default:default"

    fileprivate static var dbQueue: DatabaseQueue?
    fileprivate static var schemaRegistry: [String: SchemaDef] = [:]
    fileprivate static var trainingDataSink: (any EdgeTrainingDataSink)?

    @MainActor fileprivate static var onClassifiedHandlers: [(Fact) -> Void] = []
    @MainActor fileprivate static var onClassificationFailedHandlers: [(Fact, ClassificationParseError) -> Void] = []

    /// Initializes EdgeData storage.
    ///
    /// - Parameters:
    ///   - dbQueue: GRDB database queue used by EdgeData.
    ///   - trainingDataSink: Optional sink for forwarding event samples.
    public static func bootstrap(dbQueue: DatabaseQueue,
                                  trainingDataSink: (any EdgeTrainingDataSink)? = nil) {
        Edge.dbQueue = dbQueue
        Edge.trainingDataSink = trainingDataSink
        let sinkStatus = trainingDataSink == nil ? "nil (events only)" : "configured"
        NSLog("[Edge] Bootstrap done, schemaRegistry size=\(schemaRegistry.count), trainingDataSink=\(sinkStatus)")
    }

    /// Registers a schema for the current process.
    public static func registerSchema(_ schema: SchemaDef) {
        schemaRegistry[schema.name] = schema
        NSLog("[Edge] registerSchema: \(schema.name) (\(schema.fields.count) fields, primaryEntity=\(schema.semanticLabels.primaryEntity ?? "nil"))")
    }

    /// Returns a registered schema by name.
    public static func schema(_ name: String) -> SchemaDef? {
        return schemaRegistry[name]
    }

    /// Lists registered schema names.
    public static func registeredSchemaNames() -> [String] {
        return Array(schemaRegistry.keys).sorted()
    }

    /// Writes a raw fact and returns its fact identifier.
    ///
    /// - Parameters:
    ///   - rawFact: Raw payload, namespace, candidate schemas, and sensitivity.
    ///   - customFactId: Optional caller-provided id. If it already exists,
    ///     the insert is skipped and the existing id is returned.
    @discardableResult
    public static func recordRaw(fact rawFact: RawFact, customFactId: String? = nil) throws -> String {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }

        let factId = customFactId ?? generateFactId()
        let tsMs = Int64(Date().timeIntervalSince1970 * 1000)
        let payloadBlob = try jsonBlob(rawFact.rawPayload)
        let bundleId = parseBundleId(rawFact.namespace)
        let placeholderSchema = "raw_\(bundleId)_record"

        do {
            try dbQueue.write { db in
                if customFactId != nil {
                    let existing = try Int.fetchOne(
                        db,
                        sql: "SELECT 1 FROM facts WHERE id = ?",
                        arguments: [factId]
                    )
                    if existing != nil {
                        NSLog("[Edge.recordRaw] customFactId=\(factId) already exists, skip (idempotent)")
                        return
                    }
                }
                try db.execute(
                    sql: """
                    INSERT INTO facts (
                        id, ts_ms, schema, namespace, payload,
                        derived_from_event_id, sensitivity, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        factId,
                        tsMs,
                        placeholderSchema,
                        rawFact.namespace,
                        payloadBlob,
                        nil,
                        rawFact.sensitivity.rawValue,
                        FactStatus.rawUnclassified.rawValue,
                        Date()
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }

        NSLog("[Edge.recordRaw] fact.id=\(factId), schema=\(placeholderSchema), status=raw_unclassified, candidates=\(rawFact.candidateSchemas)")
        return factId
    }

    /// Writes an already-classified fact.
    public static func record(fact: Fact) throws {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        guard schemaRegistry[fact.schema] != nil else {
            throw EdgeError.schemaNotRegistered(fact.schema)
        }

        let payloadBlob = try jsonBlob(fact.payload)

        do {
            try dbQueue.write { db in
                try db.execute(
                    sql: """
                    INSERT INTO facts (
                        id, ts_ms, schema, namespace, payload,
                        derived_from_event_id, sensitivity, status,
                        classification_confidence, classification_model_ver, classified_at,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        fact.id,
                        fact.tsMs,
                        fact.schema,
                        fact.namespace,
                        payloadBlob,
                        fact.derivedFromEventId,
                        fact.sensitivity.rawValue,
                        FactStatus.classified.rawValue,
                        fact.classificationConfidence,
                        fact.classificationModelVer,
                        fact.classifiedAt,
                        Date()
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }

        NSLog("[Edge.record] fact.id=\(fact.id), schema=\(fact.schema), status=classified (explicit)")
    }

    /// Records one AI interaction event.
    ///
    /// - Parameters:
    ///   - namespace: Application namespace that owns the event.
    ///   - sessionId: Optional app-defined session identifier.
    ///   - userInput: Optional JSON-encodable user input payload.
    ///   - aiOutput: Optional JSON-encodable AI output payload.
    ///   - feedback: Optional feedback label. Defaults to `none`.
    ///   - feedbackDetail: Optional JSON-encodable feedback detail payload.
    ///   - derivedFactIds: Optional fact ids associated with the event.
    ///   - sensitivity: Handling policy for the event.
    /// - Returns: event id.
    @discardableResult
    public static func recordEvent(
        namespace: String,
        sessionId: String? = nil,
        userInput: [String: Any]? = nil,
        aiOutput: [String: Any]? = nil,
        feedback: String? = nil,
        feedbackDetail: [String: Any]? = nil,
        derivedFactIds: [String]? = nil,
        sensitivity: Sensitivity = .meshOk
    ) throws -> String {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }

        let eventId = generateFactId()
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)

        let userInputBlob: Data? = try userInput.map { try jsonBlob($0) }
        let aiOutputBlob: Data? = try aiOutput.map { try jsonBlob($0) }
        let feedbackDetailBlob: Data? = try feedbackDetail.map { try jsonBlob($0) }

        let derivedFactsJSON: String? = derivedFactIds.flatMap { ids in
            guard !ids.isEmpty else { return nil }
            let escaped = ids.map { "\"\($0)\"" }.joined(separator: ",")
            return "[\(escaped)]"
        }

        let resolvedFeedback = feedback ?? "none"

        do {
            try dbQueue.write { db in
                try db.execute(
                    sql: """
                    INSERT INTO events (
                        id, ts_ms, session_id, namespace,
                        user_input, ai_output, feedback, feedback_detail,
                        derived_facts, derived_artifacts,
                        sensitivity, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        eventId, nowMs, sessionId, namespace,
                        userInputBlob, aiOutputBlob,
                        resolvedFeedback,
                        feedbackDetailBlob,
                        derivedFactsJSON,
                        nil,
                        sensitivity.rawValue,
                        Date()
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }

        NSLog("[Edge.recordEvent] event.id=\(eventId), namespace=\(namespace), feedback=\(resolvedFeedback), session=\(sessionId ?? "nil")")

        var forkPayload: [String: Any] = [
            "namespace": namespace,
            "session_id": sessionId ?? "",
            "feedback": resolvedFeedback,
            "ts_ms": nowMs
        ]
        if let userInput = userInput { forkPayload["user_input"] = userInput }
        if let aiOutput = aiOutput { forkPayload["ai_output"] = aiOutput }
        if let feedbackDetail = feedbackDetail { forkPayload["feedback_detail"] = feedbackDetail }
        if let derivedFactIds = derivedFactIds, !derivedFactIds.isEmpty {
            forkPayload["derived_facts"] = derivedFactIds
        }
        let forkTags = trainingTags(for: resolvedFeedback)
        forkToTrainingSink(
            appId: namespace,
            eventType: "edge_event:\(resolvedFeedback)",
            payload: forkPayload,
            tags: forkTags
        )

        return eventId
    }

    /// Counts events that carry explicit user feedback.
    public static func countFeedbackEvents(namespace: String? = nil) throws -> Int {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        return try dbQueue.read { db in
            var sql = "SELECT COUNT(*) FROM events WHERE feedback IS NOT NULL AND feedback != 'none'"
            var arguments: [DatabaseValueConvertible?] = []
            if let namespace = namespace {
                sql += " AND namespace = ?"
                arguments.append(namespace)
            }
            let count = try Int.fetchOne(db, sql: sql, arguments: StatementArguments(arguments)) ?? 0
            return count
        }
    }

    /// Queries facts by namespace, schema, status, and limit.
    public static func queryFacts(namespace: String? = nil,
                                   schema: String? = nil,
                                   status: QueryStatus = .classifiedOnly,
                                   limit: Int? = nil) throws -> [Fact] {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }

        var conditions: [String] = []
        var arguments: [DatabaseValueConvertible?] = []

        if let namespace = namespace {
            conditions.append("namespace = ?")
            arguments.append(namespace)
        }
        if let schema = schema {
            conditions.append("schema = ?")
            arguments.append(schema)
        }
        switch status {
        case .classifiedOnly:
            conditions.append("status = ?")
            arguments.append(FactStatus.classified.rawValue)
        case .rawUnclassified:
            conditions.append("status = ?")
            arguments.append(FactStatus.rawUnclassified.rawValue)
        case .classificationFailed:
            conditions.append("status IN (?, ?)")
            arguments.append(FactStatus.classificationFailed.rawValue)
            arguments.append(FactStatus.manualClassificationRequired.rawValue)
        case .all:
            break
        }

        let whereClause = conditions.isEmpty ? "" : "WHERE " + conditions.joined(separator: " AND ")
        let limitClause = limit.map { "LIMIT \($0)" } ?? ""
        let sql = """
            SELECT id, ts_ms, schema, namespace, payload,
                   derived_from_event_id, sensitivity, status,
                   classification_confidence, classification_model_ver,
                   classified_at, classification_corrected_at, created_at
            FROM facts
            \(whereClause)
            ORDER BY ts_ms DESC
            \(limitClause)
            """

        do {
            return try dbQueue.read { db in
                let rows = try Row.fetchAll(db, sql: sql, arguments: StatementArguments(arguments))
                return try rows.compactMap { row -> Fact? in
                    guard let payloadData = row["payload"] as Data?,
                          let payloadObj = try? JSONSerialization.jsonObject(with: payloadData),
                          let payloadDict = payloadObj as? [String: Any] else {
                        NSLog("[Edge.queryFacts] ⚠️ skip row, invalid payload: id=\(row["id"] ?? "nil")")
                        return nil
                    }
                    let rowValues = Fact.RowValues(
                        id: row["id"],
                        namespace: row["namespace"],
                        schema: row["schema"],
                        payload: payloadDict,
                        derivedFromEventId: row["derived_from_event_id"],
                        sensitivity: row["sensitivity"],
                        status: row["status"],
                        classificationConfidence: row["classification_confidence"],
                        classificationModelVer: row["classification_model_ver"],
                        classifiedAt: row["classified_at"],
                        classificationCorrectedAt: row["classification_corrected_at"],
                        tsMs: row["ts_ms"],
                        createdAt: row["created_at"]
                    )
                    return Fact(rowValues: rowValues)
                }
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Counts facts by namespace and status.
    public static func countFacts(namespace: String? = nil,
                                   status: QueryStatus = .classifiedOnly) throws -> Int {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }

        var conditions: [String] = []
        var arguments: [DatabaseValueConvertible?] = []
        if let namespace = namespace {
            conditions.append("namespace = ?")
            arguments.append(namespace)
        }
        switch status {
        case .classifiedOnly:
            conditions.append("status = ?")
            arguments.append(FactStatus.classified.rawValue)
        case .rawUnclassified:
            conditions.append("status = ?")
            arguments.append(FactStatus.rawUnclassified.rawValue)
        case .classificationFailed:
            conditions.append("status IN (?, ?)")
            arguments.append(FactStatus.classificationFailed.rawValue)
            arguments.append(FactStatus.manualClassificationRequired.rawValue)
        case .all:
            break
        }

        let whereClause = conditions.isEmpty ? "" : "WHERE " + conditions.joined(separator: " AND ")
        let sql = "SELECT COUNT(*) FROM facts \(whereClause)"

        do {
            return try dbQueue.read { db in
                try Int.fetchOne(db, sql: sql, arguments: StatementArguments(arguments)) ?? 0
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Applies a classifier result to a raw fact.
    /// `ClassificationDaemon` uses this after an SDK-owned classify pass.
    /// Apps may also call it when their own model-facing flow already produced
    /// a structured result and only needs to persist that result into EdgeData.
    public static func applyClassification(factId: String,
                                           schema: String,
                                           payload: [String: Any],
                                           confidence: Double,
                                           modelVer: String,
                                           reasoning: String?) throws {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        let payloadBlob = try jsonBlob(payload)
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)

        let aiOutputDict: [String: Any] = [
            "tool_calls": [[
                "name": "classify_fact",
                "arguments": [
                    "fact_id": factId,
                    "schema": schema,
                    "payload": payload,
                    "confidence": confidence,
                    "reasoning": reasoning ?? ""
                ]
            ]]
        ]
        let aiOutputBlob = try jsonBlob(aiOutputDict)
        let eventId = generateFactId()

        var namespaceCaptured = Edge.defaultNamespace
        var sensitivityCaptured = Sensitivity.meshOk.rawValue

        do {
            try dbQueue.write { db in
                guard let row = try Row.fetchOne(
                    db,
                    sql: "SELECT namespace, sensitivity FROM facts WHERE id = ?",
                    arguments: [factId]
                ) else {
                    throw EdgeError.dbError(NSError(
                        domain: "Edge.applyClassification",
                        code: 404,
                        userInfo: [NSLocalizedDescriptionKey: "fact not found: \(factId)"]
                    ))
                }
                namespaceCaptured = row["namespace"]
                sensitivityCaptured = row["sensitivity"]

                try db.execute(
                    sql: """
                    UPDATE facts SET
                        schema = ?, payload = ?, status = ?,
                        classification_confidence = ?,
                        classification_model_ver = ?,
                        classified_at = ?
                    WHERE id = ?
                    """,
                    arguments: [
                        schema, payloadBlob, FactStatus.classified.rawValue,
                        confidence, modelVer, nowMs, factId
                    ]
                )
                try db.execute(
                    sql: """
                    INSERT INTO events (
                        id, ts_ms, session_id, namespace,
                        user_input, ai_output, feedback, feedback_detail,
                        derived_facts, derived_artifacts,
                        sensitivity, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        eventId, nowMs, "classification",
                        namespaceCaptured,
                        nil,
                        aiOutputBlob,
                        "none",
                        nil,
                        "[\"\(factId)\"]",
                        nil,
                        sensitivityCaptured,
                        Date()
                    ]
                )
            }
        } catch let edgeError as EdgeError {
            throw edgeError
        } catch {
            throw EdgeError.dbError(error)
        }

        let forkPayload: [String: Any] = [
            "namespace": namespaceCaptured,
            "session_id": "classification",
            "feedback": "none",
            "ts_ms": nowMs,
            "ai_output": aiOutputDict,
            "derived_facts": [factId],
            "schema": schema,
            "confidence": confidence,
            "model_ver": modelVer
        ]
        forkToTrainingSink(
            appId: namespaceCaptured,
            eventType: "classification",
            payload: forkPayload,
            tags: ["trainingSample", "conversation"]
        )
    }

    /// Marks a fact as permanently failed after classification retries are exhausted.
    internal static func markClassificationFailed(factId: String,
                                                    modelVer: String,
                                                    error: ClassificationParseError) throws {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        do {
            try dbQueue.write { db in
                try db.execute(
                    sql: """
                    UPDATE facts SET status = ?, classification_model_ver = ?
                    WHERE id = ?
                    """,
                    arguments: [
                        FactStatus.classificationFailed.rawValue,
                        "\(modelVer):parse_failed:\(error)",
                        factId
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Records a failed classification attempt while keeping the fact retryable.
    internal static func markRetry(factId: String,
                                    modelVer: String,
                                    newRetryCount: Int,
                                    lastAttemptAtMs: Int64,
                                    parseError: ClassificationParseError) throws {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        do {
            try dbQueue.write { db in
                try db.execute(
                    sql: """
                    UPDATE facts SET
                        classification_retry_count = ?,
                        classification_last_attempt_at = ?,
                        classification_model_ver = ?
                    WHERE id = ?
                    """,
                    arguments: [
                        newRetryCount,
                        lastAttemptAtMs,
                        "\(modelVer):retry_\(newRetryCount):\(parseError)",
                        factId
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Returns raw facts whose classification retry backoff has elapsed.
    internal static func queryRetryEligibleFacts(namespace: String,
                                                   limit: Int = 5) throws -> [Fact] {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)

        let sql = """
            SELECT id, ts_ms, schema, namespace, payload,
                   derived_from_event_id, sensitivity, status,
                   classification_confidence, classification_model_ver,
                   classified_at, classification_corrected_at, created_at
            FROM facts
            WHERE namespace = ?
              AND status = ?
              AND classification_retry_count < 3
              AND (
                classification_last_attempt_at IS NULL
                OR (classification_retry_count = 0)
                OR (classification_retry_count = 1 AND classification_last_attempt_at <= ? - 5000)
                OR (classification_retry_count = 2 AND classification_last_attempt_at <= ? - 30000)
              )
            ORDER BY (classification_last_attempt_at IS NULL) DESC,
                     classification_last_attempt_at ASC,
                     ts_ms ASC
            LIMIT ?
            """

        do {
            return try dbQueue.read { db in
                let rows = try Row.fetchAll(db, sql: sql, arguments: [
                    namespace,
                    FactStatus.rawUnclassified.rawValue,
                    nowMs, nowMs, limit
                ])
                return rows.compactMap { row -> Fact? in
                    guard let payloadData = row["payload"] as Data?,
                          let payloadObj = try? JSONSerialization.jsonObject(with: payloadData),
                          let payloadDict = payloadObj as? [String: Any] else {
                        NSLog("[Edge.queryRetryEligibleFacts] ⚠️ skip row, invalid payload: id=\(row["id"] ?? "nil")")
                        return nil
                    }
                    let rowValues = Fact.RowValues(
                        id: row["id"],
                        namespace: row["namespace"],
                        schema: row["schema"],
                        payload: payloadDict,
                        derivedFromEventId: row["derived_from_event_id"],
                        sensitivity: row["sensitivity"],
                        status: row["status"],
                        classificationConfidence: row["classification_confidence"],
                        classificationModelVer: row["classification_model_ver"],
                        classifiedAt: row["classified_at"],
                        classificationCorrectedAt: row["classification_corrected_at"],
                        tsMs: row["ts_ms"],
                        createdAt: row["created_at"]
                    )
                    return Fact(rowValues: rowValues)
                }
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Returns the current classification retry count for a fact.
    internal static func currentRetryCount(factId: String) throws -> Int {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        do {
            return try dbQueue.read { db in
                try Int.fetchOne(
                    db,
                    sql: "SELECT classification_retry_count FROM facts WHERE id = ?",
                    arguments: [factId]
                ) ?? 0
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    /// Applies a user correction to a fact schema and payload.
    ///
    /// throws:
    /// - EdgeError.notBootstrapped
    /// - EdgeError.schemaNotRegistered(newSchema)
    /// - EdgeError.payloadEncodingFailed
    /// - EdgeError.dbError, including missing fact errors
    public static func correctClassification(factId: String,
                                              newSchema: String,
                                              newPayload: [String: Any]) async throws {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        guard schemaRegistry[newSchema] != nil else {
            throw EdgeError.schemaNotRegistered(newSchema)
        }

        let newPayloadBlob = try jsonBlob(newPayload)
        let nowMs = Int64(Date().timeIntervalSince1970 * 1000)

        let raw: OldFactSnapshotRaw
        do {
            raw = try await dbQueue.read { db -> OldFactSnapshotRaw in
                guard let row = try Row.fetchOne(
                    db,
                    sql: "SELECT schema, payload, namespace, sensitivity FROM facts WHERE id = ?",
                    arguments: [factId]
                ) else {
                    throw EdgeError.dbError(NSError(
                        domain: "Edge.correctClassification",
                        code: 404,
                        userInfo: [NSLocalizedDescriptionKey: "fact not found: \(factId)"]
                    ))
                }
                return OldFactSnapshotRaw(
                    schema: row["schema"],
                    payloadData: row["payload"] ?? Data(),
                    namespace: row["namespace"],
                    sensitivity: row["sensitivity"]
                )
            }
        } catch let edgeError as EdgeError {
            throw edgeError
        } catch {
            throw EdgeError.dbError(error)
        }

        let oldPayload: [String: Any] = {
            if let obj = try? JSONSerialization.jsonObject(with: raw.payloadData),
               let dict = obj as? [String: Any] {
                return dict
            }
            return [:]
        }()
        let feedbackDetail: [String: Any] = [
            "old_schema": raw.schema,
            "old_payload": oldPayload,
            "new_schema": newSchema,
            "new_payload": newPayload
        ]
        let feedbackDetailBlob = try jsonBlob(feedbackDetail)
        let eventId = generateFactId()

        let oldSchemaCaptured = raw.schema
        let namespaceCaptured = raw.namespace
        let sensitivityCaptured = raw.sensitivity

        do {
            try await dbQueue.write { db in
                try db.execute(
                    sql: """
                    UPDATE facts SET
                        schema = ?, payload = ?,
                        status = ?,
                        classification_corrected_at = ?
                    WHERE id = ?
                    """,
                    arguments: [
                        newSchema, newPayloadBlob,
                        FactStatus.classified.rawValue,
                        nowMs, factId
                    ]
                )
                try db.execute(
                    sql: """
                    INSERT INTO events (
                        id, ts_ms, session_id, namespace,
                        user_input, ai_output, feedback, feedback_detail,
                        derived_facts, derived_artifacts,
                        sensitivity, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    arguments: [
                        eventId, nowMs, "classification_correction",
                        namespaceCaptured,
                        nil,
                        nil,
                        "modified",
                        feedbackDetailBlob,
                        "[\"\(factId)\"]",
                        nil,
                        sensitivityCaptured,
                        Date()
                    ]
                )
            }
        } catch {
            throw EdgeError.dbError(error)
        }

        NSLog("[Edge.correctClassification] fact.id=\(factId) schema \(oldSchemaCaptured) → \(newSchema), event.id=\(eventId), feedback='modified'")

        let forkPayload: [String: Any] = [
            "namespace": namespaceCaptured,
            "session_id": "classification_correction",
            "feedback": "modified",
            "ts_ms": nowMs,
            "feedback_detail": feedbackDetail,
            "derived_facts": [factId]
        ]
        forkToTrainingSink(
            appId: namespaceCaptured,
            eventType: "user_correction",
            payload: forkPayload,
            tags: ["trainingSample", "userCorrection", "preference"]
        )
    }

    /// Registers a callback for successful fact classification.
    @MainActor
    public static func onClassified(_ handler: @escaping (Fact) -> Void) {
        onClassifiedHandlers.append(handler)
        NSLog("[Edge.onClassified] handler registered, total=\(onClassifiedHandlers.count)")
    }

    /// Registers a callback for failed fact classification.
    @MainActor
    public static func onClassificationFailed(_ handler: @escaping (Fact, ClassificationParseError) -> Void) {
        onClassificationFailedHandlers.append(handler)
        NSLog("[Edge.onClassificationFailed] handler registered, total=\(onClassificationFailedHandlers.count)")
    }

    internal static func emitClassified(factId: String) {
        Task { @MainActor in
            guard let fact = try? fetchFact(id: factId) else {
                NSLog("[Edge.emitClassified] fact not found: \(factId)")
                return
            }
            for h in onClassifiedHandlers { h(fact) }
            NotificationCenter.default.post(
                name: .edgeClassified,
                object: nil,
                userInfo: ["factId": factId]
            )
        }
    }

    internal static func emitClassificationFailed(factId: String, error: ClassificationParseError) {
        Task { @MainActor in
            guard let fact = try? fetchFact(id: factId) else {
                NSLog("[Edge.emitClassificationFailed] fact not found: \(factId)")
                return
            }
            for h in onClassificationFailedHandlers { h(fact, error) }
            NotificationCenter.default.post(
                name: .edgeClassificationFailed,
                object: nil,
                userInfo: [
                    "factId": factId,
                    "errorDescription": error.localizedDescription
                ]
            )
        }
    }

    /// Fetches one fact by id, regardless of status.
    public static func fetchFact(id: String) throws -> Fact? {
        guard let dbQueue = dbQueue else { throw EdgeError.notBootstrapped }
        let sql = """
            SELECT id, ts_ms, schema, namespace, payload,
                   derived_from_event_id, sensitivity, status,
                   classification_confidence, classification_model_ver,
                   classified_at, classification_corrected_at, created_at
            FROM facts WHERE id = ? LIMIT 1
            """
        do {
            return try dbQueue.read { db -> Fact? in
                guard let row = try Row.fetchOne(db, sql: sql, arguments: [id]) else { return nil }
                guard let payloadData = row["payload"] as Data?,
                      let payloadObj = try? JSONSerialization.jsonObject(with: payloadData),
                      let payloadDict = payloadObj as? [String: Any] else {
                    return nil
                }
                let rowValues = Fact.RowValues(
                    id: row["id"],
                    namespace: row["namespace"],
                    schema: row["schema"],
                    payload: payloadDict,
                    derivedFromEventId: row["derived_from_event_id"],
                    sensitivity: row["sensitivity"],
                    status: row["status"],
                    classificationConfidence: row["classification_confidence"],
                    classificationModelVer: row["classification_model_ver"],
                    classifiedAt: row["classified_at"],
                    classificationCorrectedAt: row["classification_corrected_at"],
                    tsMs: row["ts_ms"],
                    createdAt: row["created_at"]
                )
                return Fact(rowValues: rowValues)
            }
        } catch {
            throw EdgeError.dbError(error)
        }
    }

    private struct OldFactSnapshotRaw: Sendable {
        let schema: String
        let payloadData: Data
        let namespace: String
        let sensitivity: Int
    }

    public static func reclassifyAsync(namespace: String,
                                        completion: @escaping (Int) -> Void) {
        completion(0)
    }

    private static func generateFactId() -> String {
        let tsMs = UInt64(Date().timeIntervalSince1970 * 1000)
        let tsHex = String(format: "%012x", tsMs)
        let randHex = (0..<10).map { _ in
            String(format: "%02x", UInt8.random(in: 0...255))
        }.joined()
        return "\(tsHex)-\(randHex)"
    }

    fileprivate static func forkToTrainingSink(
        appId: String,
        eventType: String,
        payload: [String: Any],
        tags: [String]
    ) {
        guard let sink = trainingDataSink else { return }
        do {
            let payloadData = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
            try sink.collectTrainingSample(
                appId: appId,
                eventType: eventType,
                payload: payloadData,
                tags: tags
            )
        } catch {
            NSLog("[Edge] forkToTrainingSink(\(eventType)) failed: \(error.localizedDescription)")
        }
    }

    fileprivate static func trainingTags(for feedback: String) -> [String] {
        switch feedback {
        case "modified":
            return ["trainingSample", "userCorrection", "preference"]
        case "rejected":
            return ["trainingSample", "preference"]
        case "accepted":
            return ["trainingSample", "conversation"]
        default:
            return ["trainingSample", "conversation"]
        }
    }

    private static func parseBundleId(_ namespace: String) -> String {
        let bundleId = namespace.split(separator: ":").first.map(String.init) ?? "unknown"
        let lastComponent = bundleId.split(separator: ".").last.map(String.init) ?? bundleId
        return lastComponent
    }

    /// Python-compatible canonical JSON for EdgeData payload/hash contracts.
    ///
    /// Rows written before this encoder may have been persisted with Foundation's
    /// JSONSerialization byte format. Decode and re-encode historical rows through
    /// this method before comparing their payload hash against new canonical hashes.
    static func canonicalJSONData(_ obj: [String: Any]) throws -> Data {
        guard let data = try canonicalJSONString(obj).data(using: .utf8) else {
            throw EdgeError.payloadEncodingFailed("canonical JSON is not UTF-8 encodable")
        }
        return data
    }

    private static func jsonBlob(_ obj: [String: Any]) throws -> Data {
        try canonicalJSONData(obj)
    }

    private static func canonicalJSONString(_ value: Any) throws -> String {
        switch value {
        case is NSNull:
            return "null"
        case let string as String:
            return escapeJSONString(string)
        case let bool as Bool:
            return bool ? "true" : "false"
        case let int as Int:
            return String(int)
        case let int as Int8:
            return String(int)
        case let int as Int16:
            return String(int)
        case let int as Int32:
            return String(int)
        case let int as Int64:
            return String(int)
        case let uint as UInt:
            return String(uint)
        case let uint as UInt8:
            return String(uint)
        case let uint as UInt16:
            return String(uint)
        case let uint as UInt32:
            return String(uint)
        case let uint as UInt64:
            return String(uint)
        case let float as Float:
            return try finiteFloatString(float)
        case let double as Double:
            return try finiteFloatString(double)
        case let number as NSNumber:
            return try canonicalJSONString(number)
        case let dict as [String: Any]:
            return try canonicalJSONObjectString(dict)
        case let array as [Any]:
            return try "[" + array.map { try canonicalJSONString($0) }.joined(separator: ",") + "]"
        default:
            throw EdgeError.payloadEncodingFailed("unsupported JSON payload value: \(type(of: value))")
        }
    }

    private static func canonicalJSONString(_ number: NSNumber) throws -> String {
        guard !(number is NSDecimalNumber) else {
            throw EdgeError.payloadEncodingFailed("Decimal values are not valid canonical JSON payload values")
        }
        if CFGetTypeID(number) == CFBooleanGetTypeID() {
            return number.boolValue ? "true" : "false"
        }
        if CFNumberIsFloatType(number) {
            return try finiteFloatString(number.doubleValue)
        }
        return number.stringValue
    }

    private static func canonicalJSONObjectString(_ dict: [String: Any]) throws -> String {
        let fields = try dict.keys.sorted().map { key in
            try "\(escapeJSONString(key)):\(canonicalJSONString(dict[key] as Any))"
        }
        return "{\(fields.joined(separator: ","))}"
    }

    private static func finiteFloatString(_ value: Float) throws -> String {
        guard value.isFinite else {
            throw EdgeError.payloadEncodingFailed("non-finite float values are not valid canonical JSON")
        }
        return String(value)
    }

    private static func finiteFloatString(_ value: Double) throws -> String {
        guard value.isFinite else {
            throw EdgeError.payloadEncodingFailed("non-finite double values are not valid canonical JSON")
        }
        return String(value)
    }

    private static func escapeJSONString(_ string: String) -> String {
        var result = "\""
        for scalar in string.unicodeScalars {
            switch scalar {
            case "\"":
                result += "\\\""
            case "\\":
                result += "\\\\"
            case "\u{08}":
                result += "\\b"
            case "\u{0C}":
                result += "\\f"
            case "\n":
                result += "\\n"
            case "\r":
                result += "\\r"
            case "\t":
                result += "\\t"
            case _ where scalar.value < 0x20:
                result += String(format: "\\u%04x", scalar.value)
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        result += "\""
        return result
    }
}

public extension Edge {
    static func verifyStageC1(namespace: String) {
        do {
            let bootstrapped = (Edge.dbQueue != nil)
            let schemas = Edge.registeredSchemaNames()
            let raw = try Edge.countFacts(namespace: namespace, status: .rawUnclassified)
            let classified = try Edge.countFacts(namespace: namespace, status: .classifiedOnly)
            let failed = try Edge.countFacts(namespace: namespace, status: .classificationFailed)
            NSLog("[Edge] verify: bootstrapped=\(bootstrapped), schemas=\(schemas.count) \(schemas), raw=\(raw), classified=\(classified), failed=\(failed)")
        } catch {
            NSLog("[Edge] verify failed: \(error.localizedDescription)")
        }
    }
}
