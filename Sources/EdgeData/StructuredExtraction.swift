// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Structured output from an app-owned extractor.
///
/// The app still owns the prompt, model call, and domain mapping. EdgeData only
/// persists the already-structured result as a classified fact so the same raw
/// fact is not sent through the classification daemon again.
public struct StructuredExtractionResult {
    public let schema: String
    public let payload: [String: Any]
    public let confidence: Double
    public let modelVersion: String
    public let reasoning: String?

    public init(
        schema: String,
        payload: [String: Any],
        confidence: Double,
        modelVersion: String,
        reasoning: String? = nil
    ) {
        self.schema = schema
        self.payload = payload
        self.confidence = confidence
        self.modelVersion = modelVersion
        self.reasoning = reasoning
    }
}

public struct StructuredExtractionWriteReceipt: Equatable {
    public let factId: String
    public let schema: String
    public let status: FactStatus
    public let confidence: Double
    public let modelVersion: String
    public let didPostClassificationNotification: Bool
}

public enum StructuredExtractionError: Error, Equatable, LocalizedError {
    case invalidConfidence(Double)

    public var errorDescription: String? {
        switch self {
        case .invalidConfidence(let confidence):
            return "Structured extraction confidence must be in 0.0...1.0: \(confidence)"
        }
    }
}

public extension Edge {
    /// Persist an app-owned structured extraction result as a classified fact.
    ///
    /// Use this when the model-facing UX has already produced a structured JSON
    /// object, for example receipt OCR → expense draft. The app maps that JSON
    /// into its registered `SchemaDef` payload, then calls this method to update
    /// the raw fact and emit the standard classification event.
    static func applyStructuredExtraction(
        factId: String,
        result: StructuredExtractionResult,
        postClassificationNotification: Bool = true
    ) throws -> StructuredExtractionWriteReceipt {
        try applyStructuredExtraction(
            factId: factId,
            schema: result.schema,
            payload: result.payload,
            confidence: result.confidence,
            modelVersion: result.modelVersion,
            reasoning: result.reasoning,
            postClassificationNotification: postClassificationNotification
        )
    }

    /// Persist an app-owned structured extraction result as a classified fact.
    static func applyStructuredExtraction(
        factId: String,
        schema: String,
        payload: [String: Any],
        confidence: Double,
        modelVersion: String,
        reasoning: String?,
        postClassificationNotification: Bool = true
    ) throws -> StructuredExtractionWriteReceipt {
        guard Edge.schema(schema) != nil else {
            throw EdgeError.schemaNotRegistered(schema)
        }
        guard confidence >= 0.0 && confidence <= 1.0 else {
            throw StructuredExtractionError.invalidConfidence(confidence)
        }

        try Edge.applyClassification(
            factId: factId,
            schema: schema,
            payload: payload,
            confidence: confidence,
            modelVer: modelVersion,
            reasoning: reasoning
        )

        if postClassificationNotification {
            NotificationCenter.default.post(
                name: .edgeClassified,
                object: nil,
                userInfo: ["factId": factId]
            )
        }

        return StructuredExtractionWriteReceipt(
            factId: factId,
            schema: schema,
            status: .classified,
            confidence: confidence,
            modelVersion: modelVersion,
            didPostClassificationNotification: postClassificationNotification
        )
    }
}
