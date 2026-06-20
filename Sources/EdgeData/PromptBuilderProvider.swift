// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Parsed classification result returned by a prompt builder.
public struct ClassificationResult {
    public let schema: String
    public let payload: [String: Any]
    public let confidence: Double
    public let reasoning: String?

    public init(schema: String,
                payload: [String: Any],
                confidence: Double,
                reasoning: String? = nil) {
        self.schema = schema
        self.payload = payload
        self.confidence = confidence
        self.reasoning = reasoning
    }
}

/// Classification output parse failure.
public enum ClassificationParseError: Error, LocalizedError {
    case invalidJSON(String)
    case missingRequiredField(String)
    case schemaNotInCandidates(String, [String])
    case payloadFieldTypeError(field: String, expected: String, got: String)
    case confidenceOutOfRange(Double)
    case schemaIsFailed

    public var errorDescription: String? {
        switch self {
        case .invalidJSON(let s):
            return "Invalid JSON: \(s.prefix(200))"
        case .missingRequiredField(let f):
            return "Missing required field: \(f)"
        case .schemaNotInCandidates(let got, let candidates):
            return "Schema '\(got)' not in candidates \(candidates)"
        case .payloadFieldTypeError(let f, let e, let g):
            return "Payload field '\(f)' type mismatch: expected \(e), got \(g)"
        case .confidenceOutOfRange(let c):
            return "Confidence out of range [0,1]: \(c)"
        case .schemaIsFailed:
            return "LLM returned __failed__ (unable to classify)"
        }
    }
}

/// App-provided classifier prompt and parser contract.
public protocol PromptBuilderProvider: Sendable {
    /// Builds OpenAI-style chat messages for the classifier model.
    ///
    /// - Parameters:
    ///   - rawFact: Raw fact to classify.
    ///   - candidateSchemas: Candidate schema names.
    ///   - toolNames: Optional app-owned tool names available to the model.
    /// - Returns: `[["role": "system|user", "content": "..."]]`
    func buildMessages(
        rawFact: RawFact,
        candidateSchemas: [String],
        toolNames: [String]
    ) -> [[String: String]]

    /// Parses model output into a classification result.
    func parse(
        llmOutput: String,
        candidateSchemas: [String]
    ) throws -> ClassificationResult
}
