// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Parser for classifier model output.
public enum ClassificationParser {

    /// Parses classifier output into a structured result.
    ///
    /// The parser accepts a JSON object directly or inside a Markdown code fence.
    /// It validates the selected schema, payload shape, and confidence range.
    public static func parse(llmOutput: String,
                              candidateSchemas: [String]) throws -> ClassificationResult {
        let cleaned = stripMarkdownCodeFence(llmOutput)

        guard let jsonString = extractJSONObject(cleaned) else {
            throw ClassificationParseError.invalidJSON("No JSON object found in output")
        }

        let jsonData = jsonString.data(using: .utf8) ?? Data()
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: jsonData)
        } catch {
            throw ClassificationParseError.invalidJSON(error.localizedDescription)
        }
        guard let dict = parsed as? [String: Any] else {
            throw ClassificationParseError.invalidJSON("JSON root is not an object")
        }

        guard let schema = dict["schema"] as? String, !schema.isEmpty else {
            throw ClassificationParseError.missingRequiredField("schema")
        }
        if schema == "__failed__" {
            throw ClassificationParseError.schemaIsFailed
        }
        guard candidateSchemas.contains(schema) else {
            throw ClassificationParseError.schemaNotInCandidates(schema, candidateSchemas)
        }
        guard let payload = dict["payload"] as? [String: Any] else {
            throw ClassificationParseError.missingRequiredField("payload (must be JSON object)")
        }
        guard let confidenceRaw = dict["confidence"] else {
            throw ClassificationParseError.missingRequiredField("confidence")
        }
        let confidence: Double
        if let d = confidenceRaw as? Double {
            confidence = d
        } else if let i = confidenceRaw as? Int {
            confidence = Double(i)
        } else if let n = confidenceRaw as? NSNumber {
            confidence = n.doubleValue
        } else {
            throw ClassificationParseError.payloadFieldTypeError(
                field: "confidence",
                expected: "number",
                got: "\(type(of: confidenceRaw))"
            )
        }
        guard confidence >= 0.0 && confidence <= 1.0 else {
            throw ClassificationParseError.confidenceOutOfRange(confidence)
        }

        let reasoning = dict["reasoning"] as? String

        return ClassificationResult(
            schema: schema,
            payload: payload,
            confidence: confidence,
            reasoning: reasoning
        )
    }

    /// Removes a surrounding Markdown code fence when present.
    public static func stripMarkdownCodeFence(_ s: String) -> String {
        var trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("```") {
            if let firstNewline = trimmed.firstIndex(of: "\n") {
                trimmed = String(trimmed[trimmed.index(after: firstNewline)...])
            }
            if trimmed.hasSuffix("```") {
                trimmed = String(trimmed.dropLast(3)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return trimmed
    }

    /// Extracts the first complete JSON object from mixed model output.
    public static func extractJSONObject(_ s: String) -> String? {
        guard let startIdx = s.firstIndex(of: "{") else { return nil }
        var braceCount = 0
        var inString = false
        var escapeNext = false
        var endIdx: String.Index?

        for i in s.indices[startIdx...] {
            let c = s[i]
            if escapeNext {
                escapeNext = false
                continue
            }
            if c == "\\" {
                escapeNext = true
                continue
            }
            if c == "\"" {
                inString.toggle()
                continue
            }
            if inString { continue }
            if c == "{" { braceCount += 1 }
            else if c == "}" {
                braceCount -= 1
                if braceCount == 0 {
                    endIdx = i
                    break
                }
            }
        }
        guard let end = endIdx else { return nil }
        return String(s[startIdx...end])
    }
}
