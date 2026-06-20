// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum ToolCallTextParser {
    public static func containsToolCall(_ text: String) -> Bool {
        text.contains("<tool_call>")
    }

    public static func toolCalls(in text: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var searchStart = text.startIndex

        while let openRange = text.range(of: "<tool_call>", range: searchStart..<text.endIndex),
              let closeRange = text.range(of: "</tool_call>", range: openRange.upperBound..<text.endIndex) {
            let block = String(text[openRange.upperBound..<closeRange.lowerBound])
            if let call = parseJSONToolCall(block) ?? parseXMLToolCall(block) {
                calls.append(call)
            }
            searchStart = closeRange.upperBound
        }

        return calls
    }

    private static func parseXMLToolCall(_ block: String) -> ToolCall? {
        guard let functionOpen = block.range(of: "<function="),
              let functionNameEnd = block.range(of: ">", range: functionOpen.upperBound..<block.endIndex),
              let functionClose = block.range(of: "</function>", range: functionNameEnd.upperBound..<block.endIndex)
        else {
            return nil
        }

        let rawName = String(block[functionOpen.upperBound..<functionNameEnd.lowerBound])
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }

        let functionBody = String(block[functionNameEnd.upperBound..<functionClose.lowerBound])
        var arguments: [String: any Sendable] = [:]
        var searchStart = functionBody.startIndex

        while let parameterOpen = functionBody.range(of: "<parameter=", range: searchStart..<functionBody.endIndex),
              let parameterNameEnd = functionBody.range(of: ">", range: parameterOpen.upperBound..<functionBody.endIndex),
              let parameterClose = functionBody.range(of: "</parameter>", range: parameterNameEnd.upperBound..<functionBody.endIndex) {
            let rawParameterName = String(functionBody[parameterOpen.upperBound..<parameterNameEnd.lowerBound])
            let parameterName = rawParameterName.trimmingCharacters(in: .whitespacesAndNewlines)
            if !parameterName.isEmpty {
                let rawValue = String(functionBody[parameterNameEnd.upperBound..<parameterClose.lowerBound])
                arguments[parameterName] = parseArgumentValue(rawValue)
            }
            searchStart = parameterClose.upperBound
        }

        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    private static func parseJSONToolCall(_ block: String) -> ToolCall? {
        let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("{"),
              let data = trimmed.data(using: .utf8)
        else {
            return nil
        }

        if let call = try? JSONDecoder().decode(ToolCall.self, from: data) {
            return call
        }
        if let call = try? JSONDecoder().decode(FlatToolCall.self, from: data) {
            return ToolCall(function: .init(name: call.name, arguments: call.arguments))
        }
        return nil
    }

    private static func parseArgumentValue(_ rawValue: String) -> any Sendable {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        if let data = trimmed.data(using: .utf8),
           let first = trimmed.first,
           first == "{" || first == "[" || first == "\"" ||
            first == "t" || first == "f" || first == "n",
           let value = try? JSONDecoder().decode(ToolCallJSONValue.self, from: data) {
            return value.sendableValue
        }

        if let intValue = Int(trimmed), String(intValue) == trimmed {
            return intValue
        }
        if let doubleValue = Double(trimmed),
           trimmed.contains(".") || trimmed.contains("e") || trimmed.contains("E") {
            return doubleValue
        }
        return trimmed
    }

    private struct FlatToolCall: Decodable {
        var name: String
        var arguments: [String: any Sendable]

        private enum CodingKeys: String, CodingKey {
            case name
            case arguments
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            let decoded = try container.decodeIfPresent(
                [String: ToolCallJSONValue].self,
                forKey: .arguments
            ) ?? [:]
            arguments = decoded.mapValues(\.sendableValue)
        }
    }
}
