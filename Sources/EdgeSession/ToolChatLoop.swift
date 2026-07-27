// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

public enum ToolChatLoop {
    public struct PlannedToolCall {
        public var plan: ToolCallPlan
        public var source: String
        public var reason: String?

        public init(
            plan: ToolCallPlan,
            source: String = "planned",
            reason: String? = nil
        ) {
            self.plan = plan
            self.source = source
            self.reason = reason
        }
    }

    public struct ToolResult {
        public var name: String
        public var result: String
        public var source: String

        public init(name: String, result: String, source: String) {
            self.name = name
            self.result = result
            self.source = source
        }
    }

    public struct Request {
        public var messages: [ChatMessage]
        public var mode: ChatSessionController.Mode
        public var tools: [EdgeSessionToolSpec]
        public var allowedToolNames: [String]
        public var plannedToolCalls: [PlannedToolCall]
        public var maxRounds: Int
        public var parameters: EdgeGenerateParameters?
        public var timeoutSeconds: TimeInterval?
        public var watchdogConfiguration: EdgeGenerationWatchdog.Configuration?
        public var emptyFinalText: String
        public var suppressNextChunkAfterToolCall: Bool

        public init(
            messages: [ChatMessage],
            mode: ChatSessionController.Mode,
            tools: [EdgeSessionToolSpec],
            allowedToolNames: [String],
            plannedToolCalls: [PlannedToolCall] = [],
            maxRounds: Int = 3,
            parameters: EdgeGenerateParameters? = nil,
            timeoutSeconds: TimeInterval? = nil,
            watchdogConfiguration: EdgeGenerationWatchdog.Configuration? = nil,
            emptyFinalText: String = "",
            suppressNextChunkAfterToolCall: Bool = true
        ) {
            self.messages = messages
            self.mode = mode
            self.tools = tools
            self.allowedToolNames = allowedToolNames
            self.plannedToolCalls = plannedToolCalls
            self.maxRounds = max(0, maxRounds)
            self.parameters = parameters
            self.timeoutSeconds = timeoutSeconds
            self.watchdogConfiguration = watchdogConfiguration
            self.emptyFinalText = emptyFinalText
            self.suppressNextChunkAfterToolCall = suppressNextChunkAfterToolCall
        }
    }

    public struct Hooks {
        public var executePlannedTool: (ToolCallPlan) async throws -> String
        public var executeModelTool: (ToolCall) async throws -> String
        public var summarizeToolResults: ([ToolResult]) -> String
        public var streamSummary: (String, @escaping @MainActor (String) -> Void) async -> Void
        public var fallbackWhenToolsUnavailable: (() async throws -> String)?
        public var onToolsSelected: ([String], Int, [PlannedToolCall]) async -> Void
        public var onToolsUnavailable: ([String], Int) async -> Void
        public var onPlannedToolCall: (PlannedToolCall) async -> Void
        public var onModelToolCall: (ToolCall) async -> Void
        public var onToolResult: (ToolResult) async -> Void
        public var onToolSummary: (String, [ToolResult]) async -> Void
        public var onPlannedToolSummaryEmpty: (PlannedToolCall, ToolResult) async -> Void
        public var onRoundWillContinue: ([ToolResult]) async -> Void
        public var onRoundMetrics: (Int, InferenceMetrics?) async -> Void
        public var disallowedToolResult: (String) -> String
        public var toolExecutionErrorResult: (String, Error) -> String

        public init(
            executePlannedTool: @escaping (ToolCallPlan) async throws -> String,
            executeModelTool: @escaping (ToolCall) async throws -> String,
            summarizeToolResults: @escaping ([ToolResult]) -> String,
            streamSummary: @escaping (String, @escaping @MainActor (String) -> Void) async -> Void = ToolChatLoop.defaultStreamSummary,
            fallbackWhenToolsUnavailable: (() async throws -> String)? = nil,
            onToolsSelected: @escaping ([String], Int, [PlannedToolCall]) async -> Void = { _, _, _ in },
            onToolsUnavailable: @escaping ([String], Int) async -> Void = { _, _ in },
            onPlannedToolCall: @escaping (PlannedToolCall) async -> Void = { _ in },
            onModelToolCall: @escaping (ToolCall) async -> Void = { _ in },
            onToolResult: @escaping (ToolResult) async -> Void = { _ in },
            onToolSummary: @escaping (String, [ToolResult]) async -> Void = { _, _ in },
            onPlannedToolSummaryEmpty: @escaping (PlannedToolCall, ToolResult) async -> Void = { _, _ in },
            onRoundWillContinue: @escaping ([ToolResult]) async -> Void = { _ in },
            onRoundMetrics: @escaping (Int, InferenceMetrics?) async -> Void = { _, _ in },
            disallowedToolResult: @escaping (String) -> String = ToolChatLoop.defaultDisallowedToolResult,
            toolExecutionErrorResult: @escaping (String, Error) -> String = ToolChatLoop.defaultToolExecutionErrorResult
        ) {
            self.executePlannedTool = executePlannedTool
            self.executeModelTool = executeModelTool
            self.summarizeToolResults = summarizeToolResults
            self.streamSummary = streamSummary
            self.fallbackWhenToolsUnavailable = fallbackWhenToolsUnavailable
            self.onToolsSelected = onToolsSelected
            self.onToolsUnavailable = onToolsUnavailable
            self.onPlannedToolCall = onPlannedToolCall
            self.onModelToolCall = onModelToolCall
            self.onToolResult = onToolResult
            self.onToolSummary = onToolSummary
            self.onPlannedToolSummaryEmpty = onPlannedToolSummaryEmpty
            self.onRoundWillContinue = onRoundWillContinue
            self.onRoundMetrics = onRoundMetrics
            self.disallowedToolResult = disallowedToolResult
            self.toolExecutionErrorResult = toolExecutionErrorResult
        }
    }

    @MainActor
    public static func run(
        session: ChatSessionController,
        request: Request,
        hooks: Hooks,
        onChunk: @escaping @MainActor (String) -> Void
    ) async throws -> String {
        await hooks.onToolsSelected(
            request.allowedToolNames,
            request.tools.count,
            request.plannedToolCalls
        )

        guard !request.tools.isEmpty else {
            await hooks.onToolsUnavailable(request.allowedToolNames, request.tools.count)
            if let fallback = hooks.fallbackWhenToolsUnavailable {
                return try await fallback()
            }
            return request.emptyFinalText
        }

        var currentMessages = request.messages
        var finalText = ""
        let allowedToolNames = Set(request.allowedToolNames)

        for planned in request.plannedToolCalls where allowedToolNames.contains(planned.plan.toolName) {
            await hooks.onPlannedToolCall(planned)
            let output = try await hooks.executePlannedTool(planned.plan)
            let result = ToolResult(
                name: planned.plan.toolName,
                result: output,
                source: planned.source
            )
            await hooks.onToolResult(result)
            if let summary = nonEmptySummary(
                hooks.summarizeToolResults([result])
            ) {
                await emitSummary(summary, results: [result], hooks: hooks, onChunk: onChunk)
                return summary
            }
            await hooks.onPlannedToolSummaryEmpty(planned, result)
        }

        for roundIndex in 0..<request.maxRounds {
            var roundText = ""
            let roundState = ToolChatRoundState()

            _ = try await session.generatePrepared(
                messages: currentMessages,
                mode: request.mode,
                tools: request.tools,
                onToolCall: { toolCall in
                    await hooks.onModelToolCall(toolCall)
                    let resultText: String
                    if allowedToolNames.contains(toolCall.function.name) {
                        do {
                            resultText = try await hooks.executeModelTool(toolCall)
                        } catch {
                            resultText = hooks.toolExecutionErrorResult(toolCall.function.name, error)
                        }
                    } else {
                        resultText = hooks.disallowedToolResult(toolCall.function.name)
                    }
                    let result = ToolResult(
                        name: toolCall.function.name,
                        result: resultText,
                        source: "model_generated"
                    )
                    await hooks.onToolResult(result)
                    roundState.append(
                        result,
                        suppressNextChunk: request.suppressNextChunkAfterToolCall
                    )
                    return resultText
                },
                parameters: request.parameters,
                timeoutSeconds: request.timeoutSeconds,
                watchdogConfiguration: request.watchdogConfiguration,
                onChunk: { chunk in
                    if roundState.consumeNextChunkSuppression() {
                        return
                    }
                    roundText += chunk
                    onChunk(chunk)
                }
            )
            await hooks.onRoundMetrics(roundIndex, session.lastMetrics)

            finalText = roundText.trimmingCharacters(in: .whitespacesAndNewlines)
            let toolCallsThisRound = roundState.toolResults()
            guard !toolCallsThisRound.isEmpty else {
                return finalText
            }

            if let summary = nonEmptySummary(
                hooks.summarizeToolResults(toolCallsThisRound)
            ) {
                await emitSummary(summary, results: toolCallsThisRound, hooks: hooks, onChunk: onChunk)
                return summary
            }

            compactConsumedToolMessages(&currentMessages)
            if !finalText.isEmpty {
                currentMessages.append(.assistant(finalText))
            }
            currentMessages.append(contentsOf: toolCallsThisRound.map {
                .tool("[\($0.name)] \($0.result)")
            })
            await hooks.onRoundWillContinue(toolCallsThisRound)
        }

        return finalText.isEmpty ? request.emptyFinalText : finalText
    }

    public static func compactConsumedToolMessages(_ messages: inout [ChatMessage]) {
        for index in messages.indices where messages[index].role == .tool {
            let content = messages[index].content
            guard content.hasPrefix("["),
                  let nameEnd = content.firstIndex(of: "]") else {
                continue
            }
            let name = String(content[content.index(after: content.startIndex)..<nameEnd])
            messages[index] = .tool("[\(name)] (consumed in earlier round)")
        }
    }

    @MainActor
    public static func defaultStreamSummary(
        _ summary: String,
        onChunk: @escaping @MainActor (String) -> Void
    ) async {
        onChunk(summary)
    }

    nonisolated public static func defaultDisallowedToolResult(_ toolName: String) -> String {
        "{\"error\":\"tool '\\(toolName)' was not selected for this route\"}"
    }

    nonisolated public static func defaultToolExecutionErrorResult(
        _ toolName: String,
        error: Error
    ) -> String {
        let message = "tool '\(toolName)' failed: \(toolExecutionErrorMessage(error))"
        return errorJSON(message)
    }

    @MainActor
    private static func emitSummary(
        _ summary: String,
        results: [ToolResult],
        hooks: Hooks,
        onChunk: @escaping @MainActor (String) -> Void
    ) async {
        await hooks.onToolSummary(summary, results)
        await hooks.streamSummary(summary, onChunk)
    }

    private static func nonEmptySummary(_ summary: String) -> String? {
        let trimmed = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : summary
    }

    private static func toolExecutionErrorMessage(_ error: Error) -> String {
        if let decodingError = error as? DecodingError {
            return decodingErrorMessage(decodingError)
        }
        if let conversionError = error as? ToolArgumentConversionError {
            switch conversionError {
            case .objectUnsupported(let path):
                return "unsupported nested object argument at `\(path)`"
            case .valueUnsupported(let path):
                return "unsupported argument value at `\(path)`"
            }
        }
        if let registryError = error as? ToolRegistryError {
            switch registryError {
            case .toolNotFound(let name):
                return "tool '\(name)' is not registered"
            case .outputEncodingFailed:
                return "tool output could not be encoded"
            }
        }

        let localized = error.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        return localized.isEmpty ? String(describing: error) : localized
    }

    private static func decodingErrorMessage(_ error: DecodingError) -> String {
        switch error {
        case .dataCorrupted(let context):
            return contextMessage(context)
        case .keyNotFound(let key, let context):
            return "missing key `\(key.stringValue)`: \(contextMessage(context))"
        case .typeMismatch(let type, let context):
            return "type mismatch for \(type): \(contextMessage(context))"
        case .valueNotFound(let type, let context):
            return "missing value for \(type): \(contextMessage(context))"
        @unknown default:
            return String(describing: error)
        }
    }

    private static func contextMessage(_ context: DecodingError.Context) -> String {
        let message = context.debugDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !context.codingPath.isEmpty else {
            return message.isEmpty ? "decoding failed" : message
        }
        let path = context.codingPath.map(\.stringValue).joined(separator: ".")
        if message.isEmpty {
            return "decoding failed at `\(path)`"
        }
        return "\(message) at `\(path)`"
    }

    private static func errorJSON(_ message: String) -> String {
        let payload = ["error": message]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            return text
        }
        let escaped = message
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "{\"error\":\"\(escaped)\"}"
    }
}

public extension ToolChatLoop.Hooks {
    /// Build hooks that execute allowed tool calls against a caller-provided registry.
    ///
    /// This helper does not register tools, select tools, generate tool calls, or
    /// bypass `ToolChatLoop.Request.allowedToolNames`; it only supplies execution
    /// closures for runtimes that already opted into a registry and request.
    static func registryBacked(
        registry: ToolRegistry,
        summarizeToolResults: @escaping ([ToolChatLoop.ToolResult]) -> String,
        streamSummary: @escaping (
            String,
            @escaping @MainActor (String) -> Void
        ) async -> Void = ToolChatLoop.defaultStreamSummary,
        fallbackWhenToolsUnavailable: (() async throws -> String)? = nil,
        onToolsSelected: @escaping ([String], Int, [ToolChatLoop.PlannedToolCall]) async -> Void = { _, _, _ in },
        onToolsUnavailable: @escaping ([String], Int) async -> Void = { _, _ in },
        onPlannedToolCall: @escaping (ToolChatLoop.PlannedToolCall) async -> Void = { _ in },
        onModelToolCall: @escaping (ToolCall) async -> Void = { _ in },
        onToolResult: @escaping (ToolChatLoop.ToolResult) async -> Void = { _ in },
        onToolSummary: @escaping (String, [ToolChatLoop.ToolResult]) async -> Void = { _, _ in },
        onPlannedToolSummaryEmpty: @escaping (
            ToolChatLoop.PlannedToolCall,
            ToolChatLoop.ToolResult
        ) async -> Void = { _, _ in },
        onRoundWillContinue: @escaping ([ToolChatLoop.ToolResult]) async -> Void = { _ in },
        disallowedToolResult: @escaping (String) -> String = ToolChatLoop.defaultDisallowedToolResult,
        toolExecutionErrorResult: @escaping (String, Error) -> String = ToolChatLoop.defaultToolExecutionErrorResult
    ) -> Self {
        Self(
            executePlannedTool: { plan in
                try await registry.execute(plan)
            },
            executeModelTool: { toolCall in
                try await registry.execute(toolCall)
            },
            summarizeToolResults: summarizeToolResults,
            streamSummary: streamSummary,
            fallbackWhenToolsUnavailable: fallbackWhenToolsUnavailable,
            onToolsSelected: onToolsSelected,
            onToolsUnavailable: onToolsUnavailable,
            onPlannedToolCall: onPlannedToolCall,
            onModelToolCall: onModelToolCall,
            onToolResult: onToolResult,
            onToolSummary: onToolSummary,
            onPlannedToolSummaryEmpty: onPlannedToolSummaryEmpty,
            onRoundWillContinue: onRoundWillContinue,
            disallowedToolResult: disallowedToolResult,
            toolExecutionErrorResult: toolExecutionErrorResult
        )
    }
}

private final class ToolChatRoundState: @unchecked Sendable {
    private let lock = NSLock()
    private var results: [ToolChatLoop.ToolResult] = []
    private var shouldSuppressNextChunk = false

    func append(_ result: ToolChatLoop.ToolResult, suppressNextChunk: Bool) {
        lock.lock()
        results.append(result)
        shouldSuppressNextChunk = shouldSuppressNextChunk || suppressNextChunk
        lock.unlock()
    }

    func consumeNextChunkSuppression() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard shouldSuppressNextChunk else { return false }
        shouldSuppressNextChunk = false
        return true
    }

    func toolResults() -> [ToolChatLoop.ToolResult] {
        lock.lock()
        defer { lock.unlock() }
        return results
    }
}
