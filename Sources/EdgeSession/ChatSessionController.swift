// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import CoreImage
import EdgeInference
import Foundation

public enum EdgeChatSessionError: LocalizedError, Equatable {
    case timeout(seconds: TimeInterval)

    public var errorDescription: String? {
        switch self {
        case .timeout(let seconds):
            return "Chat generation timed out after \(Int(seconds))s"
        }
    }
}

public struct EdgeChatSessionEvent: Sendable {
    public let mode: ChatSessionController.Mode
    public let resetReason: String?
    public let historyMessages: Int
    public let historyCharacters: Int
    public let memoryPolicyCompaction: ChatSessionMemoryPolicy.CompactionAudit?
    public let memoryPolicyQuality: MemoryPolicyQualitySignalRecorder.Result?
    public let metrics: InferenceMetrics?

    public init(
        mode: ChatSessionController.Mode,
        resetReason: String?,
        historyMessages: Int,
        historyCharacters: Int,
        memoryPolicyCompaction: ChatSessionMemoryPolicy.CompactionAudit? = nil,
        memoryPolicyQuality: MemoryPolicyQualitySignalRecorder.Result? = nil,
        metrics: InferenceMetrics?
    ) {
        self.mode = mode
        self.resetReason = resetReason
        self.historyMessages = historyMessages
        self.historyCharacters = historyCharacters
        self.memoryPolicyCompaction = memoryPolicyCompaction
        self.memoryPolicyQuality = memoryPolicyQuality
        self.metrics = metrics
    }
}

@MainActor
public final class ChatSessionController: ObservableObject {
    public enum Mode: Equatable, Sendable {
        case plain
        case image
        case tool
        case isolated(String)
    }

    @Published public private(set) var history: [ChatMessage]
    @Published public private(set) var isGenerating: Bool
    @Published public private(set) var lastMetrics: InferenceMetrics?
    @Published public private(set) var lastEvent: EdgeChatSessionEvent?

    private let client: any EdgeGenerationClient
    private var currentMode: Mode?
    private var cancelRequested = false
    private let maxHistoryMessages: Int
    private let historyCharacterBudget: Int

    public init(
        client: any EdgeGenerationClient,
        maxHistoryMessages: Int = 24,
        historyCharacterBudget: Int = 12_000
    ) {
        self.client = client
        self.maxHistoryMessages = maxHistoryMessages
        self.historyCharacterBudget = historyCharacterBudget
        self.history = []
        self.isGenerating = false
        self.lastMetrics = nil
        self.lastEvent = nil
    }

    public func runTurn(
        userText: String,
        systemPrompt: String,
        mode: Mode = .plain,
        images: [CIImage] = [],
        tools: [EdgeSessionToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters? = nil,
        memoryPolicy: ChatSessionMemoryPolicy? = nil,
        memoryQualityObservation: MemoryPolicyQualitySignalRecorder.Observation? = nil,
        timeoutSeconds: TimeInterval? = nil,
        watchdogConfiguration: EdgeGenerationWatchdog.Configuration? = nil,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String {
        prepareHistory(systemPrompt: systemPrompt, mode: mode)
        history.append(.user(userText))
        do {
            return try await generatePrepared(
                messages: history,
                mode: mode,
                images: images,
                tools: tools,
                onToolCall: onToolCall,
                parameters: parameters,
                memoryPolicy: memoryPolicy,
                memoryQualityObservation: memoryQualityObservation,
                timeoutSeconds: timeoutSeconds,
                watchdogConfiguration: watchdogConfiguration,
                onChunk: onChunk
            )
        } catch {
            if history.last?.role == .user {
                history.removeLast()
            }
            throw error
        }
    }

    public func reset(systemPrompt: String? = nil, reason: String) {
        history.removeAll(keepingCapacity: true)
        if let systemPrompt {
            history.append(.system(systemPrompt))
        }
        currentMode = nil
        lastMetrics = nil
        lastEvent = EdgeChatSessionEvent(
            mode: .isolated("reset"),
            resetReason: reason,
            historyMessages: history.count,
            historyCharacters: historyCharacterCount(history),
            metrics: nil
        )
        Task { await client.resetRuntime(reason: reason) }
    }

    public func cancel(reason: String) {
        cancelRequested = true
        isGenerating = false
        Task { await client.resetRuntime(reason: reason) }
    }

    public func replaceHistory(_ messages: [ChatMessage], mode: Mode? = nil) {
        history = HistoryCompactor.compact(
            messages,
            config: compactorConfig
        )
        currentMode = mode
    }

    public func generatePrepared(
        messages: [ChatMessage],
        mode: Mode,
        images: [CIImage] = [],
        tools: [EdgeSessionToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters? = nil,
        memoryPolicy: ChatSessionMemoryPolicy? = nil,
        memoryQualityObservation: MemoryPolicyQualitySignalRecorder.Observation? = nil,
        timeoutSeconds: TimeInterval? = nil,
        watchdogConfiguration: EdgeGenerationWatchdog.Configuration? = nil,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String {
        try Task.checkCancellation()
        let resetReason = await enterMode(mode)
        let baseCompactorConfig = compactorConfig
        let effectiveCompactorConfig = memoryPolicy?.compactorConfig(base: baseCompactorConfig)
            ?? baseCompactorConfig
        let preparedMessages = HistoryCompactor.compact(
            messages,
            config: effectiveCompactorConfig
        )
        let memoryPolicyCompaction = memoryPolicy?.compactionAudit(
            base: baseCompactorConfig,
            effective: effectiveCompactorConfig,
            compactedHistory: messagesDiffer(messages, preparedMessages),
            sourceCharacterCount: historyCharacterCount(messages),
            preparedCharacterCount: historyCharacterCount(preparedMessages)
        )
        history = preparedMessages
        cancelRequested = false
        isGenerating = true
        defer { isGenerating = false }

        var output = ""
        let finalOutput: String
        if let watchdogConfiguration {
            do {
                let result = try await EdgeGenerationWatchdog.run(
                    client: client,
                    messages: preparedMessages,
                    ciImages: images,
                    tools: tools,
                    onToolCall: onToolCall,
                    parameters: parameters,
                    configuration: watchdogConfiguration,
                    scenarioID: mode.logName,
                    turnIndex: history.count,
                    diagnosticSink: { message in
                        NSLog("[EdgeGenerationWatchdog] \(message)")
                    }
                ) { [self] chunk in
                    if self.cancelRequested {
                        return
                    }
                    output += chunk
                    onChunk(chunk)
                }
                if let errorDescription = result.errorDescription, result.endReason == .generationError {
                    throw NSError(
                        domain: "EdgeSession.EdgeGenerationWatchdog",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: errorDescription]
                    )
                }
                finalOutput = output.isEmpty ? result.text : output
            } catch let error as EdgeGenerationWatchdog.TimeoutError {
                throw EdgeChatSessionError.timeout(seconds: error.configuration.turnTimeoutSeconds)
            }
        } else {
            let deadline = timeoutSeconds.map { Date().addingTimeInterval($0) }
            let reply = try await client.generate(
                messages: preparedMessages,
                ciImages: images,
                tools: tools,
                onToolCall: onToolCall,
                parameters: parameters
            ) { [self] chunk in
                if self.cancelRequested {
                    return
                }
                if let deadline, Date() >= deadline {
                    return
                }
                output += chunk
                onChunk(chunk)
            }
            if let deadline, Date() >= deadline {
                await client.resetRuntime(reason: "timeout")
                throw EdgeChatSessionError.timeout(seconds: timeoutSeconds ?? 0)
            }
            finalOutput = output.isEmpty ? reply : output
        }

        try Task.checkCancellation()
        if cancelRequested {
            throw CancellationError()
        }

        let memoryPolicyQuality = memoryPolicy.map { policy in
            MemoryPolicyQualitySignalRecorder().record(
                memoryPlan: policy.plan,
                observation: mergedMemoryQualityObservation(
                    memoryQualityObservation,
                    compactionAudit: memoryPolicyCompaction
                )
            )
        }
        history = preparedMessages + [.assistant(finalOutput)]
        captureMetrics(
            mode: mode,
            resetReason: resetReason,
            memoryPolicyCompaction: memoryPolicyCompaction,
            memoryPolicyQuality: memoryPolicyQuality
        )
        return finalOutput
    }

    private var compactorConfig: HistoryCompactor.Config {
        .init(
            maxMessages: maxHistoryMessages,
            characterBudget: historyCharacterBudget
        )
    }

    private func prepareHistory(systemPrompt: String, mode: Mode) {
        if history.isEmpty {
            history.append(.system(systemPrompt))
        } else if history.first?.role == .system {
            history[0] = .system(systemPrompt)
        } else {
            history.insert(.system(systemPrompt), at: 0)
        }
        _ = mode
    }

    @discardableResult
    private func enterMode(_ mode: Mode) async -> String? {
        guard currentMode != mode else { return nil }
        let reason = currentMode == nil ? "session_start" : "mode_changed"
        currentMode = mode
        await client.resetRuntime(reason: reason)
        return reason
    }

    private func captureMetrics(
        mode: Mode,
        resetReason: String?,
        memoryPolicyCompaction: ChatSessionMemoryPolicy.CompactionAudit?,
        memoryPolicyQuality: MemoryPolicyQualitySignalRecorder.Result?
    ) {
        let metrics = client.currentInferenceMetrics
        lastMetrics = metrics
        lastEvent = EdgeChatSessionEvent(
            mode: mode,
            resetReason: resetReason,
            historyMessages: history.count,
            historyCharacters: historyCharacterCount(history),
            memoryPolicyCompaction: memoryPolicyCompaction,
            memoryPolicyQuality: memoryPolicyQuality,
            metrics: metrics
        )
    }

    private func mergedMemoryQualityObservation(
        _ observation: MemoryPolicyQualitySignalRecorder.Observation?,
        compactionAudit: ChatSessionMemoryPolicy.CompactionAudit?
    ) -> MemoryPolicyQualitySignalRecorder.Observation {
        let compactionApplied = observation?.compactionApplied == true
            || compactionAudit?.compactedHistory == true
        guard let observation else {
            return .init(compactionApplied: compactionApplied)
        }
        return .init(
            sessionFingerprint: observation.sessionFingerprint,
            turnFingerprint: observation.turnFingerprint,
            compactionApplied: compactionApplied,
            recallAttempted: observation.recallAttempted,
            recallSucceeded: observation.recallSucceeded,
            toolRecallAttempted: observation.toolRecallAttempted,
            toolRecallSucceeded: observation.toolRecallSucceeded,
            fallbackOccurred: observation.fallbackOccurred,
            userCorrectionObserved: observation.userCorrectionObserved,
            postCompactionProbeCompleted: observation.postCompactionProbeCompleted
        )
    }

    private func historyCharacterCount(_ messages: [ChatMessage]) -> Int {
        messages.reduce(0) { $0 + $1.content.count }
    }

    private func messagesDiffer(_ lhs: [ChatMessage], _ rhs: [ChatMessage]) -> Bool {
        guard lhs.count == rhs.count else { return true }
        return zip(lhs, rhs).contains { left, right in
            left.role != right.role || left.content != right.content
        }
    }
}

public extension ChatSessionController.Mode {
    var logName: String {
        switch self {
        case .plain:
            return "plain"
        case .image:
            return "image"
        case .tool:
            return "tool"
        case .isolated(let name):
            return "isolated:\(name)"
        }
    }
}
