// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import CoreImage
import CryptoKit
import EdgeEngine
import Foundation
import Tokenizers

public enum EngineState: Equatable, Sendable {
    case idle
    case loading
    case ready
    case generating
}

public struct GenerateChunk: Sendable {
    public let text: String
    public let generatedTokenCount: Int?

    public init(text: String, generatedTokenCount: Int? = nil) {
        self.text = text
        self.generatedTokenCount = generatedTokenCount
    }
}

public struct ChatMessage: Sendable {
    public enum Role: Sendable, Equatable {
        case system
        case user
        case assistant
        case tool
    }

    public let role: Role
    public let content: String

    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    public static func system(_ content: String) -> ChatMessage {
        .init(role: .system, content: content)
    }

    public static func user(_ content: String) -> ChatMessage {
        .init(role: .user, content: content)
    }

    public static func assistant(_ content: String) -> ChatMessage {
        .init(role: .assistant, content: content)
    }

    public static func tool(_ content: String) -> ChatMessage {
        .init(role: .tool, content: content)
    }

    public var roleString: String {
        switch role {
        case .system: return "system"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "tool"
        }
    }

    var tokenizerMessage: Message {
        [
            "role": roleString,
            "content": content,
        ]
    }

    func promptTemplateMessage(isHistorical: Bool, preserveThinking: Bool) -> Message {
        let promptContent = role == .assistant && isHistorical && !preserveThinking
            ? Self.strippingThinkingContent(from: content)
            : content
        return [
            "role": roleString,
            "content": promptContent,
        ]
    }

    static func strippingThinkingContent(from content: String) -> String {
        var result = ""
        var cursor = content.startIndex
        while let open = content.range(of: "<think>", range: cursor..<content.endIndex) {
            result += String(content[cursor..<open.lowerBound])
            guard let close = content.range(of: "</think>", range: open.upperBound..<content.endIndex) else {
                result += String(content[open.lowerBound...])
                return result
            }
            cursor = close.upperBound
        }
        result += String(content[cursor...])
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public extension Array where Element == ChatMessage {
    func toChatTemplateInput() -> String {
        map { message in
            "<|im_start|>\(message.roleString)\n\(message.content)<|im_end|>"
        }.joined(separator: "\n") + "\n<|im_start|>assistant\n"
    }

}

extension Array where Element == ChatMessage {
    func chatTemplateMessages(preserveThinking: Bool) -> [Message] {
        let lastUserIndex = lastIndex { $0.role == .user }
        return enumerated().map { index, message in
            let isHistorical = lastUserIndex.map { index < $0 } ?? true
            return message.promptTemplateMessage(
                isHistorical: isHistorical,
                preserveThinking: preserveThinking
            )
        }
    }

    func promptCacheMessages(preserveThinking: Bool) -> [ChatMessage] {
        let lastUserIndex = lastIndex { $0.role == .user }
        return enumerated().map { index, message in
            let isHistorical = lastUserIndex.map { index < $0 } ?? true
            guard message.role == .assistant, isHistorical, !preserveThinking else {
                return message
            }
            return ChatMessage(
                role: .assistant,
                content: ChatMessage.strippingThinkingContent(from: message.content)
            )
        }
    }
}

public struct EdgeGenerateParameters: Sendable {
    public var temperature: Float
    public var topK: Int?
    public var topP: Float
    public var minP: Float
    public var repetitionPenalty: Float
    public var repetitionContextSize: Int
    public var presencePenalty: Float
    public var presenceContextSize: Int
    public var frequencyPenalty: Float
    public var frequencyContextSize: Int
    public var maxTokens: Int
    public var quantizedKVStart: Int
    public var kvBits: Int?
    public var kvGroupSize: Int
    public var maxKVSize: Int?
    public var prefillStepSize: Int
    public var useDSR: Bool
    public var dsrMaxCritical: Int?
    public var dsrHeavyBudget: Int?
    public var dsrRecentBudget: Int?
    public var dsrScene: DSRSceneType
    public var dsrEvictionInterval: Int
    public var syncEval: Bool
    public var enableThinking: Bool
    public var preserveThinking: Bool
    public var frogJumpEnabled: Bool
    public var stopOnEndToken: Bool
    public var minimumGeneratedTokens: Int
    public var eosPenaltyUntilToken: Int

    public init(
        temperature: Float = 0.7,
        topK: Int? = 40,
        topP: Float = 0.9,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.1,
        repetitionContextSize: Int = 20,
        presencePenalty: Float = 0.0,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float = 0.0,
        frequencyContextSize: Int = 20,
        maxTokens: Int = 2048,
        quantizedKVStart: Int = 0,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        maxKVSize: Int? = nil,
        prefillStepSize: Int = 512,
        useDSR: Bool = false,
        dsrMaxCritical: Int? = nil,
        dsrHeavyBudget: Int? = nil,
        dsrRecentBudget: Int? = nil,
        dsrScene: DSRSceneType = .chat,
        dsrEvictionInterval: Int = 32,
        syncEval: Bool = false,
        enableThinking: Bool = false,
        preserveThinking: Bool = false,
        frogJumpEnabled: Bool = false,
        stopOnEndToken: Bool = true,
        minimumGeneratedTokens: Int = 0,
        eosPenaltyUntilToken: Int = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.maxTokens = maxTokens
        self.quantizedKVStart = quantizedKVStart
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.maxKVSize = maxKVSize
        self.prefillStepSize = prefillStepSize
        self.useDSR = useDSR
        self.dsrMaxCritical = dsrMaxCritical
        self.dsrHeavyBudget = dsrHeavyBudget
        self.dsrRecentBudget = dsrRecentBudget
        self.dsrScene = dsrScene
        self.dsrEvictionInterval = dsrEvictionInterval
        self.syncEval = syncEval
        self.enableThinking = enableThinking
        self.preserveThinking = preserveThinking
        self.frogJumpEnabled = frogJumpEnabled
        self.stopOnEndToken = stopOnEndToken
        self.minimumGeneratedTokens = minimumGeneratedTokens
        self.eosPenaltyUntilToken = eosPenaltyUntilToken
    }

    public static let `default` = EdgeGenerateParameters()

    public mutating func applyPolicy(_ policy: KVCacheMemoryPolicy) {
        guard policy.mode != .manual else { return }
        if quantizedKVStart == 0, policy.quantizeAtTokens > 0 {
            quantizedKVStart = policy.quantizeAtTokens
            kvBits = policy.kvBits
            kvGroupSize = policy.kvGroupSize
        }
        if useDSR || policy.useDSR {
            useDSR = true
            if dsrMaxCritical == nil { dsrMaxCritical = policy.dsrMaxCritical }
            if dsrHeavyBudget == nil { dsrHeavyBudget = policy.dsrHeavyBudget }
            if dsrRecentBudget == nil { dsrRecentBudget = policy.dsrRecentBudget }
        }
        if !useDSR, let maxKV = policy.maxKVSize, maxKVSize == nil {
            maxKVSize = maxKV
        }
        if policy.mode == .aggressive, prefillStepSize > 256 {
            prefillStepSize = 256
        }
        syncEval = policy.syncEval
    }
}

public struct NativeRuntimeLoadOptions: Sendable {
    public var maxOpsPerBuffer: Int?
    public var maxMBPerBuffer: Int?
    public var maxInFlightCommandBuffers: Int?
    public var memoryLimitBytes: Int?
    public var quantizedBufferCacheLimitBytes: Int?
    public var quantizedNoCopyBuffersEnabled: Bool?
    public var vendoredQuantizedMatmulEnabled: Bool?
    public var vendoredQuantizedPrefillMatmulEnabled: Bool?
    public var vendoredCommandBufferPrefillQMMEnabled: Bool?
    public var singleCommandBufferPrefillEnabled: Bool?
    public var singleCommandBufferDecodeEnabled: Bool?
    public var prefillLayerCommandBufferBatchingEnabled: Bool?
    public var fusedGDNDecodeEnabled: Bool?
    public var cmlxFastRMSNormEnabled: Bool?
    public var cmlxLazyOutputHeadEnabled: Bool?
    public var cmlxLazyDecodeEnabled: Bool?
    public var greedyOutputHeadArgmaxEnabled: Bool?
    public var dynamicOpsEnabled: Bool?
    public var dynamicOpsFloor: Int?
    public var dynamicOpsCtxLow: Int?
    public var dynamicOpsCtxHigh: Int?
    public var prefillStepSize: Int?
    public var syncEval: Bool?
    public var onlineCalibrationEnabled: Bool?
    public var memoryIntent: EdgeMemoryIntent?

    public init(
        maxOpsPerBuffer: Int? = nil,
        maxMBPerBuffer: Int? = nil,
        maxInFlightCommandBuffers: Int? = nil,
        memoryLimitBytes: Int? = nil,
        quantizedBufferCacheLimitBytes: Int? = nil,
        quantizedNoCopyBuffersEnabled: Bool? = nil,
        vendoredQuantizedMatmulEnabled: Bool? = nil,
        vendoredQuantizedPrefillMatmulEnabled: Bool? = nil,
        vendoredCommandBufferPrefillQMMEnabled: Bool? = nil,
        singleCommandBufferPrefillEnabled: Bool? = nil,
        singleCommandBufferDecodeEnabled: Bool? = nil,
        prefillLayerCommandBufferBatchingEnabled: Bool? = nil,
        fusedGDNDecodeEnabled: Bool? = nil,
        cmlxFastRMSNormEnabled: Bool? = nil,
        cmlxLazyOutputHeadEnabled: Bool? = nil,
        cmlxLazyDecodeEnabled: Bool? = nil,
        greedyOutputHeadArgmaxEnabled: Bool? = nil,
        dynamicOpsEnabled: Bool? = nil,
        dynamicOpsFloor: Int? = nil,
        dynamicOpsCtxLow: Int? = nil,
        dynamicOpsCtxHigh: Int? = nil,
        prefillStepSize: Int? = nil,
        syncEval: Bool? = nil,
        onlineCalibrationEnabled: Bool? = nil,
        memoryIntent: EdgeMemoryIntent? = nil
    ) {
        self.maxOpsPerBuffer = maxOpsPerBuffer
        self.maxMBPerBuffer = maxMBPerBuffer
        self.maxInFlightCommandBuffers = maxInFlightCommandBuffers
        self.memoryLimitBytes = memoryLimitBytes
        self.quantizedBufferCacheLimitBytes = quantizedBufferCacheLimitBytes
        self.quantizedNoCopyBuffersEnabled = quantizedNoCopyBuffersEnabled
        self.vendoredQuantizedMatmulEnabled = vendoredQuantizedMatmulEnabled
        self.vendoredQuantizedPrefillMatmulEnabled = vendoredQuantizedPrefillMatmulEnabled
        self.vendoredCommandBufferPrefillQMMEnabled = vendoredCommandBufferPrefillQMMEnabled
        self.singleCommandBufferPrefillEnabled = singleCommandBufferPrefillEnabled
        self.singleCommandBufferDecodeEnabled = singleCommandBufferDecodeEnabled
        self.prefillLayerCommandBufferBatchingEnabled = prefillLayerCommandBufferBatchingEnabled
        self.fusedGDNDecodeEnabled = fusedGDNDecodeEnabled
        self.cmlxFastRMSNormEnabled = cmlxFastRMSNormEnabled
        self.cmlxLazyOutputHeadEnabled = cmlxLazyOutputHeadEnabled
        self.cmlxLazyDecodeEnabled = cmlxLazyDecodeEnabled
        self.greedyOutputHeadArgmaxEnabled = greedyOutputHeadArgmaxEnabled
        self.dynamicOpsEnabled = dynamicOpsEnabled
        self.dynamicOpsFloor = dynamicOpsFloor
        self.dynamicOpsCtxLow = dynamicOpsCtxLow
        self.dynamicOpsCtxHigh = dynamicOpsCtxHigh
        self.prefillStepSize = prefillStepSize
        self.syncEval = syncEval
        self.onlineCalibrationEnabled = onlineCalibrationEnabled
        self.memoryIntent = memoryIntent
    }
}

public extension MemoryBudgetPlanner.Plan {
    func applying(_ options: NativeRuntimeLoadOptions?) -> MemoryBudgetPlanner.Plan {
        guard let options else {
            return self
        }
        return MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: options.maxOpsPerBuffer ?? maxOpsPerBuffer,
            maxMBPerBuffer: options.maxMBPerBuffer ?? maxMBPerBuffer,
            memoryLimitBytes: options.memoryLimitBytes ?? memoryLimitBytes,
            cacheLimitBytes: options.quantizedBufferCacheLimitBytes ?? cacheLimitBytes,
            forwardPassReserveMB: forwardPassReserveMB,
            kvBudgetMB: kvBudgetMB,
            dsrMaxCritical: dsrMaxCritical,
            quantizeAtTokens: quantizeAtTokens,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            dynamicOpsEnabled: options.dynamicOpsEnabled ?? dynamicOpsEnabled,
            dynamicOpsFloor: options.dynamicOpsFloor ?? dynamicOpsFloor,
            dynamicOpsCtxLow: options.dynamicOpsCtxLow ?? dynamicOpsCtxLow,
            dynamicOpsCtxHigh: options.dynamicOpsCtxHigh ?? dynamicOpsCtxHigh,
            prefillStepSize: options.prefillStepSize ?? prefillStepSize,
            syncEval: options.syncEval ?? syncEval,
            vendoredCommandBufferPrefillQMMEnabled: options.vendoredCommandBufferPrefillQMMEnabled ?? vendoredCommandBufferPrefillQMMEnabled,
            fusedGDNDecodeEnabled: options.fusedGDNDecodeEnabled ?? fusedGDNDecodeEnabled,
            mode: mode,
            reasoning: reasoning + options.reasoningSuffix,
            measuredBandwidthGBs: measuredBandwidthGBs,
            memoryIntent: memoryIntent
        )
    }
}

private extension NativeRuntimeLoadOptions {
    var reasoningSuffix: String {
        var parts: [String] = []
        if let maxOps = maxOpsPerBuffer {
            parts.append("override-maxOps=\(maxOps)")
        }
        if let maxMB = maxMBPerBuffer {
            parts.append("override-maxMB=\(maxMB)")
        }
        if let maxInFlight = maxInFlightCommandBuffers {
            parts.append("override-maxInflight=\(maxInFlight)")
        }
        if let memoryLimit = memoryLimitBytes {
            parts.append("override-memoryLimit=\(memoryLimit / 1_048_576)MB")
        }
        if let cache = quantizedBufferCacheLimitBytes {
            parts.append("override-qcache=\(cache / 1_048_576)MB")
        }
        if let noCopy = quantizedNoCopyBuffersEnabled {
            parts.append("override-qNoCopy=\(noCopy ? "on" : "off")")
        }
        if let qmm = vendoredQuantizedMatmulEnabled {
            parts.append("override-vendoredQMM=\(qmm ? "on" : "off")")
        }
        if let prefillQMM = vendoredQuantizedPrefillMatmulEnabled {
            parts.append("override-vendoredPrefillQMM=\(prefillQMM ? "on" : "off")")
        }
        if let commandBufferPrefillQMM = vendoredCommandBufferPrefillQMMEnabled {
            parts.append("override-vendoredCBPrefillQMM=\(commandBufferPrefillQMM ? "on" : "off")")
        }
        if let singleCBPrefill = singleCommandBufferPrefillEnabled {
            parts.append("override-singleCBPrefill=\(singleCBPrefill ? "on" : "off")")
        }
        if let singleCBDecode = singleCommandBufferDecodeEnabled {
            parts.append("override-singleCBDecode=\(singleCBDecode ? "on" : "off")")
        }
        if let prefillLayerCB = prefillLayerCommandBufferBatchingEnabled {
            parts.append("override-prefillLayerCB=\(prefillLayerCB ? "on" : "off")")
        }
        if let fusedGDNDecode = fusedGDNDecodeEnabled {
            parts.append("override-fusedGDNDecode=\(fusedGDNDecode ? "on" : "off")")
        }
        if let cmlxFastRMSNorm = cmlxFastRMSNormEnabled {
            parts.append("override-cmlxFastRMSNorm=\(cmlxFastRMSNorm ? "on" : "off")")
        }
        if let cmlxLazyOutputHead = cmlxLazyOutputHeadEnabled {
            parts.append("override-cmlxLazyOutputHead=\(cmlxLazyOutputHead ? "on" : "off")")
        }
        if let cmlxLazyDecode = cmlxLazyDecodeEnabled {
            parts.append("override-cmlxLazyDecode=\(cmlxLazyDecode ? "on" : "off")")
        }
        if let greedyOutputHeadArgmax = greedyOutputHeadArgmaxEnabled {
            parts.append("override-greedyOutputHeadArgmax=\(greedyOutputHeadArgmax ? "on" : "off")")
        }
        if let enabled = dynamicOpsEnabled {
            parts.append("override-dynOps=\(enabled ? "on" : "off")")
        }
        if let prefill = prefillStepSize {
            parts.append("override-prefill=\(prefill)")
        }
        if let syncEval {
            parts.append("override-syncEval=\(syncEval ? "on" : "off")")
        }
        if let onlineCalibrationEnabled {
            parts.append("onlineCalibration=\(onlineCalibrationEnabled ? "on" : "off")")
        }
        return parts.isEmpty ? "" : " | " + parts.joined(separator: " | ")
    }
}

public final class PromptCacheManager: @unchecked Sendable {
    private let lock = NSLock()
    private var cachedTokenCount: Int = 0
    private var cachedTokenPrefix: [Int32] = []
    private var enabled = true

    public var isEnabled: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return enabled
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            enabled = newValue
        }
    }
    nonisolated static let prefixCheckLength = 128

    public init() {}

    public var currentCache: [PromptCacheEntry]? {
        lock.lock()
        defer { lock.unlock() }
        guard enabled, cachedTokenCount > 0 else { return nil }
        return []
    }

    public func update(
        cache: [PromptCacheEntry],
        totalTokenCount: Int,
        tokenPrefix: [Int32]
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard enabled else { return }
        cachedTokenCount = totalTokenCount
        cachedTokenPrefix = Array(tokenPrefix.prefix(Self.prefixCheckLength))
    }

    public func update(cache: [PromptCacheEntry], totalTokenCount: Int) {
        update(cache: cache, totalTokenCount: totalTokenCount, tokenPrefix: [])
    }

    public var tokenCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cachedTokenCount
    }

    public var hasCache: Bool {
        lock.lock()
        defer { lock.unlock() }
        return enabled && cachedTokenCount > 0
    }

    public func validatePrefix(_ newTokenPrefix: [Int32]) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !cachedTokenPrefix.isEmpty else { return true }
        let checkLength = min(cachedTokenPrefix.count, newTokenPrefix.count)
        guard checkLength > 0 else { return false }
        return cachedTokenPrefix[..<checkLength] == newTokenPrefix[..<checkLength]
    }

    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cachedTokenCount = 0
        cachedTokenPrefix = []
    }
}

public struct PromptCacheEntry: Sendable {
    public init() {}
}

struct NativePromptSessionReuse {
    struct Match: Equatable {
        let cachedTokenLength: Int
        let promptTokenLength: Int
    }

    enum IncrementalSuffixResult: Equatable {
        case match(String)
        case reject(String)
    }

    static func reusablePrefixLength(
        cachedTokenIds: [Int],
        promptTokenIds: [Int]
    ) -> Int? {
        reusablePrefixMatch(
            cachedTokenIds: cachedTokenIds,
            promptTokenIds: promptTokenIds,
            skippableCachedTokenSequences: []
        )?.cachedTokenLength
    }

    static func reusablePrefixMatch(
        cachedTokenIds: [Int],
        promptTokenIds: [Int],
        skippableCachedTokenSequences: [[Int]]
    ) -> Match? {
        guard !cachedTokenIds.isEmpty else { return nil }

        var cachedIndex = 0
        var promptIndex = 0
        while cachedIndex < cachedTokenIds.count {
            if promptIndex < promptTokenIds.count,
               cachedTokenIds[cachedIndex] == promptTokenIds[promptIndex] {
                cachedIndex += 1
                promptIndex += 1
                continue
            }

            if let skippedCount = skippableCachedTokenSequences.firstMatchLength(
                in: cachedTokenIds,
                at: cachedIndex
            ) {
                cachedIndex += skippedCount
                continue
            }

            return nil
        }

        return Match(cachedTokenLength: cachedIndex, promptTokenLength: promptIndex)
    }

    /// Normalize assistant text by trimming trailing whitespace/newlines and
    /// removing Qwen's empty thinking sentinels. Token-prefix reuse already
    /// treats these sentinels as skippable, so text-suffix reuse must apply the
    /// same semantic normalization before comparing stored and current history.
    static func normalizeAssistantText(_ text: String) -> String {
        let withoutEmptyThinking = text
            .replacingOccurrences(of: "<think>\n\n</think>\n\n", with: "\n\n")
            .replacingOccurrences(of: "<think>\n</think>\n\n", with: "\n\n")
            .replacingOccurrences(of: "<think></think>\n\n", with: "\n\n")
            .replacingOccurrences(of: "<think></think>", with: "")
        return String(trimmingTrailingWhitespaceAndNewlines(withoutEmptyThinking))
    }

    static func qwenIncrementalSuffixText(
        previousPromptMessages: [ChatMessage],
        lastAssistantText: String,
        currentMessages: [ChatMessage],
        enableThinking: Bool,
        matchPath: String = "text_suffix"
    ) -> String? {
        guard case let .match(text) = qwenIncrementalSuffix(
            previousPromptMessages: previousPromptMessages,
            lastAssistantText: lastAssistantText,
            currentMessages: currentMessages,
            enableThinking: enableThinking,
            matchPath: matchPath
        ) else {
            return nil
        }
        return text
    }

    static func qwenIncrementalSuffix(
        previousPromptMessages: [ChatMessage],
        lastAssistantText: String,
        currentMessages: [ChatMessage],
        enableThinking: Bool,
        matchPath: String = "text_suffix"
    ) -> IncrementalSuffixResult {
        let assistantIndex = previousPromptMessages.count
        guard !previousPromptMessages.isEmpty else {
            return .reject("empty_previous_messages")
        }
        guard currentMessages.count > assistantIndex + 1 else {
            return .reject("current_too_short previous=\(previousPromptMessages.count) current=\(currentMessages.count)")
        }
        guard currentMessages.starts(withChatMessages: previousPromptMessages) else {
            return .reject("prompt_prefix_mismatch previous=\(previousPromptMessages.count) current=\(currentMessages.count)")
        }
        guard currentMessages[assistantIndex].role == .assistant else {
            return .reject("missing_assistant role=\(currentMessages[assistantIndex].roleString)")
        }
        let normalizedStored = normalizeAssistantText(lastAssistantText)
        let normalizedCurrent = normalizeAssistantText(currentMessages[assistantIndex].content)
        guard normalizedStored == normalizedCurrent else {
            return .reject(assistantTextMismatchDiagnostic(
                stored: normalizedStored,
                current: normalizedCurrent,
                matchPath: matchPath
            ))
        }

        let suffixMessages = currentMessages.dropFirst(assistantIndex + 1)
        guard suffixMessages.first?.role == .user,
              !suffixMessages.contains(where: { $0.role == .tool })
        else {
            return .reject("unsupported_suffix_roles count=\(suffixMessages.count)")
        }

        var text = "<|im_end|>\n"
        for message in suffixMessages {
            text += "<|im_start|>\(message.roleString)\n"
            text += message.content
            text += "<|im_end|>\n"
        }
        text += "<|im_start|>assistant\n"
        text += enableThinking ? "<think>\n" : "<think>\n\n</think>\n\n"
        return .match(text)
    }

    private static func assistantTextMismatchDiagnostic(
        stored: String,
        current: String,
        matchPath: String
    ) -> String {
        let diffIndex = firstDifferentCharacterIndex(stored, current)
        return "assistant_text_mismatch"
            + " matchPath=\(matchPath)"
            + " storedChars=\(stored.count)"
            + " currentChars=\(current.count)"
            + " storedHash=\(sha256Prefix(stored))"
            + " currentHash=\(sha256Prefix(current))"
            + " firstDiffIndex=\(diffIndex)"
            + " storedSnippet=\"\(escapedSnippet(around: diffIndex, in: stored))\""
            + " currentSnippet=\"\(escapedSnippet(around: diffIndex, in: current))\""
    }

    private static func firstDifferentCharacterIndex(_ stored: String, _ current: String) -> Int {
        var index = 0
        var storedIndex = stored.startIndex
        var currentIndex = current.startIndex
        while storedIndex < stored.endIndex, currentIndex < current.endIndex {
            guard stored[storedIndex] == current[currentIndex] else {
                return index
            }
            stored.formIndex(after: &storedIndex)
            current.formIndex(after: &currentIndex)
            index += 1
        }
        return index
    }

    private static func sha256Prefix(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .prefix(8)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func escapedSnippet(around characterIndex: Int, in text: String) -> String {
        guard !text.isEmpty else { return "" }
        let context = 20
        let lower = max(0, characterIndex - context)
        let upper = min(text.count, characterIndex + context)
        let start = text.index(text.startIndex, offsetBy: lower)
        let end = text.index(text.startIndex, offsetBy: upper)
        let prefix = lower > 0 ? "..." : ""
        let suffix = upper < text.count ? "..." : ""
        return logEscaped(prefix + String(text[start..<end]) + suffix)
    }

    private static func logEscaped(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\t", with: "\\t")
    }

    static func trimmingTrailingWhitespaceAndNewlines(_ text: String) -> Substring {
        var end = text.endIndex
        while end > text.startIndex {
            let previous = text.index(before: end)
            guard text[previous].unicodeScalars.allSatisfy({
                CharacterSet.whitespacesAndNewlines.contains($0)
            }) else {
                break
            }
            end = previous
        }
        return text[..<end]
    }
}

struct NativeCmlxAttentionCacheQuantization: Equatable {
    let groupSize: Int
    let bits: Int
}

private extension Array where Element == ChatMessage {
    func starts(withChatMessages prefix: [ChatMessage]) -> Bool {
        guard count >= prefix.count else { return false }
        for index in prefix.indices {
            guard self[index].role == prefix[index].role,
                  self[index].content == prefix[index].content
            else {
                return false
            }
        }
        return true
    }
}

private extension Array where Element == [Int] {
    func firstMatchLength(in tokens: [Int], at index: Int) -> Int? {
        for sequence in self where !sequence.isEmpty {
            guard index + sequence.count <= tokens.count else { continue }
            var matches = true
            for offset in sequence.indices where tokens[index + offset] != sequence[offset] {
                matches = false
                break
            }
            if matches {
                return sequence.count
            }
        }
        return nil
    }
}


public struct TranscriptionResult: @unchecked Sendable {
    public let text: String
    public let segments: [[String: Any]]?
    public let language: String?
    public let promptTokens: Int
    public let generationTokens: Int
    public let totalTokens: Int
    public let promptTps: Double
    public let generationTps: Double
    public let totalTime: Double
}

public enum STTStreamEvent: Sendable {
    case token(String)
    case info(STTStreamInfo)
    case result(TranscriptionResult)
}

public struct STTStreamInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let tokensPerSecond: Double
}

public enum TTSEvent: Sendable {
    case progress(Int)
    case audio(AudioResult)
    case audioChunk(AudioChunkResult)
}

public struct AudioChunkResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let chunkIndex: Int
    public let generationTimeMs: Int

    public var audioDuration: TimeInterval {
        Double(samples.count) / Double(sampleRate)
    }

    public var instantaneousRTF: Double {
        guard audioDuration > 0 else { return 0 }
        return Double(generationTimeMs) / 1000.0 / audioDuration
    }
}

public struct AudioResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public var duration: TimeInterval {
        Double(samples.count) / Double(sampleRate)
    }
}
