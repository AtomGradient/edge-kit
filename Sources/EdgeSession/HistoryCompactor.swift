// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

/// Generic chat-history compaction utility.
public struct HistoryCompactor: Sendable {
    public struct Config: Sendable, Equatable {
        public var maxMessages: Int
        public var characterBudget: Int
        public var preserveSystemPrompt: Bool
        public var preserveLastNTurns: Int

        public static let `default` = Config()

        public init(
            maxMessages: Int = 24,
            characterBudget: Int = 12_000,
            preserveSystemPrompt: Bool = true,
            preserveLastNTurns: Int = 2
        ) {
            self.maxMessages = maxMessages
            self.characterBudget = characterBudget
            self.preserveSystemPrompt = preserveSystemPrompt
            self.preserveLastNTurns = preserveLastNTurns
        }
    }

    public static func compact(
        _ messages: [ChatMessage],
        config: Config = .default
    ) -> [ChatMessage] {
        guard !messages.isEmpty else { return [] }
        let normalizedMaxMessages = max(1, config.maxMessages)
        let normalizedBudget = max(0, config.characterBudget)

        guard messages.count > normalizedMaxMessages || characterCount(messages) > normalizedBudget else {
            return messages
        }

        let system = config.preserveSystemPrompt && messages.first?.role == .system
            ? messages.first
            : nil
        let bodyStart = system == nil ? messages.startIndex : messages.index(after: messages.startIndex)
        let body = messages[bodyStart...].filter { $0.role != .system && $0.role != .tool }
        guard !body.isEmpty else {
            return system.map { [$0] } ?? []
        }

        let protected = protectedSuffix(
            from: body,
            preserveLastNTurns: max(0, config.preserveLastNTurns)
        )
        let protectedCount = protected.count
        var kept = Array(body.suffix(max(normalizedMaxMessages, protectedCount)))
        trimLeadingNonUserMessages(&kept, minimumCount: protectedCount)

        while kept.count > protectedCount
            && characterCount(assembled(system: system, kept: kept, dropped: body.count - kept.count)) > normalizedBudget {
            kept.removeFirst()
            trimLeadingNonUserMessages(&kept, minimumCount: protectedCount)
        }

        return assembled(system: system, kept: kept, dropped: body.count - kept.count)
    }

    private static func protectedSuffix(
        from messages: [ChatMessage],
        preserveLastNTurns: Int
    ) -> [ChatMessage] {
        guard preserveLastNTurns > 0 else { return [] }
        var userTurns = 0
        var startIndex = messages.startIndex

        for index in messages.indices.reversed() {
            if messages[index].role == .user {
                userTurns += 1
                if userTurns == preserveLastNTurns {
                    startIndex = index
                    break
                }
            }
            if index == messages.startIndex {
                startIndex = index
            }
        }

        return Array(messages[startIndex...])
    }

    private static func trimLeadingNonUserMessages(
        _ messages: inout [ChatMessage],
        minimumCount: Int
    ) {
        while messages.count > minimumCount, let first = messages.first, first.role != .user {
            messages.removeFirst()
        }
    }

    private static func assembled(
        system: ChatMessage?,
        kept: [ChatMessage],
        dropped: Int
    ) -> [ChatMessage] {
        var result: [ChatMessage] = []
        if let system {
            let suffix = dropped > 0
                ? "\n\n[Conversation summary] \(dropped) older messages were compacted to keep the local chat session inside the prompt budget."
                : ""
            result.append(.system(system.content + suffix))
        } else if dropped > 0 {
            result.append(.system("[Conversation summary] \(dropped) older messages were compacted to keep the local chat session inside the prompt budget."))
        }
        result.append(contentsOf: kept)
        return result
    }

    private static func characterCount(_ messages: [ChatMessage]) -> Int {
        messages.reduce(0) { $0 + $1.content.count }
    }
}

