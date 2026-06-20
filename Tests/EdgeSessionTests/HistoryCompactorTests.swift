// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
@testable import EdgeSession
import XCTest

final class HistoryCompactorTests: XCTestCase {
    func test_compactReturnsEmptyHistoryForEmptyInput() {
        XCTAssertEqual(HistoryCompactor.compact([]).count, 0)
    }

    func test_compactPreservesSystemPromptAndRecentTurns() {
        let messages: [ChatMessage] = [
            .system("system"),
            .user("old user"),
            .assistant("old assistant"),
            .user("new user"),
            .assistant("new assistant"),
        ]

        let compacted = HistoryCompactor.compact(
            messages,
            config: .init(maxMessages: 2, characterBudget: 40, preserveLastNTurns: 1)
        )

        XCTAssertEqual(compacted.first?.role, .system)
        XCTAssertTrue(compacted.first?.content.contains("system") == true)
        XCTAssertTrue(compacted.first?.content.contains("older messages were compacted") == true)
        XCTAssertEqual(compacted.suffix(2).map(\.content), ["new user", "new assistant"])
    }

    func test_compactDropsToolMessagesFromStoredPromptHistory() {
        let messages: [ChatMessage] = [
            .system("system"),
            .user("question"),
            .tool("[query] result"),
            .assistant("answer"),
        ]

        let compacted = HistoryCompactor.compact(
            messages,
            config: .init(maxMessages: 2, characterBudget: 10, preserveLastNTurns: 1)
        )

        XCTAssertFalse(compacted.contains { $0.role == .tool })
        XCTAssertTrue(compacted.contains { $0.content == "question" })
        XCTAssertTrue(compacted.contains { $0.content == "answer" })
    }

    func test_compactDoesNotChangeHistoryWithinBudget() {
        let messages: [ChatMessage] = [
            .system("system"),
            .user("hello"),
            .assistant("world"),
        ]

        let compacted = HistoryCompactor.compact(
            messages,
            config: .init(maxMessages: 10, characterBudget: 1_000)
        )

        XCTAssertEqual(compacted.map(\.role), messages.map(\.role))
        XCTAssertEqual(compacted.map(\.content), messages.map(\.content))
    }
}

