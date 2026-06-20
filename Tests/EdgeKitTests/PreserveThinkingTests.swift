// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class PreserveThinkingTests: XCTestCase {

    func testHistoricalAssistantThinkingIsStrippedByDefault() {
        let messages: [ChatMessage] = [
            .user("Hi"),
            .assistant("<think>\nLet me think.\n</think>\n\nHello!"),
            .user("How are you?"),
        ]

        let promptMessages = messages.promptCacheMessages(preserveThinking: false)

        XCTAssertEqual(promptMessages[1].content, "Hello!")
    }

    func testHistoricalAssistantThinkingIsPreservedWhenRequested() {
        let assistantContent = "<think>\nLet me think.\n</think>\n\nHello!"
        let messages: [ChatMessage] = [
            .user("Hi"),
            .assistant(assistantContent),
            .user("How are you?"),
        ]

        let promptMessages = messages.promptCacheMessages(preserveThinking: true)

        XCTAssertEqual(promptMessages[1].content, assistantContent)
    }

    func testAssistantAfterLastUserKeepsThinkingForAssistantPrefill() {
        let assistantContent = "<think>\nCurrent turn reasoning.\n</think>\n\nPartial answer"
        let messages: [ChatMessage] = [
            .user("Continue this answer."),
            .assistant(assistantContent),
        ]

        let promptMessages = messages.promptCacheMessages(preserveThinking: false)

        XCTAssertEqual(promptMessages[1].content, assistantContent)
    }

    func testThinkingStripRemovesMultipleClosedBlocks() {
        let content = """
        before
        <think>first</think>
        middle
        <think>second</think>
        after
        """

        XCTAssertEqual(
            ChatMessage.strippingThinkingContent(from: content),
            """
            before

            middle

            after
            """
        )
    }

    func testEdgeGenerateParametersPreserveThinkingDefaultsFalse() {
        XCTAssertFalse(EdgeGenerateParameters.default.preserveThinking)
    }
}
