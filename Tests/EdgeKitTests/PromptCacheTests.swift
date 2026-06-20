// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

@MainActor
final class PromptCacheTests: XCTestCase {

    func testValidatePrefixRejectsChangedSystemOrToolPrompt() {
        let cache = PromptCacheManager()
        cache.update(cache: [], totalTokenCount: 3, tokenPrefix: [1, 2, 3])

        XCTAssertTrue(cache.validatePrefix([1, 2, 3, 4]))
        XCTAssertFalse(cache.validatePrefix([9, 2, 3, 4]))
    }

    func testNativePromptSessionReuseAcceptsStrictPrefix() {
        let reused = NativePromptSessionReuse.reusablePrefixLength(
            cachedTokenIds: [10, 11, 12],
            promptTokenIds: [10, 11, 12, 13, 14]
        )

        XCTAssertEqual(reused, 3)
    }

    func testNativePromptSessionReuseRejectsMismatchOrShorterPrompt() {
        XCTAssertNil(NativePromptSessionReuse.reusablePrefixLength(
            cachedTokenIds: [10, 11, 12],
            promptTokenIds: [10, 99, 12, 13]
        ))
        XCTAssertNil(NativePromptSessionReuse.reusablePrefixLength(
            cachedTokenIds: [10, 11, 12],
            promptTokenIds: [10, 11]
        ))
        XCTAssertNil(NativePromptSessionReuse.reusablePrefixLength(
            cachedTokenIds: [],
            promptTokenIds: [10, 11]
        ))
    }

    func testNativePromptSessionReuseCanSkipCachedQwenThinkingSentinel() {
        let match = NativePromptSessionReuse.reusablePrefixMatch(
            cachedTokenIds: [1, 98, 99, 2, 3],
            promptTokenIds: [1, 2, 3, 4],
            skippableCachedTokenSequences: [[98, 99]]
        )

        XCTAssertEqual(match, .init(cachedTokenLength: 5, promptTokenLength: 3))
    }

    func testNativePromptSessionReuseCanSkipMultipleCachedQwenThinkingSentinels() {
        let match = NativePromptSessionReuse.reusablePrefixMatch(
            cachedTokenIds: [1, 98, 99, 2, 3, 98, 99, 4],
            promptTokenIds: [1, 2, 3, 4, 5],
            skippableCachedTokenSequences: [[98, 99]]
        )

        XCTAssertEqual(match, .init(cachedTokenLength: 8, promptTokenLength: 4))
    }

    func testNativePromptSessionReuseKeepsStopTokenInTrackedPrefix() {
        let imEnd = 151_645
        let match = NativePromptSessionReuse.reusablePrefixMatch(
            cachedTokenIds: [10, 11, imEnd],
            promptTokenIds: [10, 11, imEnd, 12, 13],
            skippableCachedTokenSequences: []
        )

        XCTAssertEqual(match, .init(cachedTokenLength: 3, promptTokenLength: 3))
    }

    func testQwenIncrementalSuffixRendersNewUserAfterGeneratedAssistant() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer"),
                .user("next"),
            ],
            enableThinking: false
        )

        XCTAssertEqual(
            suffix,
            "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "next<|im_end|>\n"
                + "<|im_start|>assistant\n"
                + "<think>\n\n</think>\n\n"
        )
    }

    func testQwenIncrementalSuffixPreservesMessageWhitespace() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer"),
                .user(" next\n"),
            ],
            enableThinking: false
        )

        XCTAssertEqual(
            suffix,
            "<|im_end|>\n"
                + "<|im_start|>user\n"
                + " next\n<|im_end|>\n"
                + "<|im_start|>assistant\n"
                + "<think>\n\n</think>\n\n"
        )
    }

    func testQwenIncrementalSuffixMatchesDespiteTrailingNewline() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer\n",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer"),
                .user("next"),
            ],
            enableThinking: false
        )

        XCTAssertNotNil(suffix, "trailing newline drift should be normalized away")
    }

    func testQwenIncrementalSuffixMatchesDespiteTrailingSpaces() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer   ",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer\n"),
                .user("next"),
            ],
            enableThinking: false
        )

        XCTAssertNotNil(suffix, "both sides normalize trailing whitespace")
    }

    func testNormalizeAssistantTextTrimsTrailingOnly() {
        XCTAssertEqual(
            NativePromptSessionReuse.normalizeAssistantText("hello\n"),
            "hello"
        )
        XCTAssertEqual(
            NativePromptSessionReuse.normalizeAssistantText("hello  \n\t"),
            "hello"
        )
        XCTAssertEqual(
            NativePromptSessionReuse.normalizeAssistantText("  hello"),
            "  hello",
            "leading whitespace preserved"
        )
        XCTAssertEqual(
            NativePromptSessionReuse.normalizeAssistantText(""),
            ""
        )
    }

    func testQwenIncrementalSuffixRejectsRetokenizedAssistantMismatch() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("different"),
                .user("next"),
            ],
            enableThinking: false
        )

        XCTAssertNil(suffix)
    }

    func testQwenIncrementalSuffixRejectsSemanticAppend() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer!"),
                .user("next"),
            ],
            enableThinking: false
        )

        XCTAssertNil(suffix)
    }

    func testQwenIncrementalSuffixAssistantMismatchReportsDiffDetails() {
        let result = NativePromptSessionReuse.qwenIncrementalSuffix(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer alpha",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer beta"),
                .user("next"),
            ],
            enableThinking: false,
            matchPath: "test_match"
        )

        guard case let .reject(reason) = result else {
            XCTFail("expected mismatch rejection")
            return
        }
        XCTAssertTrue(reason.contains("assistant_text_mismatch"))
        XCTAssertTrue(reason.contains("matchPath=test_match"))
        XCTAssertTrue(reason.contains("firstDiffIndex=7"))
        XCTAssertNotNil(reason.range(
            of: #"storedHash=[0-9a-f]{16}"#,
            options: .regularExpression
        ))
    }

    func testQwenIncrementalSuffixAllowsEmptyThinkingSentinelDrift() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("previous"),
            ],
            lastAssistantText: "assistant\n<think>\n\n</think>\n\n# Upgrade Path Design",
            currentMessages: [
                .system("system"),
                .user("previous"),
                .assistant("assistant\n\n\n# Upgrade Path Design"),
                .user("next"),
            ],
            enableThinking: false,
            matchPath: "test_match"
        )

        XCTAssertNotNil(suffix)
        XCTAssertTrue(suffix?.contains("<|im_start|>user\nnext") == true)
    }

    func testQwenIncrementalSuffixRejectsToolMessages() {
        let suffix = NativePromptSessionReuse.qwenIncrementalSuffixText(
            previousPromptMessages: [
                .system("system"),
                .user("hello"),
            ],
            lastAssistantText: "answer",
            currentMessages: [
                .system("system"),
                .user("hello"),
                .assistant("answer"),
                .user("call tool"),
                .tool("result"),
            ],
            enableThinking: false
        )

        XCTAssertNil(suffix)
    }

    func testNativeVLMTextDeltaPrefillPlannerHitsAfterImageTurn() {
        let result = VLMEngine.NativeVLMTextDeltaPrefillPlanner.makePlan(
            cachedTokenCount: 1_280,
            previousPromptMessages: [
                .system("system"),
                .user("Describe this image."),
            ],
            lastAssistantText: "It shows a red bicycle.",
            currentMessages: [
                .system("system"),
                .user("Describe this image."),
                .assistant("It shows a red bicycle."),
                .user("What color is the bicycle?"),
            ],
            enableThinking: false
        )

        guard case .success(let plan) = result else {
            XCTFail("expected VLM text delta prefill hit")
            return
        }
        XCTAssertEqual(plan.cachedTokensReused, 1_280)
        XCTAssertEqual(
            plan.suffixText,
            "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "What color is the bicycle?<|im_end|>\n"
                + "<|im_start|>assistant\n"
                + "<think>\n\n</think>\n\n"
        )
        XCTAssertFalse(plan.suffixText.contains("<|vision_start|>"))
        XCTAssertFalse(plan.suffixText.contains("<|image_pad|>"))
    }

}
