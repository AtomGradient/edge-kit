// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class QueryRouterTests: XCTestCase {

    func test_failureModesFailClosedToBaseChat() {
        let router = QueryRouter(base: MockBaseClassifier(defaultLabel: .mixed))

        for mode in [QueryRouter.FailureMode.oom, .timeout, .modelError] {
            let result = router.classify("上周花了多少钱", failureMode: mode)

            XCTAssertEqual(result.label, .baseChat)
            XCTAssertEqual(result.confidence, 0.0)
            XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled_\(mode.rawValue)")
            XCTAssertNil(result.rawModelLabel)
        }
    }

    func test_nilBaseFailsClosedToBaseChat() {
        let router = QueryRouter(base: nil)

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled")
        XCTAssertNil(result.rawModelLabel)
    }

    func test_unavailableBaseFailsClosedToBaseChat() {
        let router = QueryRouter(base: MockBaseClassifier(available: false))

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled")
        XCTAssertNil(result.rawModelLabel)
    }

    func test_classifierNoMatchFailsClosedToBaseChat() {
        let router = QueryRouter(base: MockBaseClassifier(labelHook: { _ in
            throw BaseClassifierError.noMatch
        }))

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled_no_match")
        XCTAssertNil(result.rawModelLabel)
    }

    func test_classifierErrorFailsClosedToBaseChat() {
        let router = QueryRouter(base: MockBaseClassifier(labelHook: { _ in
            throw NSError(domain: "test", code: 1)
        }))

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled_model_error")
        XCTAssertNil(result.rawModelLabel)
    }

    func test_lowConfidenceReturnsMixedWithoutRules() {
        let router = QueryRouter(base: MockBaseClassifier(
            defaultLabel: .fact,
            defaultConfidence: 0.5
        ))

        let result = router.classify("plain input")

        XCTAssertEqual(result.label, .mixed)
        XCTAssertEqual(result.confidence, 0.5)
        XCTAssertEqual(result.reason, "low_confidence")
        XCTAssertEqual(result.rawModelLabel, .fact)
    }

    func test_highConfidenceReturnsBaseLabel() {
        let router = QueryRouter(base: MockBaseClassifier(
            defaultLabel: .fact,
            defaultConfidence: 0.95
        ))

        let result = router.classify("plain input")

        XCTAssertEqual(result.label, .fact)
        XCTAssertEqual(result.confidence, 0.95)
        XCTAssertEqual(result.reason, "few_shot_high_conf")
        XCTAssertEqual(result.rawModelLabel, .fact)
    }

    func test_highConfidenceCanReturnBaseChatLabel() {
        let router = QueryRouter(base: MockBaseClassifier(
            defaultLabel: .baseChat,
            defaultConfidence: 0.96
        ))

        let result = router.classify("中国的首都是哪里")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.reason, "few_shot_high_conf")
        XCTAssertEqual(result.rawModelLabel, .baseChat)
    }

    func test_forceConfidenceOverridesBaseConfidence() {
        let router = QueryRouter(base: MockBaseClassifier(
            defaultLabel: .fact,
            defaultConfidence: 0.95
        ))

        let result = router.classify("plain input", forceConfidence: 0.3)

        XCTAssertEqual(result.label, .mixed)
        XCTAssertEqual(result.confidence, 0.3)
        XCTAssertEqual(result.reason, "low_confidence")
        XCTAssertEqual(result.rawModelLabel, .fact)
    }

    func test_resultContainsLatencyAndQuery() {
        let router = QueryRouter()

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.query, "上周花了多少钱")
        XCTAssertGreaterThanOrEqual(result.latencyMs, 0)
        XCTAssertLessThan(result.latencyMs, 150)
    }
}
