// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class DefaultPersonalIntentRouterTests: XCTestCase {

    func test_defaultRouterDoesNotUseLegacyKeywordFallback() async throws {
        let router = DefaultPersonalIntentRouter()

        let decision = try await router.route(UserInputContext(text: "上周花了多少钱"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.reason, "legacy_rule_based_fallback_disabled")
        XCTAssertEqual(decision.fallbackChain, [.baseChat])
    }

    func test_factLabel_mapsToExactFact() async throws {
        let router = DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .fact, defaultConfidence: 0.93)
        ))

        let decision = try await router.route(UserInputContext(text: "plain input"))

        guard case .exactFact(let plan) = decision.intent else {
            return XCTFail("Expected exactFact, got \(decision.intent)")
        }
        XCTAssertEqual(plan, FactQueryPlan())
        XCTAssertEqual(decision.confidence, 0.93)
        XCTAssertEqual(decision.fallbackChain, [.exactFact, .baseChat])
        XCTAssertEqual(decision.auditPayload["router_label"], .string("fact"))
        XCTAssertEqual(decision.auditPayload["router_reason"], .string("few_shot_high_conf"))
    }

    func test_personaLabel_mapsToUserProfile() async throws {
        let router = DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .persona, defaultConfidence: 0.91)
        ))

        let decision = try await router.route(UserInputContext(text: "plain input"))

        guard case .userProfile(let detail) = decision.intent else {
            return XCTFail("Expected userProfile, got \(decision.intent)")
        }
        XCTAssertEqual(detail.kind, .summary)
        XCTAssertEqual(decision.confidence, 0.91)
        XCTAssertEqual(decision.fallbackChain, [.userProfile, .baseChat])
        XCTAssertEqual(decision.auditPayload["router_label"], .string("persona"))
    }

    func test_mixedLabel_mapsToFlatMixedCandidates() async throws {
        let router = DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .mixed, defaultConfidence: 0.9)
        ))

        let decision = try await router.route(UserInputContext(text: "plain input"))

        guard case .mixed(let candidates) = decision.intent else {
            return XCTFail("Expected mixed, got \(decision.intent)")
        }
        XCTAssertEqual(candidates.map(\.tag), [.exactFact, .userProfile, .baseChat])
        XCTAssertFalse(candidates.contains { candidate in
            if case .mixed = candidate { return true }
            return false
        })
        XCTAssertEqual(decision.fallbackChain, [.exactFact, .userProfile, .baseChat])
        XCTAssertEqual(decision.auditPayload["router_label"], .string("mixed"))
    }

    func test_baseChatLabel_mapsToBaseChatWithNoFallbackTools() async throws {
        let router = DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .baseChat, defaultConfidence: 0.94)
        ))

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.confidence, 0.94)
        XCTAssertEqual(decision.fallbackChain, [.baseChat])
        XCTAssertEqual(decision.auditPayload["router_label"], .string("base_chat"))
    }

    func test_lowConfidenceRouterResult_mapsToMixedWithRawModelAudit() async throws {
        let router = DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .fact, defaultConfidence: 0.4)
        ))

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .mixed)
        XCTAssertEqual(decision.reason, "low_confidence")
        XCTAssertEqual(decision.auditPayload["router_label"], .string("mixed"))
        XCTAssertEqual(decision.auditPayload["raw_model_label"], .string("fact"))
    }
}
