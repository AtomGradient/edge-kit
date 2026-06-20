// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class RouteTrainingPairTests: XCTestCase {

    func test_routeTrainingPair_roundTripsBaseChatContract() throws {
        let pair = RouteTrainingPair(
            input: UserInputContext(
                text: "中国的首都是哪里",
                localeIdentifier: "zh-Hans",
                appID: "test.app",
                appContext: ["surface": .string("eval")]
            ),
            expectedIntentTag: .baseChat,
            selectedToolNames: [],
            toolCallPlan: [
                ToolCallPlan(
                    toolName: "query_expenses",
                    arguments: ["limit": .int(3), "orderBy": .string("date")],
                    reason: "contract replay"
                )
            ],
            confidence: 0.96,
            source: .eval,
            rationale: "General-knowledge query should stay on base chat.",
            createdAtMs: 1_777_777_777_000,
            metadata: ["case_id": .string("base_1")]
        )

        let data = try JSONEncoder().encode(pair)
        let encoded = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(encoded.contains("\"selectedToolNames\""))
        XCTAssertFalse(encoded.contains("\"expectedToolNames\""))

        let decoded = try JSONDecoder().decode(RouteTrainingPair.self, from: data)

        XCTAssertEqual(decoded, pair)
        XCTAssertEqual(decoded.toolCallPlan?.first?.toolName, "query_expenses")
        XCTAssertEqual(decoded.toolCallPlan?.first?.arguments["limit"], .int(3))
        XCTAssertEqual(decoded.routerLabel, .baseChat)
    }

    func test_routeTrainingPairClassifier_emitsBaseChatOnExactNormalizedMatch() {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: " Explain   Photosynthesis "),
            expectedIntentTag: .baseChat,
            confidence: 0.97,
            source: .eval
        )
        let router = QueryRouter(routeTrainingPairs: [pair])

        let result = router.classify("explain photosynthesis")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.97)
        XCTAssertEqual(result.reason, "few_shot_high_conf")
        XCTAssertEqual(result.rawModelLabel, .baseChat)
    }

    func test_routeTrainingPairClassifier_canonicalizesFullwidthPunctuation() {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "中国的首都是哪里?"),
            expectedIntentTag: .baseChat,
            confidence: 0.97,
            source: .eval
        )
        let router = QueryRouter(routeTrainingPairs: [pair])

        let result = router.classify("中国的首都是哪里？")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.97)
        XCTAssertEqual(result.reason, "few_shot_high_conf")
        XCTAssertEqual(result.rawModelLabel, .baseChat)
    }

    func test_routeTrainingPairClassifier_ignoresTerminalSentencePunctuation() {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "What is photosynthesis?"),
            expectedIntentTag: .baseChat,
            confidence: 0.97,
            source: .eval
        )
        let router = QueryRouter(routeTrainingPairs: [pair])

        let result = router.classify("what is photosynthesis")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.97)
        XCTAssertEqual(result.reason, "few_shot_high_conf")
        XCTAssertEqual(result.rawModelLabel, .baseChat)
    }

    func test_routeTrainingPairClassifier_missFailsClosedWithoutLegacyRules() {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "中国的首都是哪里"),
            expectedIntentTag: .baseChat,
            confidence: 0.97,
            source: .eval
        )
        let router = QueryRouter(routeTrainingPairs: [pair])

        let result = router.classify("上周花了多少钱")

        XCTAssertEqual(result.label, .baseChat)
        XCTAssertEqual(result.confidence, 0.0)
        XCTAssertEqual(result.reason, "legacy_rule_based_fallback_disabled_no_match")
        XCTAssertNil(result.rawModelLabel)
    }

    func test_defaultRouter_mapsTrainingPairBaseChatToZeroToolIntent() async throws {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "中国的首都是哪里"),
            expectedIntentTag: .baseChat,
            confidence: 0.97,
            source: .eval
        )
        let router = DefaultPersonalIntentRouter(
            queryRouter: QueryRouter(routeTrainingPairs: [pair])
        )

        let decision = try await router.route(UserInputContext(text: "中国的首都是哪里"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.confidence, 0.97)
        XCTAssertEqual(decision.fallbackChain, [.baseChat])
        XCTAssertEqual(decision.auditPayload["router_label"], .string("base_chat"))
    }

    func test_routeTrainingPairMatcher_matchesExactPunctuationVariant() throws {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "What is photosynthesis?"),
            expectedIntentTag: .baseChat,
            confidence: 0.95,
            source: .runtimeAudit
        )
        let matcher = RouteTrainingPairMatcher(pairs: [pair])

        let match = try XCTUnwrap(matcher.match("what is photosynthesis"))

        XCTAssertEqual(match.mode, .exact)
        XCTAssertEqual(match.score, 1.0)
        XCTAssertEqual(match.pair.expectedIntentTag, .baseChat)
    }

    func test_routeTrainingPairEvidenceRouter_matchesNearbyEvidenceWithoutAppKeywords() async throws {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "Please explain photosynthesis briefly."),
            expectedIntentTag: .baseChat,
            confidence: 0.94,
            source: .runtimeAudit
        )
        let router = RouteTrainingPairEvidenceRouter(
            baseRouter: DefaultPersonalIntentRouter(),
            routeTrainingPairs: [pair]
        )

        let decision = try await router.route(
            UserInputContext(text: "Explain photosynthesis in one sentence")
        )

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.reason, "route_pair_similarity_base_chat")
        XCTAssertEqual(decision.auditPayload["route_pair_match_mode"], .string("similarity"))
        XCTAssertEqual(decision.auditPayload["route_pair_tool_plan_status"], .string("no_selected_tools"))
    }

    func test_routeTrainingPairDecisionApplier_attachesToolPlanOnlyWhenAllowed() async throws {
        let plan = ToolCallPlan(
            toolName: "query_expenses",
            arguments: [
                "timeRange": .string("all"),
                "orderBy": .string("date"),
                "ascending": .bool(false),
                "limit": .int(3),
            ],
            reason: "contract replay"
        )
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "show my latest three expense records"),
            expectedIntentTag: .exactFact,
            selectedToolNames: ["query_expenses"],
            toolCallPlan: [plan],
            confidence: 0.93,
            source: .runtimeAudit
        )
        let router = RouteTrainingPairEvidenceRouter(
            baseRouter: DefaultPersonalIntentRouter(),
            routeTrainingPairs: [pair],
            policy: RouteTrainingPairDecisionPolicy(allowedToolNames: ["query_expenses"])
        )

        let decision = try await router.route(
            UserInputContext(text: "please show recent three expense record details")
        )

        XCTAssertEqual(decision.intent.tag, .exactFact)
        XCTAssertEqual(decision.toolPlan, plan)
        XCTAssertEqual(decision.auditPayload["route_pair_tool_plan_status"], .string("selected"))
        XCTAssertNil(decision.auditPayload["expected_tool_names"])
        XCTAssertNil(decision.auditPayload["route_pair_expected_tool_names"])
        XCTAssertEqual(decision.auditPayload["route_pair_selected_tool_names"], .array([.string("query_expenses")]))
    }

    func test_routeTrainingPairDecisionApplier_mismatchedToolPlanFailsClosed() async throws {
        let pair = RouteTrainingPair(
            input: UserInputContext(text: "show my latest three expense records"),
            expectedIntentTag: .exactFact,
            selectedToolNames: ["query_expenses"],
            toolCallPlan: [
                ToolCallPlan(
                    toolName: "delete_expense",
                    arguments: ["limit": .int(3)],
                    reason: "invalid for read-only evidence route"
                )
            ],
            confidence: 0.93,
            source: .runtimeAudit
        )
        let router = RouteTrainingPairEvidenceRouter(
            baseRouter: DefaultPersonalIntentRouter(),
            routeTrainingPairs: [pair],
            policy: RouteTrainingPairDecisionPolicy(allowedToolNames: ["query_expenses"])
        )

        let decision = try await router.route(
            UserInputContext(text: "please show recent three expense record details")
        )

        XCTAssertEqual(decision.intent.tag, .exactFact)
        XCTAssertNil(decision.toolPlan)
        XCTAssertEqual(decision.fallbackChain, [.exactFact, .baseChat])
        XCTAssertEqual(
            decision.auditPayload["route_pair_tool_plan_status"],
            .string("mismatched_selected_tools")
        )
        XCTAssertEqual(
            decision.auditPayload["route_pair_tool_plan_names"],
            .array([.string("delete_expense")])
        )
    }
}
