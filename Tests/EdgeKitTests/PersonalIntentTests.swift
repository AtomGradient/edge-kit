// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class PersonalIntentTests: XCTestCase {

    func test_personalIntentTagRawValues_areSnakeCase() {
        XCTAssertEqual(PersonalIntentTag.exactFact.rawValue, "exact_fact")
        XCTAssertEqual(PersonalIntentTag.aggregateFact.rawValue, "aggregate_fact")
        XCTAssertEqual(PersonalIntentTag.userProfile.rawValue, "user_profile")
        XCTAssertEqual(PersonalIntentTag.appAction.rawValue, "app_action")
        XCTAssertEqual(PersonalIntentTag.baseChat.rawValue, "base_chat")
        XCTAssertEqual(PersonalIntentTag.mixed.rawValue, "mixed")
    }

    func test_personalIntent_encodesAsTagDiscriminatedShape() throws {
        let plan = FactQueryPlan(
            namespace: "canonical",
            schema: "finance.expense",
            filters: [
                FactQueryFilter(
                    field: "merchant",
                    op: .contains,
                    value: .string("parking")
                )
            ],
            sort: [FactSort(field: "occurred_at", direction: .descending)],
            limit: 1,
            requestedFields: ["merchant", "amount", "occurred_at"]
        )
        let intent = PersonalIntent.exactFact(plan: plan)

        let data = try JSONEncoder().encode(intent)
        let decoded = try JSONDecoder().decode(PersonalIntent.self, from: data)
        XCTAssertEqual(decoded, intent)

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["tag"] as? String, "exact_fact")
        XCTAssertNotNil(object["plan"])
        XCTAssertNil(object["exactFact"])
    }

    func test_routeDecision_roundTripsWithToolPersonaFallbackAndAudit() throws {
        let decision = RouteDecision(
            intent: .mixed(candidates: [
                .aggregateFact(plan: FactQueryPlan(
                    schema: "finance.expense",
                    aggregation: FactAggregation(function: .sum, field: "amount")
                )),
                .userProfile(detail: ProfileDetail(
                    kind: .habit,
                    dimensions: ["category_affinity"]
                )),
            ]),
            confidence: 0.82,
            reason: "contract_test",
            toolPlan: ToolCallPlan(
                toolName: "query_facts",
                arguments: ["limit": .int(5)]
            ),
            personaSignal: PersonaSignal(
                source: .rpp,
                label: "needs_personal_context",
                confidence: 0.7
            ),
            fallbackChain: [.aggregateFact, .userProfile, .baseChat],
            auditPayload: [
                "router": .string("test"),
                "scores": .array([.double(0.82), .double(0.64)]),
                "used_tool": .bool(true),
            ]
        )

        let data = try JSONEncoder().encode(decision)
        let decoded = try JSONDecoder().decode(RouteDecision.self, from: data)
        XCTAssertEqual(decoded, decision)
    }

    func test_factQueryPlan_canExpressRecentFact() {
        let plan = FactQueryPlan(
            schema: "health.record",
            filters: [
                FactQueryFilter(
                    field: "category",
                    op: .equals,
                    value: .string("medical")
                )
            ],
            sort: [FactSort(field: "occurred_at", direction: .descending)],
            limit: 1,
            requestedFields: ["occurred_at", "provider", "summary"]
        )

        XCTAssertEqual(plan.filters.first?.op, .equals)
        XCTAssertEqual(plan.sort, [FactSort(field: "occurred_at", direction: .descending)])
        XCTAssertEqual(plan.limit, 1)
        XCTAssertEqual(plan.requestedFields, ["occurred_at", "provider", "summary"])
    }

    func test_factQueryPlan_canExpressLargestFactInTimeRange() {
        let plan = FactQueryPlan(
            schema: "finance.expense",
            filters: [
                FactQueryFilter(
                    field: "category",
                    op: .equals,
                    value: .string("subscription")
                ),
                FactQueryFilter(
                    field: "occurred_at",
                    op: .between,
                    value: .array([.string("2026-04-01"), .string("2026-04-30")])
                ),
            ],
            sort: [FactSort(field: "amount", direction: .descending)],
            limit: 1
        )

        XCTAssertEqual(plan.filters.map(\.op), [.equals, .between])
        XCTAssertEqual(plan.sort.first, FactSort(field: "amount", direction: .descending))
        XCTAssertEqual(plan.limit, 1)
    }

    func test_factQueryPlan_canExpressMultiValueFilterAndAggregate() {
        let plan = FactQueryPlan(
            schema: "finance.expense",
            filters: [
                FactQueryFilter(
                    field: "category",
                    op: .inList,
                    value: .array([.string("subscription"), .string("education")])
                )
            ],
            aggregation: FactAggregation(function: .sum, field: "amount"),
            groupBy: ["category"],
            sort: [FactSort(field: "sum_amount", direction: .descending)]
        )

        XCTAssertEqual(plan.filters.first?.op, .inList)
        XCTAssertEqual(
            plan.filters.first?.value,
            .array([.string("subscription"), .string("education")])
        )
        XCTAssertEqual(plan.aggregation, FactAggregation(function: .sum, field: "amount"))
        XCTAssertEqual(plan.groupBy, ["category"])
        XCTAssertEqual(plan.sort, [FactSort(field: "sum_amount", direction: .descending)])
    }

    func test_auditValueDocumentsNumericDecodeBehavior() throws {
        XCTAssertEqual(
            try JSONDecoder().decode(AuditValue.self, from: Data("1".utf8)),
            .int(1)
        )
        XCTAssertEqual(
            try JSONDecoder().decode(AuditValue.self, from: Data("1.5".utf8)),
            .double(1.5)
        )

        let encodedDoubleOne = try JSONEncoder().encode(AuditValue.double(1.0))
        let decodedDoubleOne = try JSONDecoder().decode(
            AuditValue.self,
            from: encodedDoubleOne
        )
        XCTAssertEqual(decodedDoubleOne, .int(1))
    }

    func test_auditValueNestedArrayRoundTrips() throws {
        let value = AuditValue.array([
            .array([.int(1), .int(2)]),
            .array([.string("a"), .string("b")]),
        ])

        let data = try JSONEncoder().encode(value)
        let decoded = try JSONDecoder().decode(AuditValue.self, from: data)

        XCTAssertEqual(decoded, value)
    }

    func test_personaSignal_defaultActivationSteeringAlphaIsZero() {
        let signal = PersonaSignal(
            source: .rpp,
            label: "profile_hint",
            confidence: 0.75
        )

        XCTAssertEqual(signal.activationSteeringAlpha, 0)
    }
}
