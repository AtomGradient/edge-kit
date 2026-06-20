// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Maps explicit `QueryRouter` output to the 6-label `PersonalIntent` contract.
///
/// The default `QueryRouter` no longer uses legacy keyword fallback. Callers
/// that need fact/profile routing must inject a learned classifier or another
/// explicit non-rule router.
///
/// LIMITATIONS (P0c v1, lifted by RouteTrainingPair / few-shot router upgrades):
/// 1. `FactQueryPlan` is emitted empty. Downstream tool planning must fill
///    namespace, schema, filters, sort, and limit from `UserInputContext.text`.
///    The plan slot is the type boundary, not a parsed query.
/// 2. `.fact` always maps to `.exactFact`. `.aggregateFact` is unreachable
///    because `QueryRouter` cannot yet distinguish exact from aggregate intent.
/// 3. `.baseChat` is reachable when a base/learned classifier emits the
///    explicit `base_chat` label, and is also the default fail-closed result
///    when legacy rule-based fallback is disabled.
public struct DefaultPersonalIntentRouter: PersonalIntentRouter {
    private let queryRouter: QueryRouter

    public init(queryRouter: QueryRouter = QueryRouter()) {
        self.queryRouter = queryRouter
    }

    public func route(_ input: UserInputContext) async throws -> RouteDecision {
        let result = queryRouter.classify(input.text)
        let auditPayload = Self.auditPayload(from: result)

        switch result.label {
        case .fact:
            let intent = PersonalIntent.exactFact(plan: FactQueryPlan())
            return RouteDecision(
                intent: intent,
                confidence: result.confidence,
                reason: result.reason,
                fallbackChain: [.exactFact, .baseChat],
                auditPayload: auditPayload
            )

        case .persona:
            let intent = PersonalIntent.userProfile(detail: ProfileDetail(kind: .summary))
            return RouteDecision(
                intent: intent,
                confidence: result.confidence,
                reason: result.reason,
                fallbackChain: [.userProfile, .baseChat],
                auditPayload: auditPayload
            )

        case .mixed:
            let candidates: [PersonalIntent] = [
                .exactFact(plan: FactQueryPlan()),
                .userProfile(detail: ProfileDetail(kind: .summary)),
                .baseChat,
            ]
            return RouteDecision(
                intent: .mixed(candidates: candidates),
                confidence: result.confidence,
                reason: result.reason,
                fallbackChain: [.exactFact, .userProfile, .baseChat],
                auditPayload: auditPayload
            )

        case .baseChat:
            return RouteDecision(
                intent: .baseChat,
                confidence: result.confidence,
                reason: result.reason,
                fallbackChain: [.baseChat],
                auditPayload: auditPayload
            )
        }
    }

    private static func auditPayload(from result: RouterResult) -> [String: AuditValue] {
        var payload: [String: AuditValue] = [
            "router_label": .string(result.label.rawValue),
            "router_reason": .string(result.reason),
            "latency_ms": .double(result.latencyMs),
            "query": .string(result.query),
        ]
        if let rawModelLabel = result.rawModelLabel {
            payload["raw_model_label"] = .string(rawModelLabel.rawValue)
        }
        return payload
    }
}
