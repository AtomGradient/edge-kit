// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Match mode for route/action evidence replay.
public enum RouteTrainingPairMatchMode: String, Sendable, Codable, Equatable {
    case exact
    case similarity
}

/// A selected route/action evidence pair plus ranking metadata.
public struct RouteTrainingPairMatch: Sendable, Equatable {
    public var pair: RouteTrainingPair
    public var score: Double
    public var runnerUpScore: Double
    public var mode: RouteTrainingPairMatchMode

    public init(
        pair: RouteTrainingPair,
        score: Double,
        runnerUpScore: Double,
        mode: RouteTrainingPairMatchMode
    ) {
        self.pair = pair
        self.score = score
        self.runnerUpScore = runnerUpScore
        self.mode = mode
    }
}

/// Tunables for route/action evidence matching.
///
/// These defaults intentionally remain conservative: exact replay still wins,
/// while similarity requires both a minimum score and separation from the
/// runner-up tool group.
public struct RouteTrainingPairMatcherOptions: Sendable, Equatable {
    public var minimumCandidateScore: Double
    public var toolRouteThreshold: Double
    public var noToolRouteThreshold: Double
    public var minimumMargin: Double

    public init(
        minimumCandidateScore: Double = 0.18,
        toolRouteThreshold: Double = 0.36,
        noToolRouteThreshold: Double = 0.40,
        minimumMargin: Double = 0.035
    ) {
        self.minimumCandidateScore = minimumCandidateScore
        self.toolRouteThreshold = toolRouteThreshold
        self.noToolRouteThreshold = noToolRouteThreshold
        self.minimumMargin = minimumMargin
    }
}

/// Generic route/action evidence matcher.
///
/// This component is deliberately app-agnostic. It does not contain business
/// keywords, tool names, or product-specific routing rules; it only matches
/// user text against host-model reviewed `RouteTrainingPair` evidence.
public final class RouteTrainingPairMatcher: @unchecked Sendable {
    private struct Candidate {
        let pair: RouteTrainingPair
        let score: Double
        let mode: RouteTrainingPairMatchMode

        var groupKey: String {
            pair.selectedToolNames.sorted().joined(separator: ",")
        }
    }

    private let pairs: [RouteTrainingPair]
    private let options: RouteTrainingPairMatcherOptions

    public init(
        pairs: [RouteTrainingPair],
        options: RouteTrainingPairMatcherOptions = RouteTrainingPairMatcherOptions()
    ) {
        self.pairs = pairs
        self.options = options
    }

    public func match(_ text: String) -> RouteTrainingPairMatch? {
        let key = Self.normalizedText(text)
        var candidates: [Candidate] = []

        for pair in pairs {
            let pairKey = Self.normalizedText(pair.input.text)
            let exact = pairKey == key
            let score = exact ? 1.0 : Self.routeSimilarity(text, pair.input.text)
            guard exact || score >= options.minimumCandidateScore else { continue }
            candidates.append(
                Candidate(
                    pair: pair,
                    score: score,
                    mode: exact ? .exact : .similarity
                )
            )
        }

        guard !candidates.isEmpty else { return nil }
        var grouped: [String: [Candidate]] = [:]
        for candidate in candidates {
            grouped[candidate.groupKey, default: []].append(candidate)
        }

        let rankedGroups = grouped.values.compactMap { group -> (candidate: Candidate, score: Double)? in
            let ranked = group.sorted { $0.score > $1.score }
            guard let best = ranked.first else { return nil }
            let support = ranked.dropFirst().prefix(2).enumerated().reduce(0.0) { total, item in
                let weight = item.offset == 0 ? 0.12 : 0.06
                return total + item.element.score * weight
            }
            return (best, min(1.0, best.score + support))
        }
        .sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.candidate.score > rhs.candidate.score
            }
            return lhs.score > rhs.score
        }

        guard let best = rankedGroups.first else { return nil }
        let runnerUpScore = rankedGroups.dropFirst().first?.score ?? 0
        let threshold = best.candidate.pair.selectedToolNames.isEmpty
            ? options.noToolRouteThreshold
            : options.toolRouteThreshold
        let margin = best.score - runnerUpScore
        guard best.candidate.mode == .exact
                || (best.score >= threshold && margin >= options.minimumMargin) else {
            return nil
        }

        return RouteTrainingPairMatch(
            pair: best.candidate.pair,
            score: best.score,
            runnerUpScore: runnerUpScore,
            mode: best.candidate.mode
        )
    }

    public static func normalizedText(_ text: String) -> String {
        let widthNormalized = text.applyingTransform(.fullwidthToHalfwidth, reverse: false) ?? text
        let whitespaceNormalized = widthNormalized
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        return whitespaceNormalized.trimmingCharacters(in: CharacterSet(charactersIn: "?!。！？."))
    }

    public static func routeSimilarity(_ lhs: String, _ rhs: String) -> Double {
        if normalizedText(lhs) == normalizedText(rhs) {
            return 1.0
        }
        let lhsFeatures = routeFeatures(lhs)
        let rhsFeatures = routeFeatures(rhs)
        guard !lhsFeatures.isEmpty && !rhsFeatures.isEmpty else { return 0 }
        let intersectionCount = lhsFeatures.intersection(rhsFeatures).count
        let unionCount = lhsFeatures.union(rhsFeatures).count
        let jaccard = Double(intersectionCount) / Double(max(1, unionCount))
        let containment = Double(intersectionCount) / Double(max(1, min(lhsFeatures.count, rhsFeatures.count)))
        let lcs = longestCommonSubstringRatio(lhs, rhs)
        return max(jaccard, containment * 0.78, lcs * 0.92)
    }

    private static func routeFeatures(_ text: String) -> Set<String> {
        let compact = compactText(text)
        let chars = Array(compact)
        guard !chars.isEmpty else { return [] }
        var features = Set<String>()
        for char in chars {
            features.insert(String(char))
        }
        if chars.count >= 2 {
            for index in 0..<(chars.count - 1) {
                features.insert(String(chars[index]) + String(chars[index + 1]))
            }
        }
        if chars.count >= 3 {
            for index in 0..<(chars.count - 2) {
                features.insert(String(chars[index]) + String(chars[index + 1]) + String(chars[index + 2]))
            }
        }
        return features
    }

    private static func compactText(_ text: String) -> String {
        let normalized = normalizedText(text)
        let scalars = normalized.unicodeScalars.filter { scalar in
            !CharacterSet.whitespacesAndNewlines.contains(scalar)
                && !CharacterSet.punctuationCharacters.contains(scalar)
                && !CharacterSet.symbols.contains(scalar)
        }
        return String(String.UnicodeScalarView(scalars))
    }

    private static func longestCommonSubstringRatio(_ lhs: String, _ rhs: String) -> Double {
        let lhsChars = Array(compactText(lhs))
        let rhsChars = Array(compactText(rhs))
        guard !lhsChars.isEmpty && !rhsChars.isEmpty else { return 0 }
        var previous = Array(repeating: 0, count: rhsChars.count + 1)
        var best = 0
        for lhsChar in lhsChars {
            var current = Array(repeating: 0, count: rhsChars.count + 1)
            for index in rhsChars.indices {
                if lhsChar == rhsChars[index] {
                    current[index + 1] = previous[index] + 1
                    best = max(best, current[index + 1])
                }
            }
            previous = current
        }
        return Double(best) / Double(max(1, min(lhsChars.count, rhsChars.count)))
    }
}

public struct RouteTrainingPairDecisionPolicy: Sendable, Equatable {
    /// Optional runtime allow-list. When nil, the pair's expected tools are the
    /// only allow-list. When present, a tool plan must satisfy both lists.
    public var allowedToolNames: Set<String>?
    public var attachToolPlans: Bool

    public init(
        allowedToolNames: Set<String>? = nil,
        attachToolPlans: Bool = true
    ) {
        self.allowedToolNames = allowedToolNames
        self.attachToolPlans = attachToolPlans
    }
}

public enum RouteTrainingPairDecisionApplier {
    public static func apply(
        match: RouteTrainingPairMatch,
        to decision: RouteDecision,
        policy: RouteTrainingPairDecisionPolicy = RouteTrainingPairDecisionPolicy()
    ) -> RouteDecision {
        let pair = match.pair
        let planSelection = selectedToolCallPlan(from: pair, policy: policy)
        let confidence = routeConfidence(for: match)
        var auditPayload = decision.auditPayload
        addEvidenceAudit(
            to: &auditPayload,
            match: match,
            planStatus: planSelection.status
        )

        let intent = intent(for: pair, selectedPlan: planSelection.plan)
        let fallbackChain = fallbackChain(for: pair.expectedIntentTag)
        return RouteDecision(
            intent: intent,
            confidence: confidence,
            reason: "route_pair_\(match.mode.rawValue)_\(pair.expectedIntentTag.rawValue)",
            toolPlan: planSelection.plan,
            fallbackChain: fallbackChain,
            auditPayload: auditPayload
        )
    }

    public static func selectedToolCallPlan(
        from pair: RouteTrainingPair,
        policy: RouteTrainingPairDecisionPolicy = RouteTrainingPairDecisionPolicy()
    ) -> (plan: ToolCallPlan?, status: String) {
        guard policy.attachToolPlans else {
            return (nil, "disabled")
        }
        let selectedToolNames = Set(pair.selectedToolNames)
        guard !selectedToolNames.isEmpty else {
            return (nil, "no_selected_tools")
        }
        guard let plans = pair.toolCallPlan, !plans.isEmpty else {
            return (nil, "missing")
        }
        if let allowedPlan = plans.first(where: { plan in
            selectedToolNames.contains(plan.toolName)
                && (policy.allowedToolNames?.contains(plan.toolName) ?? true)
        }) {
            return (allowedPlan, "selected")
        }
        if plans.contains(where: { selectedToolNames.contains($0.toolName) }) {
            return (nil, "not_allowed")
        }
        return (nil, "mismatched_selected_tools")
    }

    private static func addEvidenceAudit(
        to auditPayload: inout [String: AuditValue],
        match: RouteTrainingPairMatch,
        planStatus: String
    ) {
        let pair = match.pair
        auditPayload["route_pair_selected_tool_names"] = .array(pair.selectedToolNames.map { .string($0) })
        auditPayload["route_pair_source"] = .string(pair.source.rawValue)
        auditPayload["route_pair_match_mode"] = .string(match.mode.rawValue)
        auditPayload["route_pair_match_score"] = .double(match.score)
        auditPayload["route_pair_runner_up_score"] = .double(match.runnerUpScore)
        auditPayload["route_pair_tool_plan_status"] = .string(planStatus)
        if let plans = pair.toolCallPlan, !plans.isEmpty {
            auditPayload["route_pair_tool_plan_names"] = .array(plans.map { .string($0.toolName) })
        }
    }

    private static func intent(
        for pair: RouteTrainingPair,
        selectedPlan: ToolCallPlan?
    ) -> PersonalIntent {
        switch pair.expectedIntentTag {
        case .exactFact:
            return .exactFact(plan: FactQueryPlan())
        case .aggregateFact:
            return .aggregateFact(plan: FactQueryPlan())
        case .userProfile:
            return .userProfile(detail: ProfileDetail(kind: .summary))
        case .appAction:
            let name = selectedPlan?.toolName ?? pair.selectedToolNames.first ?? "app_action"
            return .appAction(plan: ActionPlan(
                name: name,
                arguments: selectedPlan?.arguments ?? [:],
                requiresConfirmation: true
            ))
        case .baseChat:
            return .baseChat
        case .mixed:
            return .mixed(candidates: [
                .exactFact(plan: FactQueryPlan()),
                .userProfile(detail: ProfileDetail(kind: .summary)),
                .baseChat,
            ])
        }
    }

    private static func fallbackChain(for tag: PersonalIntentTag) -> [PersonalIntentTag] {
        switch tag {
        case .exactFact:
            return [.exactFact, .baseChat]
        case .aggregateFact:
            return [.aggregateFact, .baseChat]
        case .userProfile:
            return [.userProfile, .baseChat]
        case .appAction:
            return [.appAction, .baseChat]
        case .baseChat:
            return [.baseChat]
        case .mixed:
            return [.exactFact, .userProfile, .baseChat]
        }
    }

    private static func routeConfidence(for match: RouteTrainingPairMatch) -> Double {
        let pairConfidence = min(1.0, max(0.0, match.pair.confidence))
        switch match.mode {
        case .exact:
            return min(0.99, pairConfidence)
        case .similarity:
            return min(0.92, pairConfidence, max(0.72, match.score))
        }
    }
}

/// Router wrapper that applies matched route/action evidence after a base
/// router decision. This is still evidence replay, not a learned router.
public struct RouteTrainingPairEvidenceRouter<Base: PersonalIntentRouter>: PersonalIntentRouter {
    private let baseRouter: Base
    private let matcher: RouteTrainingPairMatcher
    private let policy: RouteTrainingPairDecisionPolicy

    public init(
        baseRouter: Base,
        routeTrainingPairs: [RouteTrainingPair],
        options: RouteTrainingPairMatcherOptions = RouteTrainingPairMatcherOptions(),
        policy: RouteTrainingPairDecisionPolicy = RouteTrainingPairDecisionPolicy()
    ) {
        self.baseRouter = baseRouter
        self.matcher = RouteTrainingPairMatcher(pairs: routeTrainingPairs, options: options)
        self.policy = policy
    }

    public func route(_ input: UserInputContext) async throws -> RouteDecision {
        let decision = try await baseRouter.route(input)
        guard let match = matcher.match(input.text) else {
            return decision
        }
        return RouteTrainingPairDecisionApplier.apply(
            match: match,
            to: decision,
            policy: policy
        )
    }
}
