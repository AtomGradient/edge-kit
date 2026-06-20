// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public enum RouteMatrixControlledLiveError: Error, Equatable, Sendable {
    case probeTimedOut
}

public struct RouteMatrixPredictionAudit: Codable, Equatable, Sendable {
    public var intent: String
    public var probability: Double
    public var threshold: Double
    public var thresholdPassed: Bool
    public var trainingRunID: String?

    enum CodingKeys: String, CodingKey {
        case intent
        case probability
        case threshold
        case thresholdPassed = "threshold_passed"
        case trainingRunID = "training_run_id"
    }

    public init(
        intent: String,
        probability: Double,
        threshold: Double,
        thresholdPassed: Bool,
        trainingRunID: String?
    ) {
        self.intent = intent
        self.probability = probability
        self.threshold = threshold
        self.thresholdPassed = thresholdPassed
        self.trainingRunID = trainingRunID
    }
}

public struct RouteMatrixUserCorrection: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.route_matrix_user_correction.v0"

    public var schemaVersion: String
    public var sourceInputText: String
    public var correctionText: String
    public var correctionSource: String
    public var isFixture: Bool
    public var createdAtMs: Int?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sourceInputText = "source_input_text"
        case correctionText = "correction_text"
        case correctionSource = "correction_source"
        case isFixture = "is_fixture"
        case createdAtMs = "created_at_ms"
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        sourceInputText: String,
        correctionText: String,
        correctionSource: String = "user",
        isFixture: Bool = false,
        createdAtMs: Int? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.sourceInputText = sourceInputText.trimmingCharacters(in: .whitespacesAndNewlines)
        self.correctionText = correctionText.trimmingCharacters(in: .whitespacesAndNewlines)
        self.correctionSource = correctionSource.trimmingCharacters(in: .whitespacesAndNewlines)
        self.isFixture = isFixture
        self.createdAtMs = createdAtMs
    }

    public var auditPayload: [String: AuditValue] {
        var payload: [String: AuditValue] = [
            "schema_version": .string(schemaVersion),
            "source_input_text": .string(sourceInputText),
            "correction_text": .string(correctionText),
            "correction_source": .string(correctionSource),
            "is_fixture": .bool(isFixture),
        ]
        if let createdAtMs {
            payload["created_at_ms"] = .int(createdAtMs)
        }
        return payload
    }
}

public struct RouteMatrixLiveDecisionAudit: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.route_matrix_live_decision_audit.v0"

    public var schemaVersion: String
    public var caseID: String
    public var matrixPrediction: RouteMatrixPredictionAudit
    public var matrixCalibratedConfidence: Double
    public var evidenceAvailable: Bool
    public var evidenceRoute: [String: AuditValue]?
    public var finalDecisionSource: String
    public var fallbackReason: String?
    public var shadowModeWas: Bool
    public var userCorrection: [String: AuditValue]?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case caseID = "case_id"
        case matrixPrediction = "matrix_prediction"
        case matrixCalibratedConfidence = "matrix_calibrated_confidence"
        case evidenceAvailable = "evidence_available"
        case evidenceRoute = "evidence_route"
        case finalDecisionSource = "final_decision_source"
        case fallbackReason = "fallback_reason"
        case shadowModeWas = "shadow_mode_was"
        case userCorrection = "user_correction"
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        caseID: String,
        matrixPrediction: RouteMatrixPredictionAudit,
        matrixCalibratedConfidence: Double,
        evidenceAvailable: Bool,
        evidenceRoute: [String: AuditValue]?,
        finalDecisionSource: String,
        fallbackReason: String?,
        shadowModeWas: Bool,
        userCorrection: [String: AuditValue]? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.caseID = caseID
        self.matrixPrediction = matrixPrediction
        self.matrixCalibratedConfidence = matrixCalibratedConfidence
        self.evidenceAvailable = evidenceAvailable
        self.evidenceRoute = evidenceRoute
        self.finalDecisionSource = finalDecisionSource
        self.fallbackReason = fallbackReason
        self.shadowModeWas = shadowModeWas
        self.userCorrection = userCorrection
    }

    public func withUserCorrection(_ correction: RouteMatrixUserCorrection) -> Self {
        var copy = self
        copy.userCorrection = correction.auditPayload
        return copy
    }
}

public protocol RouteMatrixLiveDecisionAuditSink: Sendable {
    func record(_ audit: RouteMatrixLiveDecisionAudit) async
}

public actor RouteMatrixLiveDecisionAuditMemoryStore: RouteMatrixLiveDecisionAuditSink {
    public static let shared = RouteMatrixLiveDecisionAuditMemoryStore()

    private let maxCount: Int
    private var auditsByCaseID: [String: RouteMatrixLiveDecisionAudit] = [:]
    private var orderedCaseIDs: [String] = []

    public init(maxCount: Int = 100) {
        self.maxCount = max(1, maxCount)
    }

    public func record(_ audit: RouteMatrixLiveDecisionAudit) async {
        if auditsByCaseID[audit.caseID] == nil {
            orderedCaseIDs.append(audit.caseID)
        }
        auditsByCaseID[audit.caseID] = audit
        trimToLimit()
    }

    public func latestAudit() -> RouteMatrixLiveDecisionAudit? {
        guard let caseID = orderedCaseIDs.last else { return nil }
        return auditsByCaseID[caseID]
    }

    public func audit(caseID: String) -> RouteMatrixLiveDecisionAudit? {
        auditsByCaseID[caseID]
    }

    public func correctedLatestAudit(_ correction: RouteMatrixUserCorrection) -> RouteMatrixLiveDecisionAudit? {
        guard let latest = latestAudit() else { return nil }
        let corrected = latest.withUserCorrection(correction)
        auditsByCaseID[latest.caseID] = corrected
        return corrected
    }

    public func correctedAudit(
        caseID: String,
        correction: RouteMatrixUserCorrection
    ) -> RouteMatrixLiveDecisionAudit? {
        guard let audit = auditsByCaseID[caseID] else { return nil }
        let corrected = audit.withUserCorrection(correction)
        auditsByCaseID[caseID] = corrected
        return corrected
    }

    public func clear() {
        auditsByCaseID.removeAll(keepingCapacity: false)
        orderedCaseIDs.removeAll(keepingCapacity: false)
    }

    private func trimToLimit() {
        while orderedCaseIDs.count > maxCount {
            let caseID = orderedCaseIDs.removeFirst()
            auditsByCaseID.removeValue(forKey: caseID)
        }
    }
}

public struct RouteMatrixControlledLiveRouter<Base: PersonalIntentRouter>: PersonalIntentRouter {
    private let fallbackRouter: Base
    private let manifest: RouteRouterManifest?
    private let calibration: RouteRouterCalibration?
    private let policy: RouteMatrixLivePolicy?
    private let runtimeContext: RouteMatrixRuntimeContext
    private let adapterID: String
    private let probe: (any RouteMatrixShadowProbe)?
    private let toolRegistry: ToolRegistry
    private let auditSink: (any RouteMatrixLiveDecisionAuditSink)?
    private let probeTimeoutSeconds: Double

    public init(
        fallbackRouter: Base,
        manifest: RouteRouterManifest?,
        calibration: RouteRouterCalibration?,
        policy: RouteMatrixLivePolicy?,
        runtimeContext: RouteMatrixRuntimeContext,
        adapterID: String,
        probe: (any RouteMatrixShadowProbe)?,
        toolRegistry: ToolRegistry = .shared,
        auditSink: (any RouteMatrixLiveDecisionAuditSink)? = nil,
        probeTimeoutSeconds: Double = 0.05
    ) {
        self.fallbackRouter = fallbackRouter
        self.manifest = manifest
        self.calibration = calibration
        self.policy = policy
        self.runtimeContext = runtimeContext
        self.adapterID = adapterID
        self.probe = probe
        self.toolRegistry = toolRegistry
        self.auditSink = auditSink
        self.probeTimeoutSeconds = probeTimeoutSeconds
    }

    public func route(_ input: UserInputContext) async throws -> RouteDecision {
        let fallbackDecision = try await fallbackRouter.route(input)

        guard let policy else {
            return Self.fallback(
                fallbackDecision,
                status: "fallback_policy_missing",
                reason: "route_matrix_live_policy_missing"
            )
        }
        guard policy.controls.enabled else {
            return Self.fallback(
                fallbackDecision,
                status: "excluded_live_disabled",
                reason: "route_matrix_live_disabled"
            )
        }
        if policy.summary?.readyForLiveRouting == false {
            return Self.fallback(
                fallbackDecision,
                status: "blocked_live_policy_not_ready",
                reason: policy.summary?.readyForLiveRoutingReason ?? "route_matrix_live_policy_not_ready"
            )
        }
        guard let manifest else {
            return Self.fallback(
                fallbackDecision,
                status: "fallback_manifest_missing",
                reason: "route_matrix_manifest_missing"
            )
        }
        guard let calibration else {
            return Self.fallback(
                fallbackDecision,
                status: "fallback_calibration_missing",
                reason: "route_matrix_calibration_missing",
                manifest: manifest
            )
        }

        do {
            try manifest.validate(
                expectedBaseModelID: runtimeContext.baseModelID,
                expectedTokenizerSHA256: runtimeContext.tokenizerSHA256,
                currentRuntimeVersion: runtimeContext.runtimeVersion
            )
            try calibration.validate(intentVocab: manifest.intentVocab)
        } catch {
            return Self.fallback(
                fallbackDecision,
                status: "fallback_contract_rejected",
                reason: String(describing: error),
                manifest: manifest
            )
        }

        let missingToolRouteIntents = runtimeContext.registeredToolNames
            .filter { runtimeContext.toolRouteIntents[$0]?.isEmpty ?? true }
            .sorted()
        if !missingToolRouteIntents.isEmpty {
            return Self.fallback(
                fallbackDecision,
                status: "fallback_missing_tool_route_intents",
                reason: "route_matrix_tool_route_intents_missing",
                manifest: manifest
            )
        }

        guard let probe else {
            return Self.fallbackWithUnavailablePrediction(
                fallbackDecision,
                status: "fallback_probe_missing",
                reason: "route_matrix_live_probe_missing",
                input: input,
                manifest: manifest,
                auditSink: auditSink
            )
        }
        if let reason = Self.probeRejectionReason(
            supportedEncoder: probe.supportedEncoder,
            encoder: manifest.encoder
        ) {
            return Self.fallbackWithUnavailablePrediction(
                fallbackDecision,
                status: "fallback_probe_rejected",
                reason: reason,
                input: input,
                manifest: manifest,
                auditSink: auditSink
            )
        }

        let probeResult: RouteMatrixShadowProbeResult
        do {
            probeResult = try await Self.probeWithTimeout(
                probe: probe,
                input: input,
                manifest: manifest,
                calibration: calibration,
                timeoutSeconds: probeTimeoutSeconds
            )
        } catch {
            return Self.fallbackWithUnavailablePrediction(
                fallbackDecision,
                status: "fallback_probe_failed",
                reason: String(describing: error),
                input: input,
                manifest: manifest,
                auditSink: auditSink
            )
        }

        return Self.applyMatrixPrediction(
            probeResult.prediction,
            to: fallbackDecision,
            input: input,
            manifest: manifest,
            policy: policy,
            adapterID: adapterID,
            toolRegistry: toolRegistry,
            auditSink: auditSink
        )
    }

    private static func applyMatrixPrediction(
        _ prediction: RouteMatrixIntentPrediction,
        to fallbackDecision: RouteDecision,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        policy: RouteMatrixLivePolicy,
        adapterID: String,
        toolRegistry: ToolRegistry,
        auditSink: (any RouteMatrixLiveDecisionAuditSink)?
    ) -> RouteDecision {
        guard let intentTag = prediction.intentTag else {
            return fallbackWithPrediction(
                fallbackDecision,
                prediction: prediction,
                input: input,
                manifest: manifest,
                status: "blocked_unknown_matrix_intent",
                reason: "route_matrix_live_unknown_intent",
                auditSink: auditSink
            )
        }

        let selectedToolNames = fallbackDecision.toolPlan.map { [$0.toolName] } ?? []
        let eligibility = policy.eligibility(
            appID: input.appID,
            adapterID: adapterID,
            inputSHA256: inputSHA256(for: input),
            intentTag: intentTag,
            selectedToolNames: selectedToolNames,
            thresholdPassed: prediction.thresholdPassed
        )
        guard eligibility == .eligible else {
            return fallbackWithPrediction(
                fallbackDecision,
                prediction: prediction,
                input: input,
                manifest: manifest,
                status: eligibility.status,
                reason: eligibility.status,
                auditSink: auditSink
            )
        }

        if intentTag == .baseChat {
            guard fallbackDecision.intent.tag == .baseChat,
                  fallbackDecision.toolPlan == nil else {
                return fallbackWithPrediction(
                    fallbackDecision,
                    prediction: prediction,
                    input: input,
                    manifest: manifest,
                    status: "blocked_intent_disagreement",
                    reason: "route_matrix_live_base_chat_evidence_disagreement",
                    auditSink: auditSink
                )
            }
            var decision = RouteDecision(
                intent: .baseChat,
                confidence: prediction.probability,
                reason: "route_matrix_live_base_chat",
                fallbackChain: [.baseChat],
                auditPayload: fallbackDecision.auditPayload
            )
            addLiveAudit(
                to: &decision,
                status: "applied",
                reason: "route_matrix_live_applied",
                prediction: prediction,
                manifest: manifest,
                finalDecisionSource: "matrix",
                fallbackReason: nil
            )
            recordDecisionAudit(
                prediction: prediction,
                input: input,
                manifest: manifest,
                fallbackDecision: fallbackDecision,
                finalDecisionSource: "matrix",
                fallbackReason: nil,
                auditSink: auditSink
            )
            return decision
        }

        guard let selectedPlan = fallbackDecision.toolPlan else {
            return fallbackWithPrediction(
                fallbackDecision,
                prediction: prediction,
                input: input,
                manifest: manifest,
                status: "blocked_missing_runtime_tool_plan",
                reason: "route_matrix_live_missing_runtime_tool_plan",
                auditSink: auditSink
            )
        }
        guard let toolValidationReason = validateToolPlan(
            selectedPlan,
            in: toolRegistry
        ) else {
            var decision = RouteDecision(
                intent: intent(for: intentTag, selectedPlan: selectedPlan),
                confidence: prediction.probability,
                reason: "route_matrix_live_\(intentTag.rawValue)",
                toolPlan: selectedPlan,
                personaSignal: fallbackDecision.personaSignal,
                fallbackChain: fallbackChain(for: intentTag),
                auditPayload: fallbackDecision.auditPayload
            )
            addLiveAudit(
                to: &decision,
                status: "applied",
                reason: "route_matrix_live_applied",
                prediction: prediction,
                manifest: manifest,
                finalDecisionSource: "matrix",
                fallbackReason: nil
            )
            recordDecisionAudit(
                prediction: prediction,
                input: input,
                manifest: manifest,
                fallbackDecision: fallbackDecision,
                finalDecisionSource: "matrix",
                fallbackReason: nil,
                auditSink: auditSink
            )
            return decision
        }
        return fallbackWithPrediction(
            fallbackDecision,
            prediction: prediction,
            input: input,
            manifest: manifest,
            status: "blocked_runtime_tool_plan_validation",
            reason: toolValidationReason,
            auditSink: auditSink
        )
    }

    /// Returns nil when the plan is valid; otherwise returns a fail-closed reason.
    private static func validateToolPlan(
        _ plan: ToolCallPlan,
        in registry: ToolRegistry
    ) -> String? {
        guard let metadata = registry.metadata(named: plan.toolName) else {
            return "tool_not_registered"
        }
        guard let allowedArgumentNames = allowedArgumentNames(for: metadata.argumentsSchema) else {
            return "schema_validation_failed"
        }
        let argumentNames = Set(plan.arguments.keys)
        guard argumentNames.isSubset(of: allowedArgumentNames) else {
            return "schema_validation_failed"
        }
        return nil
    }

    private static func allowedArgumentNames(for schema: ToolArgumentSchema) -> Set<String>? {
        switch schema {
        case .auditMap(let value):
            return Set(value.keys)
        case .jsonSchema(let text):
            guard let data = text.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let properties = object["properties"] as? [String: Any] else {
                return nil
            }
            return Set(properties.keys)
        }
    }

    private static func fallback(
        _ fallbackDecision: RouteDecision,
        status: String,
        reason: String,
        manifest: RouteRouterManifest? = nil
    ) -> RouteDecision {
        var decision = fallbackDecision
        addLiveAudit(
            to: &decision,
            status: status,
            reason: reason,
            prediction: nil,
            manifest: manifest,
            finalDecisionSource: finalDecisionSource(for: fallbackDecision),
            fallbackReason: reason
        )
        return decision
    }

    private static func fallbackWithPrediction(
        _ fallbackDecision: RouteDecision,
        prediction: RouteMatrixIntentPrediction,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        status: String,
        reason: String,
        auditSink: (any RouteMatrixLiveDecisionAuditSink)?
    ) -> RouteDecision {
        var decision = fallbackDecision
        addLiveAudit(
            to: &decision,
            status: status,
            reason: reason,
            prediction: prediction,
            manifest: manifest,
            finalDecisionSource: finalDecisionSource(for: fallbackDecision),
            fallbackReason: reason
        )
        recordDecisionAudit(
            prediction: prediction,
            input: input,
            manifest: manifest,
            fallbackDecision: fallbackDecision,
            finalDecisionSource: finalDecisionSource(for: fallbackDecision),
            fallbackReason: reason,
            auditSink: auditSink
        )
        return decision
    }

    private static func fallbackWithUnavailablePrediction(
        _ fallbackDecision: RouteDecision,
        status: String,
        reason: String,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        auditSink: (any RouteMatrixLiveDecisionAuditSink)?
    ) -> RouteDecision {
        fallbackWithPrediction(
            fallbackDecision,
            prediction: RouteMatrixIntentPrediction(
                label: "unavailable",
                intentTag: nil,
                probability: 0,
                threshold: 1,
                thresholdPassed: false,
                probabilitiesByIntent: [:]
            ),
            input: input,
            manifest: manifest,
            status: status,
            reason: reason,
            auditSink: auditSink
        )
    }

    private static func addLiveAudit(
        to decision: inout RouteDecision,
        status: String,
        reason: String,
        prediction: RouteMatrixIntentPrediction?,
        manifest: RouteRouterManifest?,
        finalDecisionSource: String,
        fallbackReason: String?
    ) {
        decision.auditPayload["route_matrix_live_status"] = .string(status)
        decision.auditPayload["route_matrix_live_reason"] = .string(reason)
        decision.auditPayload["route_matrix_live_final_decision_source"] = .string(finalDecisionSource)
        decision.auditPayload["route_matrix_live_shadow_mode_was"] = .bool(true)
        if let fallbackReason {
            decision.auditPayload["route_matrix_live_fallback_reason"] = .string(fallbackReason)
        }
        if let manifest {
            decision.auditPayload["route_matrix_live_training_run_id"] = .string(manifest.trainingRunID)
        }
        if let prediction {
            decision.auditPayload["route_matrix_live_matrix_intent"] = .string(prediction.label)
            decision.auditPayload["route_matrix_live_matrix_probability"] = .double(prediction.probability)
            decision.auditPayload["route_matrix_live_matrix_threshold"] = .double(prediction.threshold)
            decision.auditPayload["route_matrix_live_matrix_threshold_passed"] = .bool(prediction.thresholdPassed)
        }
    }

    private static func recordDecisionAudit(
        prediction: RouteMatrixIntentPrediction,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        fallbackDecision: RouteDecision,
        finalDecisionSource: String,
        fallbackReason: String?,
        auditSink: (any RouteMatrixLiveDecisionAuditSink)?
    ) {
        guard let auditSink else { return }
        let audit = RouteMatrixLiveDecisionAudit(
            caseID: caseID(for: input),
            matrixPrediction: RouteMatrixPredictionAudit(
                intent: prediction.label,
                probability: prediction.probability,
                threshold: prediction.threshold,
                thresholdPassed: prediction.thresholdPassed,
                trainingRunID: manifest.trainingRunID
            ),
            matrixCalibratedConfidence: prediction.probability,
            evidenceAvailable: evidenceAvailable(fallbackDecision),
            evidenceRoute: evidenceRoute(fallbackDecision),
            finalDecisionSource: finalDecisionSource,
            fallbackReason: fallbackReason,
            shadowModeWas: true,
            userCorrection: nil
        )
        Task.detached(priority: .utility) {
            await auditSink.record(audit)
        }
    }

    private static func finalDecisionSource(for decision: RouteDecision) -> String {
        if decision.reason.hasPrefix("route_pair_") || decision.toolPlan != nil {
            return "evidence"
        }
        return "base"
    }

    private static func evidenceAvailable(_ decision: RouteDecision) -> Bool {
        decision.reason.hasPrefix("route_pair_") || decision.toolPlan != nil
    }

    private static func evidenceRoute(_ decision: RouteDecision) -> [String: AuditValue]? {
        guard evidenceAvailable(decision) else { return nil }
        var route: [String: AuditValue] = [
            "intent": .string(decision.intent.tag.rawValue),
            "reason": .string(decision.reason),
        ]
        if let toolPlan = decision.toolPlan {
            route["tool_name"] = .string(toolPlan.toolName)
        }
        return route
    }

    private static func caseID(for input: UserInputContext) -> String {
        if case let .string(value)? = input.appContext["case_id"],
           !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return value.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return "live:\(inputSHA256(for: input).prefix(32))"
    }

    private static func inputSHA256(for input: UserInputContext) -> String {
        let digest = SHA256.hash(data: Data(input.text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func intent(
        for tag: PersonalIntentTag,
        selectedPlan: ToolCallPlan
    ) -> PersonalIntent {
        switch tag {
        case .exactFact:
            return .exactFact(plan: FactQueryPlan())
        case .aggregateFact:
            return .aggregateFact(plan: FactQueryPlan())
        case .userProfile:
            return .userProfile(detail: ProfileDetail(kind: .summary))
        case .appAction:
            return .appAction(plan: ActionPlan(
                name: selectedPlan.toolName,
                arguments: selectedPlan.arguments,
                requiresConfirmation: true
            ))
        case .baseChat:
            return .baseChat
        case .mixed:
            return .mixed(candidates: [
                .exactFact(plan: FactQueryPlan()),
                .appAction(plan: ActionPlan(
                    name: selectedPlan.toolName,
                    arguments: selectedPlan.arguments,
                    requiresConfirmation: true
                )),
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
            return [.exactFact, .appAction, .baseChat]
        }
    }

    private static func probeRejectionReason(
        supportedEncoder: RouteMatrixShadowProbeSupportedEncoder,
        encoder: RouteRouterEncoderSpec
    ) -> String? {
        if !supportedEncoder.kinds.contains(encoder.kind) {
            return "route_matrix_probe_unsupported_encoder_kind"
        }
        if !supportedEncoder.poolings.contains(encoder.pooling) {
            return "route_matrix_probe_unsupported_pooling"
        }
        if let layerIndices = supportedEncoder.layerIndices,
           !layerIndices.contains(encoder.layerIndex) {
            return "route_matrix_probe_unsupported_layer_index"
        }
        return nil
    }

    private static func probeWithTimeout(
        probe: any RouteMatrixShadowProbe,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration,
        timeoutSeconds: Double
    ) async throws -> RouteMatrixShadowProbeResult {
        guard timeoutSeconds > 0 else {
            return try await probe.probe(
                input: input,
                manifest: manifest,
                calibration: calibration
            )
        }
        let timeoutNanoseconds = UInt64((timeoutSeconds * 1_000_000_000).rounded(.up))
        return try await withThrowingTaskGroup(of: RouteMatrixShadowProbeResult.self) { group in
            group.addTask {
                try await probe.probe(
                    input: input,
                    manifest: manifest,
                    calibration: calibration
                )
            }
            group.addTask {
                try await Task.sleep(nanoseconds: timeoutNanoseconds)
                throw RouteMatrixControlledLiveError.probeTimedOut
            }
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }
}
