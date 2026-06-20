// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import XCTest
@testable import EdgeInference

final class RouteMatrixControlledLiveRouterTests: XCTestCase {
    func test_routeMatrixLivePolicy_decodesControlsFromArtifact() throws {
        let data = """
        {
          "ok": true,
          "schema_version": "edgestudio.route_matrix_live_policy.v0",
          "result": {
            "run_id": "live-policy-001",
            "app_id": "todo.fixture",
            "adapter_id": "adapter-001",
            "controls": {
              "enabled": true,
              "allowed_app_ids": ["todo.fixture"],
              "allowed_adapter_ids": ["adapter-001"],
              "allowed_input_sha256s": ["abc123"],
              "allowed_intents": ["app_action", "base_chat"],
              "allowed_tools": ["add_task"],
              "circuit_breaker": {
                "enabled": true,
                "sample_count": 100,
                "min_sample_count": 100,
                "fallback_rate": 0.1,
                "max_fallback_rate": 0.3,
                "error_rate": 0.01,
                "max_error_rate": 0.05,
                "status": "ready"
              }
            },
            "summary": {
              "ready_for_live_routing": true,
              "ready_for_live_routing_reason": "controlled_live_policy_has_eligible_candidates"
            }
          },
          "error": null
        }
        """.data(using: .utf8)!

        let policy = try RouteMatrixLivePolicy.decodeArtifact(data)

        XCTAssertEqual(policy.runID, "live-policy-001")
        XCTAssertTrue(policy.controls.enabled)
        XCTAssertEqual(policy.controls.allowedAppIDs, ["todo.fixture"])
        XCTAssertEqual(policy.controls.allowedInputSHA256s, ["abc123"])
        XCTAssertEqual(policy.controls.allowedTools, ["add_task"])
        XCTAssertEqual(policy.controls.circuitBreaker.status, "ready")
        XCTAssertEqual(policy.summary?.readyForLiveRouting, true)
    }

    func test_controlledLiveRouter_disabledPolicyFallsBackWithoutScoring() async throws {
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.baseDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(enabled: false),
            runtimeContext: Self.runtimeContext(),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.baseChat))
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.reason, "base")
        XCTAssertEqual(decision.auditPayload["route_matrix_live_status"], .string("excluded_live_disabled"))
        XCTAssertEqual(decision.auditPayload["route_matrix_live_final_decision_source"], .string("base"))
    }

    func test_controlledLiveRouter_appliesBaseChatWhenFallbackAgrees() async throws {
        let sink = RecordingLiveAuditSink()
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.baseDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(allowedIntents: ["base_chat"], allowedTools: []),
            runtimeContext: Self.runtimeContext(),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.baseChat, probability: 0.91)),
            auditSink: sink
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.reason, "route_matrix_live_base_chat")
        XCTAssertEqual(decision.confidence, 0.91)
        XCTAssertEqual(decision.auditPayload["route_matrix_live_status"], .string("applied"))
        XCTAssertEqual(decision.auditPayload["route_matrix_live_final_decision_source"], .string("matrix"))

        let recordedAudit = await sink.waitForAudit()
        let audit = try XCTUnwrap(recordedAudit)
        XCTAssertEqual(audit.schemaVersion, RouteMatrixLiveDecisionAudit.supportedSchemaVersion)
        XCTAssertEqual(audit.caseID, "case-001")
        XCTAssertEqual(audit.matrixPrediction.intent, "base_chat")
        XCTAssertEqual(audit.finalDecisionSource, "matrix")
        XCTAssertNil(audit.fallbackReason)
    }

    func test_routeMatrixUserCorrectionAttachesNaturalLanguagePayload() throws {
        let correction = RouteMatrixUserCorrection(
            sourceInputText: "  original user request  ",
            correctionText: "  this should query the user's weekly total  ",
            createdAtMs: 1_778_400_000_000
        )
        let audit = RouteMatrixLiveDecisionAudit(
            caseID: "case-001",
            matrixPrediction: Self.predictionAudit(.baseChat),
            matrixCalibratedConfidence: 0.91,
            evidenceAvailable: false,
            evidenceRoute: nil,
            finalDecisionSource: "matrix",
            fallbackReason: nil,
            shadowModeWas: false
        ).withUserCorrection(correction)

        XCTAssertEqual(
            audit.userCorrection?["schema_version"],
            .string(RouteMatrixUserCorrection.supportedSchemaVersion)
        )
        XCTAssertEqual(audit.userCorrection?["source_input_text"], .string("original user request"))
        XCTAssertEqual(
            audit.userCorrection?["correction_text"],
            .string("this should query the user's weekly total")
        )
        XCTAssertEqual(audit.userCorrection?["correction_source"], .string("user"))
        XCTAssertEqual(audit.userCorrection?["is_fixture"], .bool(false))
        XCTAssertEqual(audit.userCorrection?["created_at_ms"], .int(1_778_400_000_000))

        let data = try JSONEncoder().encode(audit)
        let decoded = try JSONDecoder().decode(RouteMatrixLiveDecisionAudit.self, from: data)
        XCTAssertEqual(decoded, audit)
    }

    func test_routeMatrixLiveDecisionAuditMemoryStoreAttachesCorrectionToLatestAudit() async throws {
        let store = RouteMatrixLiveDecisionAuditMemoryStore(maxCount: 1)
        await store.record(RouteMatrixLiveDecisionAudit(
            caseID: "case-old",
            matrixPrediction: Self.predictionAudit(.baseChat),
            matrixCalibratedConfidence: 0.90,
            evidenceAvailable: false,
            evidenceRoute: nil,
            finalDecisionSource: "matrix",
            fallbackReason: nil,
            shadowModeWas: false
        ))
        await store.record(RouteMatrixLiveDecisionAudit(
            caseID: "case-latest",
            matrixPrediction: Self.predictionAudit(.appAction),
            matrixCalibratedConfidence: 0.93,
            evidenceAvailable: true,
            evidenceRoute: ["intent": .string("app_action")],
            finalDecisionSource: "evidence",
            fallbackReason: "schema_validation_failed",
            shadowModeWas: true
        ))

        let correction = RouteMatrixUserCorrection(
            sourceInputText: "add a recurring task",
            correctionText: "this should add a weekly task instead",
            createdAtMs: 1_778_400_000_100
        )
        let corrected = await store.correctedLatestAudit(correction)
        let oldAudit = await store.audit(caseID: "case-old")
        let latestAudit = await store.latestAudit()

        XCTAssertNil(oldAudit)
        XCTAssertEqual(corrected?.caseID, "case-latest")
        XCTAssertEqual(
            corrected?.userCorrection?["correction_text"],
            .string("this should add a weekly task instead")
        )
        XCTAssertEqual(
            latestAudit?.userCorrection?["source_input_text"],
            .string("add a recurring task")
        )
    }

    func test_controlledLiveRouter_blocksLowConfidenceMatrixPrediction() async throws {
        let sink = RecordingLiveAuditSink()
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.toolDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(allowedIntents: ["app_action"], allowedTools: ["add_task"]),
            runtimeContext: Self.runtimeContext(registeredToolNames: ["add_task"]),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(
                .appAction,
                probability: 0.50,
                threshold: 0.60,
                thresholdPassed: false
            )),
            toolRegistry: Self.toolRegistry(),
            auditSink: sink
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.reason, "route_pair_exact_app_action")
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_status"],
            .string("blocked_low_confidence_matrix_prediction")
        )
        XCTAssertEqual(decision.auditPayload["route_matrix_live_final_decision_source"], .string("evidence"))

        let recordedAudit = await sink.waitForAudit()
        let audit = try XCTUnwrap(recordedAudit)
        XCTAssertEqual(audit.finalDecisionSource, "evidence")
        XCTAssertEqual(audit.fallbackReason, "blocked_low_confidence_matrix_prediction")
    }

    func test_controlledLiveRouter_appliesToolPlanOnlyWhenRegistrySchemaValid() async throws {
        let sink = RecordingLiveAuditSink()
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.toolDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(allowedIntents: ["app_action"], allowedTools: ["add_task"]),
            runtimeContext: Self.runtimeContext(registeredToolNames: ["add_task"]),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.appAction, probability: 0.92)),
            toolRegistry: Self.toolRegistry(),
            auditSink: sink
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.intent.tag, .appAction)
        XCTAssertEqual(decision.reason, "route_matrix_live_app_action")
        XCTAssertEqual(decision.toolPlan?.toolName, "add_task")
        XCTAssertEqual(decision.auditPayload["route_matrix_live_status"], .string("applied"))

        let recordedAudit = await sink.waitForAudit()
        let audit = try XCTUnwrap(recordedAudit)
        XCTAssertEqual(audit.finalDecisionSource, "matrix")
        XCTAssertEqual(audit.evidenceRoute?["tool_name"], .string("add_task"))
    }

    func test_controlledLiveRouter_respectsInputHashAllowlist() async throws {
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.toolDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(
                allowedInputSHA256s: [String(repeating: "0", count: 64)],
                allowedIntents: ["app_action"],
                allowedTools: ["add_task"]
            ),
            runtimeContext: Self.runtimeContext(registeredToolNames: ["add_task"]),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.appAction, probability: 0.92)),
            toolRegistry: Self.toolRegistry()
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.reason, "route_pair_exact_app_action")
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_status"],
            .string("excluded_input_not_enabled")
        )
    }

    func test_controlledLiveRouter_blocksEnabledPolicyWithoutInputHashScope() async throws {
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.toolDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(
                allowedInputSHA256s: [],
                allowedIntents: ["app_action"],
                allowedTools: ["add_task"]
            ),
            runtimeContext: Self.runtimeContext(registeredToolNames: ["add_task"]),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.appAction, probability: 0.92)),
            toolRegistry: Self.toolRegistry()
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.reason, "route_pair_exact_app_action")
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_status"],
            .string("blocked_missing_input_allowlist")
        )
    }

    func test_routeMatrixInputSHA256MatchesCrossLanguageFixture() throws {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "route_matrix_input_sha256_cases",
            withExtension: "json"
        ))
        let fixture = try JSONDecoder().decode(
            RouteMatrixInputSHA256Fixture.self,
            from: Data(contentsOf: url)
        )

        XCTAssertEqual(
            fixture.contract,
            "sha256(raw_utf8(input_text)); no unicode normalization"
        )
        let casesByID = Dictionary(uniqueKeysWithValues: fixture.cases.map { ($0.id, $0) })
        for testCase in fixture.cases {
            XCTAssertEqual(
                Self.sha256Hex(testCase.inputText),
                testCase.sha256,
                testCase.id
            )
        }
        XCTAssertNotEqual(
            casesByID["nfc_cafe"].map { Array($0.inputText.utf8) },
            casesByID["nfd_cafe"].map { Array($0.inputText.utf8) }
        )
        XCTAssertNotEqual(
            casesByID["nfc_cafe"]?.sha256,
            casesByID["nfd_cafe"]?.sha256
        )
    }

    func test_controlledLiveRouter_fallsBackWhenToolPlanHasUnknownArgument() async throws {
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.toolDecision(arguments: [
                "title": .string("Pay invoice"),
                "unexpected": .string("nope"),
            ])),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(allowedIntents: ["app_action"], allowedTools: ["add_task"]),
            runtimeContext: Self.runtimeContext(registeredToolNames: ["add_task"]),
            adapterID: "adapter-001",
            probe: StaticLiveProbe(prediction: Self.prediction(.appAction, probability: 0.92)),
            toolRegistry: Self.toolRegistry()
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.reason, "route_pair_exact_app_action")
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_status"],
            .string("blocked_runtime_tool_plan_validation")
        )
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_fallback_reason"],
            .string("schema_validation_failed")
        )
    }

    func test_controlledLiveRouter_probeTimeoutFallsBack() async throws {
        let sink = RecordingLiveAuditSink()
        let router = RouteMatrixControlledLiveRouter(
            fallbackRouter: StaticIntentRouter(decision: Self.baseDecision()),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            policy: Self.policy(allowedIntents: ["base_chat"], allowedTools: []),
            runtimeContext: Self.runtimeContext(),
            adapterID: "adapter-001",
            probe: DelayedLiveProbe(delayNanoseconds: 100_000_000),
            auditSink: sink,
            probeTimeoutSeconds: 0.001
        )

        let decision = try await router.route(Self.input())

        XCTAssertEqual(decision.reason, "base")
        XCTAssertEqual(decision.auditPayload["route_matrix_live_status"], .string("fallback_probe_failed"))
        XCTAssertEqual(
            decision.auditPayload["route_matrix_live_reason"],
            .string("probeTimedOut")
        )

        let recordedAudit = await sink.waitForAudit()
        let audit = try XCTUnwrap(recordedAudit)
        XCTAssertEqual(audit.matrixPrediction.intent, "unavailable")
        XCTAssertEqual(audit.matrixPrediction.probability, 0)
        XCTAssertEqual(audit.matrixPrediction.threshold, 1)
        XCTAssertEqual(audit.matrixPrediction.thresholdPassed, false)
        XCTAssertEqual(audit.finalDecisionSource, "base")
        XCTAssertEqual(audit.fallbackReason, "probeTimedOut")
    }

    private static func input() -> UserInputContext {
        UserInputContext(
            text: "Add a task",
            appID: "todo.fixture",
            appContext: ["case_id": .string("case-001")]
        )
    }

    private static func baseDecision() -> RouteDecision {
        RouteDecision(
            intent: .baseChat,
            confidence: 0.71,
            reason: "base",
            fallbackChain: [.baseChat]
        )
    }

    private static func toolDecision(
        arguments: [String: AuditValue] = ["title": .string("Pay invoice")]
    ) -> RouteDecision {
        RouteDecision(
            intent: .appAction(plan: ActionPlan(
                name: "add_task",
                arguments: arguments,
                requiresConfirmation: true
            )),
            confidence: 0.88,
            reason: "route_pair_exact_app_action",
            toolPlan: ToolCallPlan(
                toolName: "add_task",
                arguments: arguments,
                reason: "test"
            ),
            fallbackChain: [.appAction, .baseChat]
        )
    }

    private static func prediction(
        _ tag: PersonalIntentTag,
        probability: Double = 0.88,
        threshold: Double = 0.60,
        thresholdPassed: Bool = true
    ) -> RouteMatrixIntentPrediction {
        RouteMatrixIntentPrediction(
            label: tag.rawValue,
            intentTag: tag,
            probability: probability,
            threshold: threshold,
            thresholdPassed: thresholdPassed,
            probabilitiesByIntent: [tag.rawValue: probability]
        )
    }

    private static func predictionAudit(_ tag: PersonalIntentTag) -> RouteMatrixPredictionAudit {
        RouteMatrixPredictionAudit(
            intent: tag.rawValue,
            probability: 0.91,
            threshold: 0.60,
            thresholdPassed: true,
            trainingRunID: "rrr-live"
        )
    }

    private static func policy(
        enabled: Bool = true,
        allowedInputSHA256s: [String]? = nil,
        allowedIntents: [String] = ["base_chat", "app_action"],
        allowedTools: [String] = ["add_task"]
    ) -> RouteMatrixLivePolicy {
        RouteMatrixLivePolicy(
            runID: "live-policy-001",
            appID: "todo.fixture",
            adapterID: "adapter-001",
            controls: RouteMatrixLivePolicyControls(
                enabled: enabled,
                allowedAppIDs: ["todo.fixture"],
                allowedAdapterIDs: ["adapter-001"],
                allowedInputSHA256s: allowedInputSHA256s ?? [Self.inputSHA256],
                allowedIntents: allowedIntents,
                allowedTools: allowedTools,
                circuitBreaker: RouteMatrixLiveCircuitBreaker(
                    enabled: true,
                    sampleCount: 100,
                    minSampleCount: 100,
                    fallbackRate: 0.05,
                    maxFallbackRate: 0.30,
                    errorRate: 0.01,
                    maxErrorRate: 0.05,
                    status: "ready"
                )
            ),
            summary: RouteMatrixLivePolicySummary(
                readyForLiveRouting: enabled,
                readyForLiveRoutingReason: enabled
                    ? "controlled_live_policy_has_eligible_candidates"
                    : "live_routing_disabled"
            )
        )
    }

    private static func manifest() -> RouteRouterManifest {
        RouteRouterManifest(
            encoder: RouteRouterEncoderSpec(
                kind: "base_model_last_hidden",
                hiddenSize: 4,
                layerIndex: -1,
                pooling: "mean_excluding_special",
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64)
            ),
            intentVocab: ["base_chat", "app_action"],
            matrices: [
                "intent": RouteRouterMatrixSpec(
                    file: "route_intent_matrix.safetensors",
                    tensor: "intent_weights",
                    biasTensor: "intent_bias",
                    shape: [4, 2],
                    dtype: "float16"
                )
            ],
            calibrationFile: "route_calibration.json",
            minRuntimeVersion: "0.9.0",
            trainingRunID: "rrr-live-test",
            manifestSHA256: String(repeating: "b", count: 64),
            fallbackChain: ["matrix", "evidence_matcher", "base_router"]
        )
    }

    private static func calibration() -> RouteRouterCalibration {
        RouteRouterCalibration(
            intentTemperature: 0.7,
            intentThresholds: [
                "base_chat": 0.55,
                "app_action": 0.60,
            ],
            toolThresholdDefault: 0.55,
            calibrationSetSize: 30,
            calibrationECE: 0.04
        )
    }

    private static func runtimeContext(
        registeredToolNames: [String] = []
    ) -> RouteMatrixRuntimeContext {
        RouteMatrixRuntimeContext(
            baseModelID: "qwen3.5-4b-base",
            tokenizerSHA256: String(repeating: "a", count: 64),
            runtimeVersion: "0.9.1",
            registeredToolNames: registeredToolNames,
            toolRouteIntents: Dictionary(
                uniqueKeysWithValues: registeredToolNames.map { ($0, [PersonalIntentTag.appAction]) }
            )
        )
    }

    private static func toolRegistry() -> ToolRegistry {
        let registry = ToolRegistry()
        registry.register(RegisteredTool(
            metadata: ToolMetadata(
                name: "add_task",
                description: "Add a task",
                argumentsSchema: .jsonSchema("""
                {
                  "type": "object",
                  "properties": {
                    "title": { "type": "string" }
                  }
                }
                """),
                permissions: [.appAction],
                intentTags: [.appAction]
            )
        ) { _ in
            "ok"
        })
        return registry
    }

    private static let inputSHA256 =
        "bbad3c9c6d8e243d4bb396ec31101228be26bc211a6e7550d88f3d81605983e0"

    private static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

private struct RouteMatrixInputSHA256Fixture: Decodable {
    var schemaVersion: String
    var contract: String
    var cases: [RouteMatrixInputSHA256Case]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case contract
        case cases
    }
}

private struct RouteMatrixInputSHA256Case: Decodable {
    var id: String
    var inputText: String
    var sha256: String

    enum CodingKeys: String, CodingKey {
        case id
        case inputText = "input_text"
        case sha256
    }
}

private struct StaticIntentRouter: PersonalIntentRouter {
    var decision: RouteDecision

    func route(_ input: UserInputContext) async throws -> RouteDecision {
        decision
    }
}

private struct StaticLiveProbe: RouteMatrixShadowProbe {
    var prediction: RouteMatrixIntentPrediction
    var supportedEncoder = RouteMatrixShadowProbeSupportedEncoder(
        kinds: ["base_model_last_hidden"],
        poolings: ["mean_excluding_special"],
        layerIndices: [-1]
    )

    func probe(
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration
    ) async throws -> RouteMatrixShadowProbeResult {
        RouteMatrixShadowProbeResult(prediction: prediction, latencyMs: 1.0)
    }
}

private struct DelayedLiveProbe: RouteMatrixShadowProbe {
    var delayNanoseconds: UInt64
    var supportedEncoder = RouteMatrixShadowProbeSupportedEncoder(
        kinds: ["base_model_last_hidden"],
        poolings: ["mean_excluding_special"],
        layerIndices: [-1]
    )

    func probe(
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration
    ) async throws -> RouteMatrixShadowProbeResult {
        try await Task.sleep(nanoseconds: delayNanoseconds)
        return RouteMatrixShadowProbeResult(
            prediction: RouteMatrixIntentPrediction(
                label: "base_chat",
                intentTag: .baseChat,
                probability: 0.88,
                threshold: 0.60,
                thresholdPassed: true,
                probabilitiesByIntent: ["base_chat": 0.88]
            )
        )
    }
}

private actor RecordingLiveAuditSink: RouteMatrixLiveDecisionAuditSink {
    private var audits: [RouteMatrixLiveDecisionAudit] = []

    func record(_ audit: RouteMatrixLiveDecisionAudit) async {
        audits.append(audit)
    }

    func waitForAudit() async -> RouteMatrixLiveDecisionAudit? {
        for _ in 0..<100 {
            if let first = audits.first {
                return first
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return audits.first
    }
}
