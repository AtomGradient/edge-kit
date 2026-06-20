// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class RouteMatrixShadowRouterTests: XCTestCase {
    func test_routeMatrixShadowRouter_readyDoesNotReplaceFallbackDecision() async throws {
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact, .aggregateFact]]
            )
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("ready_not_executed"))
        XCTAssertEqual(decision.auditPayload["route_matrix_training_run_id"], .string("rrr-shadow-test"))
        XCTAssertEqual(decision.auditPayload["route_matrix_calibration_ece"], .double(0.04))
    }

    func test_routeMatrixShadowRouter_probeScoresButDoesNotReplaceFallbackDecision() async throws {
        let prediction = RouteMatrixIntentPrediction(
            label: "exact_fact",
            intentTag: .exactFact,
            probability: 0.88,
            threshold: 0.60,
            thresholdPassed: true,
            probabilitiesByIntent: ["base_chat": 0.12, "exact_fact": 0.88]
        )
        let sink = RecordingProbeSink()
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact, .aggregateFact]]
            ),
            probe: StaticProbe(result: RouteMatrixShadowProbeResult(
                prediction: prediction,
                latencyMs: 3.5
            )),
            probeSink: sink
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("scoring_scheduled"))
        XCTAssertEqual(decision.auditPayload["route_matrix_probe_async"], .bool(true))
        XCTAssertNotNil(decision.auditPayload["route_matrix_input_sha256"])
        XCTAssertNil(decision.auditPayload["route_matrix_predicted_intent"])

        let recordedObservation = await sink.waitForObservation()
        let observation = try XCTUnwrap(recordedObservation)
        XCTAssertEqual(observation.status, .scoredNotApplied)
        XCTAssertEqual(observation.trainingRunID, "rrr-shadow-test")
        XCTAssertEqual(observation.auditPayload["route_matrix_shadow_status"], .string("scored_not_applied"))
        XCTAssertEqual(
            observation.auditPayload["route_matrix_input_sha256"],
            decision.auditPayload["route_matrix_input_sha256"]
        )
        XCTAssertEqual(observation.auditPayload["route_matrix_predicted_intent"], .string("exact_fact"))
        XCTAssertEqual(observation.auditPayload["route_matrix_predicted_probability"], .double(0.88))
        XCTAssertEqual(observation.auditPayload["route_matrix_predicted_threshold"], .double(0.60))
        XCTAssertEqual(observation.auditPayload["route_matrix_predicted_threshold_passed"], .bool(true))
        XCTAssertEqual(observation.auditPayload["route_matrix_prediction_latency_ms"], .double(3.5))
    }

    func test_routeMatrixShadowRouter_probeErrorFailsClosedToFallbackDecision() async throws {
        let sink = RecordingProbeSink()
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact]]
            ),
            probe: ThrowingProbe(),
            probeSink: sink
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("scoring_scheduled"))
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_enabled"], .bool(true))

        let recordedObservation = await sink.waitForObservation()
        let observation = try XCTUnwrap(recordedObservation)
        XCTAssertEqual(observation.status, .scoringRejected)
        XCTAssertEqual(observation.auditPayload["route_matrix_shadow_status"], .string("scoring_rejected"))
        XCTAssertEqual(observation.auditPayload["route_matrix_shadow_enabled"], .bool(true))
    }

    func test_routeMatrixShadowRouter_rejectsUnsupportedProbeEncoderWithoutScheduling() async throws {
        let prediction = RouteMatrixIntentPrediction(
            label: "exact_fact",
            intentTag: .exactFact,
            probability: 0.88,
            threshold: 0.60,
            thresholdPassed: true,
            probabilitiesByIntent: ["base_chat": 0.12, "exact_fact": 0.88]
        )
        let sink = RecordingProbeSink()
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact]]
            ),
            probe: StaticProbe(
                result: RouteMatrixShadowProbeResult(prediction: prediction),
                supportedEncoder: RouteMatrixShadowProbeSupportedEncoder(
                    kinds: ["other_encoder"],
                    poolings: ["mean_excluding_special"],
                    layerIndices: [-1]
                )
            ),
            probeSink: sink
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("scoring_rejected"))
        XCTAssertEqual(
            decision.auditPayload["route_matrix_shadow_reason"],
            .string("route_matrix_probe_unsupported_encoder_kind")
        )
        let observationCount = await sink.observationCount()
        XCTAssertEqual(observationCount, 0)
    }

    func test_routeMatrixShadowRouter_rejectsProbeWithoutSink() async throws {
        let prediction = RouteMatrixIntentPrediction(
            label: "exact_fact",
            intentTag: .exactFact,
            probability: 0.88,
            threshold: 0.60,
            thresholdPassed: true,
            probabilitiesByIntent: ["base_chat": 0.12, "exact_fact": 0.88]
        )
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact]]
            ),
            probe: StaticProbe(result: RouteMatrixShadowProbeResult(prediction: prediction))
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("scoring_rejected"))
        XCTAssertEqual(
            decision.auditPayload["route_matrix_shadow_reason"],
            .string("route_matrix_probe_sink_missing")
        )
    }

    func test_routeMatrixShadowRouter_rejectsBaseModelMismatchAndFallsBack() async throws {
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "other-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses"],
                toolRouteIntents: ["query_expenses": [.exactFact]]
            )
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.intent.tag, .baseChat)
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("manifest_rejected"))
        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_enabled"], .bool(true))
    }

    func test_routeMatrixShadowRouter_rejectsMissingToolRouteIntentMetadata() async throws {
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: Self.calibration(),
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: ["query_expenses", "update_expense"],
                toolRouteIntents: ["query_expenses": [.exactFact]]
            )
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("missing_tool_route_intents"))
        XCTAssertEqual(
            decision.auditPayload["route_matrix_missing_tool_route_intents"],
            .array([.string("update_expense")])
        )
    }

    func test_routeMatrixShadowRouter_rejectsIncompleteCalibration() async throws {
        let calibration = RouteRouterCalibration(
            intentTemperature: 0.6,
            intentThresholds: ["base_chat": 0.55],
            toolThresholdDefault: 0.55,
            calibrationSetSize: 20,
            calibrationECE: 0.04
        )
        let router = RouteMatrixShadowRouter(
            fallbackRouter: Self.baseChatFallbackRouter(),
            manifest: Self.manifest(),
            calibration: calibration,
            runtimeContext: RouteMatrixRuntimeContext(
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64),
                runtimeVersion: "0.9.1",
                registeredToolNames: [],
                toolRouteIntents: [:]
            )
        )

        let decision = try await router.route(UserInputContext(text: "plain input"))

        XCTAssertEqual(decision.auditPayload["route_matrix_shadow_status"], .string("calibration_rejected"))
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
            intentVocab: ["base_chat", "exact_fact", "aggregate_fact"],
            matrices: [
                "intent": RouteRouterMatrixSpec(
                    file: "route_intent_matrix.safetensors",
                    tensor: "intent_weights",
                    biasTensor: "intent_bias",
                    shape: [4, 3],
                    dtype: "float16"
                )
            ],
            calibrationFile: "route_calibration.json",
            minRuntimeVersion: "0.9.0",
            trainingRunID: "rrr-shadow-test",
            manifestSHA256: String(repeating: "b", count: 64),
            fallbackChain: ["matrix", "evidence_matcher", "base_router"]
        )
    }

    private static func baseChatFallbackRouter() -> DefaultPersonalIntentRouter {
        DefaultPersonalIntentRouter(queryRouter: QueryRouter(
            base: MockBaseClassifier(defaultLabel: .baseChat, defaultConfidence: 0.94)
        ))
    }

    private static func calibration() -> RouteRouterCalibration {
        RouteRouterCalibration(
            intentTemperature: 0.6,
            intentThresholds: [
                "base_chat": 0.55,
                "exact_fact": 0.60,
                "aggregate_fact": 0.60,
            ],
            toolThresholdDefault: 0.55,
            calibrationSetSize: 21,
            calibrationECE: 0.04
        )
    }
}

private struct StaticProbe: RouteMatrixShadowProbe {
    var result: RouteMatrixShadowProbeResult
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
        result
    }
}

private struct ThrowingProbe: RouteMatrixShadowProbe {
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
        throw ProbeError.failed
    }
}

private enum ProbeError: Error {
    case failed
}

private actor RecordingProbeSink: RouteMatrixShadowProbeSink {
    private var observations: [RouteMatrixShadowProbeObservation] = []

    func record(_ observation: RouteMatrixShadowProbeObservation) async {
        observations.append(observation)
    }

    func waitForObservation() async -> RouteMatrixShadowProbeObservation? {
        for _ in 0..<100 {
            if let first = observations.first {
                return first
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return observations.first
    }

    func observationCount() -> Int {
        observations.count
    }
}
