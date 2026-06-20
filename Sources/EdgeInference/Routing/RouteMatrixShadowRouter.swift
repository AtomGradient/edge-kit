// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public struct RouteMatrixRuntimeContext: Sendable, Equatable {
    public var baseModelID: String
    public var tokenizerSHA256: String
    public var runtimeVersion: String
    public var registeredToolNames: [String]
    public var toolRouteIntents: [String: [PersonalIntentTag]]

    public init(
        baseModelID: String,
        tokenizerSHA256: String,
        runtimeVersion: String,
        registeredToolNames: [String] = [],
        toolRouteIntents: [String: [PersonalIntentTag]] = [:]
    ) {
        self.baseModelID = baseModelID
        self.tokenizerSHA256 = tokenizerSHA256
        self.runtimeVersion = runtimeVersion
        self.registeredToolNames = registeredToolNames
        self.toolRouteIntents = toolRouteIntents
    }
}

public enum RouteMatrixShadowStatus: String, Sendable, Codable, Equatable {
    case noManifest = "no_manifest"
    case noCalibration = "no_calibration"
    case manifestRejected = "manifest_rejected"
    case calibrationRejected = "calibration_rejected"
    case missingToolRouteIntents = "missing_tool_route_intents"
    case readyNotExecuted = "ready_not_executed"
    case scoringScheduled = "scoring_scheduled"
    case scoredNotApplied = "scored_not_applied"
    case scoringRejected = "scoring_rejected"
}

public struct RouteMatrixShadowProbeSupportedEncoder: Sendable, Equatable {
    public var kinds: Set<String>
    public var poolings: Set<String>
    public var layerIndices: Set<Int>?

    public init(
        kinds: Set<String>,
        poolings: Set<String>,
        layerIndices: Set<Int>? = nil
    ) {
        self.kinds = kinds
        self.poolings = poolings
        self.layerIndices = layerIndices
    }

    fileprivate func rejectionReason(for encoder: RouteRouterEncoderSpec) -> String? {
        if !kinds.contains(encoder.kind) {
            return "route_matrix_probe_unsupported_encoder_kind"
        }
        if !poolings.contains(encoder.pooling) {
            return "route_matrix_probe_unsupported_pooling"
        }
        if let layerIndices, !layerIndices.contains(encoder.layerIndex) {
            return "route_matrix_probe_unsupported_layer_index"
        }
        return nil
    }
}

public struct RouteMatrixShadowProbeResult: Sendable, Equatable {
    public var prediction: RouteMatrixIntentPrediction
    public var latencyMs: Double?

    public init(
        prediction: RouteMatrixIntentPrediction,
        latencyMs: Double? = nil
    ) {
        self.prediction = prediction
        self.latencyMs = latencyMs
    }
}

public struct RouteMatrixShadowProbeObservation: Sendable, Equatable {
    public var status: RouteMatrixShadowStatus
    public var reason: String
    public var trainingRunID: String?
    public var auditPayload: [String: AuditValue]

    public init(
        status: RouteMatrixShadowStatus,
        reason: String,
        trainingRunID: String?,
        auditPayload: [String: AuditValue]
    ) {
        self.status = status
        self.reason = reason
        self.trainingRunID = trainingRunID
        self.auditPayload = auditPayload
    }
}

public protocol RouteMatrixShadowProbe: Sendable {
    var supportedEncoder: RouteMatrixShadowProbeSupportedEncoder { get }

    func probe(
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration
    ) async throws -> RouteMatrixShadowProbeResult
}

public protocol RouteMatrixShadowProbeSink: Sendable {
    func record(_ observation: RouteMatrixShadowProbeObservation) async
}

/// Phase 3 shadow wrapper for R2.1 matrix-router artifacts.
///
/// This wrapper deliberately does not apply matrix inference to routing. It
/// validates runtime compatibility, annotates the fallback decision, and can
/// schedule optional shadow-only probe scoring through a sink so apps can
/// dogfood artifact readiness without replacing the R2 evidence/base routing
/// chain.
public struct RouteMatrixShadowRouter<Base: PersonalIntentRouter>: PersonalIntentRouter {
    private let fallbackRouter: Base
    private let manifest: RouteRouterManifest?
    private let calibration: RouteRouterCalibration?
    private let runtimeContext: RouteMatrixRuntimeContext
    private let probe: (any RouteMatrixShadowProbe)?
    private let probeSink: (any RouteMatrixShadowProbeSink)?

    public init(
        fallbackRouter: Base,
        manifest: RouteRouterManifest?,
        calibration: RouteRouterCalibration?,
        runtimeContext: RouteMatrixRuntimeContext,
        probe: (any RouteMatrixShadowProbe)? = nil,
        probeSink: (any RouteMatrixShadowProbeSink)? = nil
    ) {
        self.fallbackRouter = fallbackRouter
        self.manifest = manifest
        self.calibration = calibration
        self.runtimeContext = runtimeContext
        self.probe = probe
        self.probeSink = probeSink
    }

    public func route(_ input: UserInputContext) async throws -> RouteDecision {
        var decision = try await fallbackRouter.route(input)
        let audit = shadowAudit(for: input)
        for (key, value) in audit {
            decision.auditPayload[key] = value
        }
        return decision
    }

    private func shadowAudit(for input: UserInputContext) -> [String: AuditValue] {
        guard let manifest else {
            return Self.audit(
                status: .noManifest,
                reason: "route_matrix_manifest_missing",
                input: input
            )
        }
        guard let calibration else {
            return Self.audit(
                status: .noCalibration,
                reason: "route_matrix_calibration_missing",
                manifest: manifest,
                input: input
            )
        }

        do {
            try manifest.validate(
                expectedBaseModelID: runtimeContext.baseModelID,
                expectedTokenizerSHA256: runtimeContext.tokenizerSHA256,
                currentRuntimeVersion: runtimeContext.runtimeVersion
            )
        } catch {
            return Self.audit(
                status: .manifestRejected,
                reason: String(describing: error),
                manifest: manifest,
                input: input
            )
        }

        do {
            try calibration.validate(intentVocab: manifest.intentVocab)
        } catch {
            return Self.audit(
                status: .calibrationRejected,
                reason: String(describing: error),
                manifest: manifest,
                input: input
            )
        }

        let missingToolRouteIntents = runtimeContext.registeredToolNames
            .filter { runtimeContext.toolRouteIntents[$0]?.isEmpty ?? true }
            .sorted()
        if !missingToolRouteIntents.isEmpty {
            return Self.audit(
                status: .missingToolRouteIntents,
                reason: "route_matrix_tool_route_intents_missing",
                manifest: manifest,
                input: input,
                extra: [
                    "route_matrix_missing_tool_route_intents": .array(
                        missingToolRouteIntents.map { .string($0) }
                    )
                ]
            )
        }

        if let probe {
            if let reason = probe.supportedEncoder.rejectionReason(for: manifest.encoder) {
                return Self.audit(
                    status: .scoringRejected,
                    reason: reason,
                    manifest: manifest,
                    input: input
                )
            }
            guard let probeSink else {
                return Self.audit(
                    status: .scoringRejected,
                    reason: "route_matrix_probe_sink_missing",
                    manifest: manifest,
                    input: input
                )
            }
            Self.scheduleProbe(
                probe: probe,
                probeSink: probeSink,
                input: input,
                manifest: manifest,
                calibration: calibration
            )
            return Self.audit(
                status: .scoringScheduled,
                reason: "route_matrix_shadow_scoring_scheduled",
                manifest: manifest,
                input: input,
                extra: [
                    "route_matrix_probe_async": .bool(true)
                ]
            )
        }

        return Self.audit(
            status: .readyNotExecuted,
            reason: "route_matrix_shadow_ready_not_executed",
            manifest: manifest,
            input: input,
            extra: [
                "route_matrix_calibration_ece": calibration.calibrationECE.map(AuditValue.double) ?? .null,
                "route_matrix_calibration_set_size": .int(calibration.calibrationSetSize),
            ]
        )
    }

    private static func scheduleProbe(
        probe: any RouteMatrixShadowProbe,
        probeSink: any RouteMatrixShadowProbeSink,
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration
    ) {
        Task.detached(priority: .utility) {
            do {
                let result = try await probe.probe(
                    input: input,
                    manifest: manifest,
                    calibration: calibration
                )
                let reason = "route_matrix_shadow_scored_not_applied"
                await probeSink.record(
                    RouteMatrixShadowProbeObservation(
                        status: .scoredNotApplied,
                        reason: reason,
                        trainingRunID: manifest.trainingRunID,
                        auditPayload: audit(
                            status: .scoredNotApplied,
                            reason: reason,
                            manifest: manifest,
                            input: input,
                            extra: probeAuditPayload(result)
                        )
                    )
                )
            } catch {
                let reason = String(describing: error)
                await probeSink.record(
                    RouteMatrixShadowProbeObservation(
                        status: .scoringRejected,
                        reason: reason,
                        trainingRunID: manifest.trainingRunID,
                        auditPayload: audit(
                            status: .scoringRejected,
                            reason: reason,
                            manifest: manifest,
                            input: input
                        )
                    )
                )
            }
        }
    }

    private static func audit(
        status: RouteMatrixShadowStatus,
        reason: String,
        manifest: RouteRouterManifest? = nil,
        input: UserInputContext? = nil,
        extra: [String: AuditValue] = [:]
    ) -> [String: AuditValue] {
        var payload: [String: AuditValue] = [
            "route_matrix_shadow_enabled": .bool(true),
            "route_matrix_shadow_status": .string(status.rawValue),
            "route_matrix_shadow_reason": .string(reason),
        ]
        if let input {
            payload["route_matrix_input_sha256"] = .string(sha256Hex(input.text))
        }
        if let manifest {
            payload["route_matrix_training_run_id"] = .string(manifest.trainingRunID)
            payload["route_matrix_router_type"] = .string(manifest.routerType)
            payload["route_matrix_min_runtime_version"] = .string(manifest.minRuntimeVersion)
        }
        for (key, value) in extra {
            payload[key] = value
        }
        return payload
    }

    private static func sha256Hex(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func probeAuditPayload(
        _ result: RouteMatrixShadowProbeResult
    ) -> [String: AuditValue] {
        var payload: [String: AuditValue] = [
            "route_matrix_predicted_intent": .string(result.prediction.label),
            "route_matrix_predicted_probability": .double(result.prediction.probability),
            "route_matrix_predicted_threshold": .double(result.prediction.threshold),
            "route_matrix_predicted_threshold_passed": .bool(result.prediction.thresholdPassed),
        ]
        if let latencyMs = result.latencyMs {
            payload["route_matrix_prediction_latency_ms"] = .double(latencyMs)
        }
        return payload
    }
}
