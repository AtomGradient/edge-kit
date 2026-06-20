// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum RouteMatrixLivePolicyError: Error, Equatable, Sendable {
    case unsupportedSchemaVersion(String)
    case artifactNotReady
    case missingResult
}

public struct RouteMatrixLiveCircuitBreaker: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var sampleCount: Int
    public var minSampleCount: Int
    public var fallbackRate: Double?
    public var maxFallbackRate: Double
    public var errorRate: Double?
    public var maxErrorRate: Double
    public var status: String

    enum CodingKeys: String, CodingKey {
        case enabled
        case sampleCount = "sample_count"
        case minSampleCount = "min_sample_count"
        case fallbackRate = "fallback_rate"
        case maxFallbackRate = "max_fallback_rate"
        case errorRate = "error_rate"
        case maxErrorRate = "max_error_rate"
        case status
    }

    public init(
        enabled: Bool,
        sampleCount: Int,
        minSampleCount: Int,
        fallbackRate: Double?,
        maxFallbackRate: Double,
        errorRate: Double?,
        maxErrorRate: Double,
        status: String
    ) {
        self.enabled = enabled
        self.sampleCount = sampleCount
        self.minSampleCount = minSampleCount
        self.fallbackRate = fallbackRate
        self.maxFallbackRate = maxFallbackRate
        self.errorRate = errorRate
        self.maxErrorRate = maxErrorRate
        self.status = status
    }
}

public struct RouteMatrixLivePolicyControls: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var allowedAppIDs: [String]
    public var allowedAdapterIDs: [String]
    public var allowedInputSHA256s: [String]
    public var allowedIntents: [String]
    public var allowedTools: [String]
    public var circuitBreaker: RouteMatrixLiveCircuitBreaker

    enum CodingKeys: String, CodingKey {
        case enabled
        case allowedAppIDs = "allowed_app_ids"
        case allowedAdapterIDs = "allowed_adapter_ids"
        case allowedInputSHA256s = "allowed_input_sha256s"
        case allowedIntents = "allowed_intents"
        case allowedTools = "allowed_tools"
        case circuitBreaker = "circuit_breaker"
    }

    public init(
        enabled: Bool,
        allowedAppIDs: [String],
        allowedAdapterIDs: [String],
        allowedInputSHA256s: [String] = [],
        allowedIntents: [String],
        allowedTools: [String],
        circuitBreaker: RouteMatrixLiveCircuitBreaker
    ) {
        self.enabled = enabled
        self.allowedAppIDs = allowedAppIDs
        self.allowedAdapterIDs = allowedAdapterIDs
        self.allowedInputSHA256s = allowedInputSHA256s
        self.allowedIntents = allowedIntents
        self.allowedTools = allowedTools
        self.circuitBreaker = circuitBreaker
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decode(Bool.self, forKey: .enabled)
        allowedAppIDs = try container.decode([String].self, forKey: .allowedAppIDs)
        allowedAdapterIDs = try container.decode([String].self, forKey: .allowedAdapterIDs)
        allowedInputSHA256s = try container.decodeIfPresent(
            [String].self,
            forKey: .allowedInputSHA256s
        ) ?? []
        allowedIntents = try container.decode([String].self, forKey: .allowedIntents)
        allowedTools = try container.decode([String].self, forKey: .allowedTools)
        circuitBreaker = try container.decode(RouteMatrixLiveCircuitBreaker.self, forKey: .circuitBreaker)
    }
}

public struct RouteMatrixLivePolicySummary: Codable, Equatable, Sendable {
    public var readyForLiveRouting: Bool
    public var readyForLiveRoutingReason: String

    enum CodingKeys: String, CodingKey {
        case readyForLiveRouting = "ready_for_live_routing"
        case readyForLiveRoutingReason = "ready_for_live_routing_reason"
    }

    public init(
        readyForLiveRouting: Bool,
        readyForLiveRoutingReason: String
    ) {
        self.readyForLiveRouting = readyForLiveRouting
        self.readyForLiveRoutingReason = readyForLiveRoutingReason
    }
}

public struct RouteMatrixLivePolicy: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.route_matrix_live_policy.v0"

    public var runID: String
    public var appID: String
    public var adapterID: String
    public var controls: RouteMatrixLivePolicyControls
    public var summary: RouteMatrixLivePolicySummary?

    enum CodingKeys: String, CodingKey {
        case runID = "run_id"
        case appID = "app_id"
        case adapterID = "adapter_id"
        case controls
        case summary
    }

    public init(
        runID: String,
        appID: String,
        adapterID: String,
        controls: RouteMatrixLivePolicyControls,
        summary: RouteMatrixLivePolicySummary? = nil
    ) {
        self.runID = runID
        self.appID = appID
        self.adapterID = adapterID
        self.controls = controls
        self.summary = summary
    }

    public static func load(from url: URL) throws -> RouteMatrixLivePolicy {
        let data = try Data(contentsOf: url)
        return try decodeArtifact(data)
    }

    public static func decodeArtifact(_ data: Data) throws -> RouteMatrixLivePolicy {
        let artifact = try JSONDecoder().decode(Artifact.self, from: data)
        guard artifact.schemaVersion == Self.supportedSchemaVersion else {
            throw RouteMatrixLivePolicyError.unsupportedSchemaVersion(artifact.schemaVersion)
        }
        guard artifact.ok else {
            throw RouteMatrixLivePolicyError.artifactNotReady
        }
        guard let result = artifact.result else {
            throw RouteMatrixLivePolicyError.missingResult
        }
        return result
    }

    public func eligibility(
        appID runtimeAppID: String?,
        adapterID runtimeAdapterID: String,
        inputSHA256: String,
        intentTag: PersonalIntentTag,
        selectedToolNames: [String],
        thresholdPassed: Bool
    ) -> RouteMatrixLivePolicyEligibility {
        guard controls.enabled else {
            return .blocked("excluded_live_disabled")
        }
        guard !controls.allowedAppIDs.isEmpty else {
            return .blocked("blocked_missing_app_allowlist")
        }
        guard !controls.allowedAdapterIDs.isEmpty else {
            return .blocked("blocked_missing_adapter_allowlist")
        }
        guard !controls.allowedIntents.isEmpty else {
            return .blocked("blocked_missing_intent_allowlist")
        }
        guard let runtimeAppID,
              Set(controls.allowedAppIDs).contains(runtimeAppID) else {
            return .blocked("excluded_app_not_enabled")
        }
        guard Set(controls.allowedAdapterIDs).contains(runtimeAdapterID) else {
            return .blocked("excluded_adapter_not_enabled")
        }
        let allowedInputHashes = Set(controls.allowedInputSHA256s.map { $0.lowercased() })
        guard !allowedInputHashes.isEmpty else {
            return .blocked("blocked_missing_input_allowlist")
        }
        if !allowedInputHashes.contains(inputSHA256.lowercased()) {
            return .blocked("excluded_input_not_enabled")
        }
        let circuitStatus = controls.circuitBreaker.status
        guard circuitStatus == "ready" else {
            return .blocked(circuitStatus)
        }
        guard Set(controls.allowedIntents).contains(intentTag.rawValue) else {
            return .blocked("excluded_intent_not_enabled")
        }
        if !selectedToolNames.isEmpty {
            let allowedTools = Set(controls.allowedTools)
            guard !allowedTools.isEmpty else {
                return .blocked("blocked_missing_tool_allowlist")
            }
            guard selectedToolNames.allSatisfy({ allowedTools.contains($0) }) else {
                return .blocked("excluded_tool_not_enabled")
            }
        }
        guard thresholdPassed else {
            return .blocked("blocked_low_confidence_matrix_prediction")
        }
        return .eligible
    }

    private struct Artifact: Decodable {
        var ok: Bool
        var schemaVersion: String
        var result: RouteMatrixLivePolicy?

        enum CodingKeys: String, CodingKey {
            case ok
            case schemaVersion = "schema_version"
            case result
        }
    }
}

public enum RouteMatrixLivePolicyEligibility: Equatable, Sendable {
    case eligible
    case blocked(String)

    public var status: String {
        switch self {
        case .eligible:
            return "eligible_for_controlled_live"
        case .blocked(let reason):
            return reason
        }
    }
}
