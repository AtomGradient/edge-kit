// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation

public enum NativeRuntimeReadinessGateStatus: String, Codable, Equatable, Sendable {
    case passed
    case pending
    case blocked
}

public struct NativeRuntimeReadinessGate: Codable, Equatable, Sendable {
    public let id: String
    public let status: NativeRuntimeReadinessGateStatus
    public let reason: String
    public let requiredEvidence: String

    public init(
        id: String,
        status: NativeRuntimeReadinessGateStatus,
        reason: String,
        requiredEvidence: String
    ) {
        self.id = id
        self.status = status
        self.reason = reason
        self.requiredEvidence = requiredEvidence
    }
}

public struct NativeRuntimeDeferredCapability: Codable, Equatable, Sendable {
    public let id: String
    public let legacyFallbackBehavior: String
    public let nativeRuntimeMigration: String

    public init(
        id: String,
        legacyFallbackBehavior: String,
        nativeRuntimeMigration: String
    ) {
        self.id = id
        self.legacyFallbackBehavior = legacyFallbackBehavior
        self.nativeRuntimeMigration = nativeRuntimeMigration
    }
}

public struct NativeRuntimeReadinessReport: Codable, Equatable, Sendable {
    public let runtimeVersion: String
    public let nativeMetalSchedulingAppliesToNativeRuntime: Bool

    /// Until native runtime owns every LLM/VLM/TTS forward pass, Metal configuration
    /// calls affect EdgeEngine native executors, not any archived compatibility path.
    public let legacyFallbackUsesNativeMetalScheduling: Bool
    public let gates: [NativeRuntimeReadinessGate]
    public let deferredCapabilities: [NativeRuntimeDeferredCapability]

    public init(
        runtimeVersion: String,
        nativeMetalSchedulingAppliesToNativeRuntime: Bool,
        legacyFallbackUsesNativeMetalScheduling: Bool,
        gates: [NativeRuntimeReadinessGate],
        deferredCapabilities: [NativeRuntimeDeferredCapability] = []
    ) {
        self.runtimeVersion = runtimeVersion
        self.nativeMetalSchedulingAppliesToNativeRuntime = nativeMetalSchedulingAppliesToNativeRuntime
        self.legacyFallbackUsesNativeMetalScheduling = legacyFallbackUsesNativeMetalScheduling
        self.gates = gates
        self.deferredCapabilities = deferredCapabilities
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        runtimeVersion = try container.decode(String.self, forKey: .runtimeVersion)
        nativeMetalSchedulingAppliesToNativeRuntime = try container.decode(
            Bool.self,
            forKey: .nativeMetalSchedulingAppliesToNativeRuntime
        )
        legacyFallbackUsesNativeMetalScheduling = try container.decode(
            Bool.self,
            forKey: .legacyFallbackUsesNativeMetalScheduling
        )
        gates = try container.decode([NativeRuntimeReadinessGate].self, forKey: .gates)
        deferredCapabilities = try container.decodeIfPresent(
            [NativeRuntimeDeferredCapability].self,
            forKey: .deferredCapabilities
        ) ?? []
    }

    public var isReleaseReady: Bool {
        gates.allSatisfy { $0.status == .passed }
    }

    public func gate(_ id: String) -> NativeRuntimeReadinessGate? {
        gates.first { $0.id == id }
    }

    public func deferredCapability(_ id: String) -> NativeRuntimeDeferredCapability? {
        deferredCapabilities.first { $0.id == id }
    }
}

public extension NativeRuntimeBridge {
    static func releaseReadinessReport(
        hasIPhoneAir9BT20DeviceGate: Bool = false,
        hasGRDBCrossLanguageHashGate: Bool = false,
        hasLLMVLMSmokeGate: Bool = false,
        hasNativeDSRKVRetentionGate: Bool = false,
        hasSTTDogfoodDecision: Bool = false
    ) -> NativeRuntimeReadinessReport {
        NativeRuntimeReadinessReport(
            runtimeVersion: runtimeVersion,
            nativeMetalSchedulingAppliesToNativeRuntime: true,
            legacyFallbackUsesNativeMetalScheduling: false,
            gates: [
                NativeRuntimeReadinessGate(
                    id: "iphone-air-9b-t20-device-gate",
                    status: hasIPhoneAir9BT20DeviceGate ? .passed : .blocked,
                    reason: hasIPhoneAir9BT20DeviceGate
                        ? "Native EdgeEngine path has been measured against the fork-era iPhone Air 9B T20 baseline."
                        : "Native Metal scheduling is enforced by EdgeEngine executors; the release path must prove no TPS, OOM, or jetsam regression on device.",
                    requiredEvidence: "Run tests/device_test/run_device_test.sh llm on iPhone Air with Qwen3.5-9B T20; require >=95% baseline TPS and zero OOM/jetsam failures."
                ),
                NativeRuntimeReadinessGate(
                    id: "grdb-cross-language-hash-gate",
                    status: hasGRDBCrossLanguageHashGate ? .passed : .blocked,
                    reason: hasGRDBCrossLanguageHashGate
                        ? "Swift and Python fact/manifest hashing have been verified byte-identical after the GRDB update."
                        : "EdgeRuntime data unit tests do not prove Swift/Python byte-equivalent hashes for persisted fact records.",
                    requiredEvidence: "Compute the same fact insert and manifest hash through EdgeScaffolding and EdgeRuntime; SHA256 outputs must match exactly."
                ),
                NativeRuntimeReadinessGate(
                    id: "llm-vlm-native-smoke-gate",
                    status: hasLLMVLMSmokeGate ? .passed : .blocked,
                    reason: hasLLMVLMSmokeGate
                        ? "LLM and VLM native paths have a current device smoke result."
                        : "The native EdgeEngine LLM/VLM path needs a current device smoke after removing legacy Swift inference dependencies.",
                    requiredEvidence: "Run one real LLM generation and one real VLM generation on device; compare output shape, failure mode, and memory behavior to the fork baseline."
                ),
                NativeRuntimeReadinessGate(
                    id: "native-dsr-kv-retention-gate",
                    status: hasNativeDSRKVRetentionGate ? .passed : .blocked,
                    reason: hasNativeDSRKVRetentionGate
                        ? "LLM/VLM forward paths use EdgeEngine native cache policy, so DSR/KV bounding is actually applied instead of only recorded in EdgeRuntime policy metadata."
                        : "Native Qwen forward construction must consume EdgeStudio DSR cache controls instead of only recording policy metadata.",
                    requiredEvidence: "Run device LLM/VLM with native EdgeEngine Qwen cache policy enabled; results must show bounded KV/DSR behavior, no fake DSR-vs-Vanilla labels, and no recall regression versus fork-era T20 gate."
                ),
                NativeRuntimeReadinessGate(
                    id: "stt-dogfood-decision-gate",
                    status: hasSTTDogfoodDecision ? .passed : .pending,
                    reason: hasSTTDogfoodDecision
                        ? "Dogfood STT policy is explicitly recorded for this release line."
                        : "Default public builds still need native ASR evidence or an explicit dogfood policy before STT is advertised.",
                    requiredEvidence: "Record whether dogfood enables native ASR or ships without STT."
                )
            ],
            deferredCapabilities: [
                NativeRuntimeDeferredCapability(
                    id: "dsr-kv-cache-bounding",
                    legacyFallbackBehavior: "Fails closed: archived compatibility code has KV size knobs but no EdgeStudio DSR cache path, so DSR labels in tests do not prove bounded-cache retention.",
                    nativeRuntimeMigration: "EdgeEngine must own Qwen LLM/VLM forward passes and apply native DSR/KV cache policy."
                ),
                NativeRuntimeDeferredCapability(
                    id: "hidden-state-capture",
                    legacyFallbackBehavior: "Fails closed because archived compatibility code has no EdgeStudio callWithCapture hook.",
                    nativeRuntimeMigration: "EdgeEngine must own Qwen forward passes and expose requested hidden states."
                ),
                NativeRuntimeDeferredCapability(
                    id: "activation-steering-injection",
                    legacyFallbackBehavior: "Vector loading and projection remain available; stackResiduals injection is not applied on the archived compatibility path.",
                    nativeRuntimeMigration: "EdgeEngine must expose a native residual injection point for target FA/GDN layers."
                ),
                NativeRuntimeDeferredCapability(
                    id: "dynamic-gdn-offload",
                    legacyFallbackBehavior: "Fails closed and logs skip because archived compatibility code cannot offload GDN layer state.",
                    nativeRuntimeMigration: "EdgeEngine must own GDN state storage and layer execution scheduling."
                ),
                NativeRuntimeDeferredCapability(
                    id: "low-rank-gdn-state",
                    legacyFallbackBehavior: "Detected Qwen3.5 models do not receive EdgeStudio low-rank GDN state compression on the archived compatibility path.",
                    nativeRuntimeMigration: "EdgeEngine must implement native GDN recurrent state compression."
                )
            ]
        )
    }
}
