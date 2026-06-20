// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

public enum HaloCapsuleAcceptPolicy {
    public static let defaultPolicySchemaVersion = "edgestudio.halo_capsule_accept_policy.v1"

    public struct Snapshot: Codable, Equatable, Sendable {
        public let schemaVersion: String
        public let baseModelID: String
        public let modelDisplayName: String
        public let currentRuntimeVersion: String
        public let supportedMessageSchemaVersion: String
        public let supportedMessageKind: String
        public let toolSchemaSHA256: String
        public let registeredToolCount: Int
        public let defaultEnableThinking: Bool

        public init(
            schemaVersion: String = HaloCapsuleAcceptPolicy.defaultPolicySchemaVersion,
            baseModelID: String,
            modelDisplayName: String,
            currentRuntimeVersion: String = EdgeKitRuntime.version,
            supportedMessageSchemaVersion: String = HaloCapsuleMeshMessage.supportedSchemaVersion,
            supportedMessageKind: String = HaloCapsuleMeshMessage.offerKind,
            toolSchemaSHA256: String,
            registeredToolCount: Int,
            defaultEnableThinking: Bool
        ) {
            self.schemaVersion = schemaVersion
            self.baseModelID = baseModelID
            self.modelDisplayName = modelDisplayName
            self.currentRuntimeVersion = currentRuntimeVersion
            self.supportedMessageSchemaVersion = supportedMessageSchemaVersion
            self.supportedMessageKind = supportedMessageKind
            self.toolSchemaSHA256 = toolSchemaSHA256
            self.registeredToolCount = registeredToolCount
            self.defaultEnableThinking = defaultEnableThinking
        }

        public var jsonString: String? {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            guard let data = try? encoder.encode(self) else { return nil }
            return String(data: data, encoding: .utf8)
        }

        private enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case baseModelID = "base_model_id"
            case modelDisplayName = "model_display_name"
            case currentRuntimeVersion = "current_runtime_version"
            case supportedMessageSchemaVersion = "supported_message_schema_version"
            case supportedMessageKind = "supported_message_kind"
            case toolSchemaSHA256 = "tool_schema_sha256"
            case registeredToolCount = "registered_tool_count"
            case defaultEnableThinking = "default_enable_thinking"
        }
    }

    public struct OfferReceipt: Codable, Equatable, Sendable {
        public let transferID: String
        public let capsuleID: String
        public let artifactID: String
        public let artifactBytes: Int
        public let canonicalSHA256: String

        public init(
            transferID: String,
            capsuleID: String,
            artifactID: String,
            artifactBytes: Int,
            canonicalSHA256: String
        ) {
            self.transferID = transferID
            self.capsuleID = capsuleID
            self.artifactID = artifactID
            self.artifactBytes = artifactBytes
            self.canonicalSHA256 = canonicalSHA256
        }

        private enum CodingKeys: String, CodingKey {
            case transferID = "transfer_id"
            case capsuleID = "capsule_id"
            case artifactID = "artifact_id"
            case artifactBytes = "artifact_bytes"
            case canonicalSHA256 = "canonical_sha256"
        }
    }

    public static func snapshot(
        schemaVersion: String = defaultPolicySchemaVersion,
        baseModelID: String,
        modelDisplayName: String,
        currentRuntimeVersion: String = EdgeKitRuntime.version,
        toolSchemaSnapshot: ToolSchemaSnapshot,
        defaultEnableThinking: Bool
    ) -> Snapshot {
        Snapshot(
            schemaVersion: schemaVersion,
            baseModelID: baseModelID,
            modelDisplayName: modelDisplayName,
            currentRuntimeVersion: currentRuntimeVersion,
            toolSchemaSHA256: toolSchemaSnapshot.sha256,
            registeredToolCount: toolSchemaSnapshot.export.tools.count,
            defaultEnableThinking: defaultEnableThinking
        )
    }

    public static func validateOffer(
        data: Data,
        policy: Snapshot
    ) throws -> OfferReceipt {
        let message = try JSONDecoder().decode(HaloCapsuleMeshMessage.self, from: data)
        return try validateOffer(message, policy: policy)
    }

    public static func validateOffer(
        _ message: HaloCapsuleMeshMessage,
        policy: Snapshot
    ) throws -> OfferReceipt {
        try message.validate(
            expectedBaseModelID: policy.baseModelID,
            expectedToolSchemaSHA256: policy.toolSchemaSHA256,
            currentRuntimeVersion: policy.currentRuntimeVersion
        )
        return OfferReceipt(
            transferID: message.transferID,
            capsuleID: message.capsule.capsuleID,
            artifactID: message.capsule.artifact.artifactID,
            artifactBytes: message.capsule.artifact.totalBytes,
            canonicalSHA256: try message.canonicalSHA256()
        )
    }
}
