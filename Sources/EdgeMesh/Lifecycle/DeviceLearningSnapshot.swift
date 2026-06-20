// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// App-provided device learning state sent over EdgeMesh after pairing.
///
/// This is a status snapshot, not a data backfill channel. Payloads should
/// contain counters, fingerprints, compatibility verdicts, and timestamps; raw
/// user records continue to use explicit data/event sync flows.
public struct DeviceLearningSnapshot: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.device_learning_snapshot.v1"

    public let schemaVersion: String
    public let capturedAtUnixSeconds: Double
    public let identity: Identity
    public let device: Device
    public let model: ModelInventory
    public let data: DataSummary
    public let learning: LearningSummary
    public let corrections: CorrectionSummary
    public let eval: EvalSummary
    public let sync: SyncSummary?

    public init(
        schemaVersion: String = DeviceLearningSnapshot.schemaVersion,
        capturedAtUnixSeconds: Double = Date().timeIntervalSince1970,
        identity: Identity,
        device: Device,
        model: ModelInventory,
        data: DataSummary,
        learning: LearningSummary,
        corrections: CorrectionSummary = CorrectionSummary(),
        eval: EvalSummary = EvalSummary(),
        sync: SyncSummary? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.capturedAtUnixSeconds = capturedAtUnixSeconds
        self.identity = identity
        self.device = device
        self.model = model
        self.data = data
        self.learning = learning
        self.corrections = corrections
        self.eval = eval
        self.sync = sync
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case capturedAtUnixSeconds = "captured_at_unix_seconds"
        case identity
        case device
        case model
        case data
        case learning
        case corrections
        case eval
        case sync
    }
}

extension DeviceLearningSnapshot {
    public struct Identity: Codable, Equatable, Sendable {
        public let peerID: String?
        public let displayName: String?
        public let appID: String?
        public let bundleIdentifier: String?
        public let bundleVersion: String?
        public let buildNumber: String?
        public let gitCommit: String?
        public let edgeKitVersion: String?
        public let edgeHaloVersion: String?
        public let edgeEngineVersion: String?

        public init(
            peerID: String? = nil,
            displayName: String? = nil,
            appID: String? = nil,
            bundleIdentifier: String? = nil,
            bundleVersion: String? = nil,
            buildNumber: String? = nil,
            gitCommit: String? = nil,
            edgeKitVersion: String? = nil,
            edgeHaloVersion: String? = nil,
            edgeEngineVersion: String? = nil
        ) {
            self.peerID = peerID
            self.displayName = displayName
            self.appID = appID
            self.bundleIdentifier = bundleIdentifier
            self.bundleVersion = bundleVersion
            self.buildNumber = buildNumber
            self.gitCommit = gitCommit
            self.edgeKitVersion = edgeKitVersion
            self.edgeHaloVersion = edgeHaloVersion
            self.edgeEngineVersion = edgeEngineVersion
        }

        private enum CodingKeys: String, CodingKey {
            case peerID = "peer_id"
            case displayName = "display_name"
            case appID = "app_id"
            case bundleIdentifier = "bundle_identifier"
            case bundleVersion = "bundle_version"
            case buildNumber = "build_number"
            case gitCommit = "git_commit"
            case edgeKitVersion = "edge_kit_version"
            case edgeHaloVersion = "edge_halo_version"
            case edgeEngineVersion = "edge_engine_version"
        }
    }

    public struct Device: Codable, Equatable, Sendable {
        public let platform: String?
        public let modelIdentifier: String?
        public let osVersion: String?
        public let physicalMemoryBytes: Int64?
        public let availableMemoryBytes: Int64?
        public let memoryLimitBytes: Int64?
        public let thermalState: String?

        public init(
            platform: String? = nil,
            modelIdentifier: String? = nil,
            osVersion: String? = nil,
            physicalMemoryBytes: Int64? = nil,
            availableMemoryBytes: Int64? = nil,
            memoryLimitBytes: Int64? = nil,
            thermalState: String? = nil
        ) {
            self.platform = platform
            self.modelIdentifier = modelIdentifier
            self.osVersion = osVersion
            self.physicalMemoryBytes = physicalMemoryBytes
            self.availableMemoryBytes = availableMemoryBytes
            self.memoryLimitBytes = memoryLimitBytes
            self.thermalState = thermalState
        }

        private enum CodingKeys: String, CodingKey {
            case platform
            case modelIdentifier = "model_identifier"
            case osVersion = "os_version"
            case physicalMemoryBytes = "physical_memory_bytes"
            case availableMemoryBytes = "available_memory_bytes"
            case memoryLimitBytes = "memory_limit_bytes"
            case thermalState = "thermal_state"
        }
    }

    public struct ModelInventory: Codable, Equatable, Sendable {
        public let selectedModelID: String?
        public let loadedModelID: String?
        public let loadState: String?
        public let loadError: String?
        public let installedModels: [InstalledModel]

        public init(
            selectedModelID: String? = nil,
            loadedModelID: String? = nil,
            loadState: String? = nil,
            loadError: String? = nil,
            installedModels: [InstalledModel] = []
        ) {
            self.selectedModelID = selectedModelID
            self.loadedModelID = loadedModelID
            self.loadState = loadState
            self.loadError = loadError
            self.installedModels = installedModels
        }

        private enum CodingKeys: String, CodingKey {
            case selectedModelID = "selected_model_id"
            case loadedModelID = "loaded_model_id"
            case loadState = "load_state"
            case loadError = "load_error"
            case installedModels = "installed_models"
        }
    }

    public struct InstalledModel: Codable, Equatable, Sendable {
        public let modelID: String
        public let displayName: String?
        public let family: String?
        public let quantization: String?
        public let pathHint: String?
        public let configSHA256: String?
        public let weightsSHA256: String?
        public let sizeBytes: Int64?
        public let isSelected: Bool?

        public init(
            modelID: String,
            displayName: String? = nil,
            family: String? = nil,
            quantization: String? = nil,
            pathHint: String? = nil,
            configSHA256: String? = nil,
            weightsSHA256: String? = nil,
            sizeBytes: Int64? = nil,
            isSelected: Bool? = nil
        ) {
            self.modelID = modelID
            self.displayName = displayName
            self.family = family
            self.quantization = quantization
            self.pathHint = pathHint
            self.configSHA256 = configSHA256
            self.weightsSHA256 = weightsSHA256
            self.sizeBytes = sizeBytes
            self.isSelected = isSelected
        }

        private enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case displayName = "display_name"
            case family
            case quantization
            case pathHint = "path_hint"
            case configSHA256 = "config_sha256"
            case weightsSHA256 = "weights_sha256"
            case sizeBytes = "size_bytes"
            case isSelected = "is_selected"
        }
    }

    public struct DataSummary: Codable, Equatable, Sendable {
        public let eventStoreTotal: Int?
        public let factsTotal: Int?
        public let factsClassified: Int?
        public let factsRawUnclassified: Int?
        public let lastImportAtUnixSeconds: Double?
        public let lastClassifiedAtUnixSeconds: Double?
        public let readiness: String?

        public init(
            eventStoreTotal: Int? = nil,
            factsTotal: Int? = nil,
            factsClassified: Int? = nil,
            factsRawUnclassified: Int? = nil,
            lastImportAtUnixSeconds: Double? = nil,
            lastClassifiedAtUnixSeconds: Double? = nil,
            readiness: String? = nil
        ) {
            self.eventStoreTotal = eventStoreTotal
            self.factsTotal = factsTotal
            self.factsClassified = factsClassified
            self.factsRawUnclassified = factsRawUnclassified
            self.lastImportAtUnixSeconds = lastImportAtUnixSeconds
            self.lastClassifiedAtUnixSeconds = lastClassifiedAtUnixSeconds
            self.readiness = readiness
        }

        private enum CodingKeys: String, CodingKey {
            case eventStoreTotal = "event_store_total"
            case factsTotal = "facts_total"
            case factsClassified = "facts_classified"
            case factsRawUnclassified = "facts_raw_unclassified"
            case lastImportAtUnixSeconds = "last_import_at_unix_seconds"
            case lastClassifiedAtUnixSeconds = "last_classified_at_unix_seconds"
            case readiness
        }
    }

    public struct LearningSummary: Codable, Equatable, Sendable {
        public let toolsOnly: Artifact
        public let rpp: Artifact
        public let neuralImprint: Artifact
        public let activeArtifactKind: String?
        public let targetLayer: Int?
        public let aLibraryID: String?
        public let aLibrarySHA256: String?
        public let toolSchemaSHA256: String?

        public init(
            toolsOnly: Artifact = Artifact(status: .unknown),
            rpp: Artifact = Artifact(status: .unknown),
            neuralImprint: Artifact = Artifact(status: .unknown),
            activeArtifactKind: String? = nil,
            targetLayer: Int? = nil,
            aLibraryID: String? = nil,
            aLibrarySHA256: String? = nil,
            toolSchemaSHA256: String? = nil
        ) {
            self.toolsOnly = toolsOnly
            self.rpp = rpp
            self.neuralImprint = neuralImprint
            self.activeArtifactKind = activeArtifactKind
            self.targetLayer = targetLayer
            self.aLibraryID = aLibraryID
            self.aLibrarySHA256 = aLibrarySHA256
            self.toolSchemaSHA256 = toolSchemaSHA256
        }

        private enum CodingKeys: String, CodingKey {
            case toolsOnly = "tools_only"
            case rpp
            case neuralImprint = "neural_imprint"
            case legacyPersonaKV = "persona_kv"
            case activeArtifactKind = "active_artifact_kind"
            case targetLayer = "target_layer"
            case aLibraryID = "a_library_id"
            case aLibrarySHA256 = "a_library_sha256"
            case toolSchemaSHA256 = "tool_schema_sha256"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.toolsOnly = try container.decodeIfPresent(Artifact.self, forKey: .toolsOnly)
                ?? Artifact(status: .unknown)
            self.rpp = try container.decodeIfPresent(Artifact.self, forKey: .rpp)
                ?? Artifact(status: .unknown)
            let decodedNeuralImprint = try container.decodeIfPresent(Artifact.self, forKey: .neuralImprint)
            let decodedLegacyPersonaKV = try container.decodeIfPresent(Artifact.self, forKey: .legacyPersonaKV)
            self.neuralImprint = decodedNeuralImprint ?? decodedLegacyPersonaKV ?? Artifact(status: .unknown)
            self.activeArtifactKind = try container.decodeIfPresent(String.self, forKey: .activeArtifactKind)
            self.targetLayer = try container.decodeIfPresent(Int.self, forKey: .targetLayer)
            self.aLibraryID = try container.decodeIfPresent(String.self, forKey: .aLibraryID)
            self.aLibrarySHA256 = try container.decodeIfPresent(String.self, forKey: .aLibrarySHA256)
            self.toolSchemaSHA256 = try container.decodeIfPresent(String.self, forKey: .toolSchemaSHA256)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(toolsOnly, forKey: .toolsOnly)
            try container.encode(rpp, forKey: .rpp)
            try container.encode(neuralImprint, forKey: .neuralImprint)
            try container.encodeIfPresent(activeArtifactKind, forKey: .activeArtifactKind)
            try container.encodeIfPresent(targetLayer, forKey: .targetLayer)
            try container.encodeIfPresent(aLibraryID, forKey: .aLibraryID)
            try container.encodeIfPresent(aLibrarySHA256, forKey: .aLibrarySHA256)
            try container.encodeIfPresent(toolSchemaSHA256, forKey: .toolSchemaSHA256)
        }
    }

    public enum ArtifactStatus: String, Codable, Equatable, Sendable {
        case unknown
        case missing
        case presentInactive = "present_inactive"
        case active
        case incompatible
        case stale
        case error
    }

    public struct Artifact: Codable, Equatable, Sendable {
        public let status: ArtifactStatus
        public let prefixTokenCount: Int?
        public let artifactSHA256: String?
        public let runID: String?
        public let createdAtUnixSeconds: Double?
        public let errorCode: String?
        public let errorMessage: String?

        public init(
            status: ArtifactStatus,
            prefixTokenCount: Int? = nil,
            artifactSHA256: String? = nil,
            runID: String? = nil,
            createdAtUnixSeconds: Double? = nil,
            errorCode: String? = nil,
            errorMessage: String? = nil
        ) {
            self.status = status
            self.prefixTokenCount = prefixTokenCount
            self.artifactSHA256 = artifactSHA256
            self.runID = runID
            self.createdAtUnixSeconds = createdAtUnixSeconds
            self.errorCode = errorCode
            self.errorMessage = errorMessage
        }

        private enum CodingKeys: String, CodingKey {
            case status
            case prefixTokenCount = "prefix_token_count"
            case artifactSHA256 = "artifact_sha256"
            case runID = "run_id"
            case createdAtUnixSeconds = "created_at_unix_seconds"
            case errorCode = "error_code"
            case errorMessage = "error_message"
        }
    }

    public struct CorrectionSummary: Codable, Equatable, Sendable {
        public let totalCount: Int?
        public let pendingCount: Int?
        public let needsRegen: Bool?
        public let countsByType: [String: Int]

        public init(
            totalCount: Int? = nil,
            pendingCount: Int? = nil,
            needsRegen: Bool? = nil,
            countsByType: [String: Int] = [:]
        ) {
            self.totalCount = totalCount
            self.pendingCount = pendingCount
            self.needsRegen = needsRegen
            self.countsByType = countsByType
        }

        private enum CodingKeys: String, CodingKey {
            case totalCount = "total_count"
            case pendingCount = "pending_count"
            case needsRegen = "needs_regen"
            case countsByType = "counts_by_type"
        }
    }

    public struct EvalSummary: Codable, Equatable, Sendable {
        public let latestStatus: String?
        public let latestRunID: String?
        public let latestScore: Double?
        public let latestLogSHA256: String?

        public init(
            latestStatus: String? = nil,
            latestRunID: String? = nil,
            latestScore: Double? = nil,
            latestLogSHA256: String? = nil
        ) {
            self.latestStatus = latestStatus
            self.latestRunID = latestRunID
            self.latestScore = latestScore
            self.latestLogSHA256 = latestLogSHA256
        }

        private enum CodingKeys: String, CodingKey {
            case latestStatus = "latest_status"
            case latestRunID = "latest_run_id"
            case latestScore = "latest_score"
            case latestLogSHA256 = "latest_log_sha256"
        }
    }

    public struct SyncSummary: Codable, Equatable, Sendable {
        public let lastSnapshotAtUnixSeconds: Double?
        public let lastEventUploadAtUnixSeconds: Double?
        public let lastRPPArtifactUploadAtUnixSeconds: Double?
        public let lastCapsuleReceivedAtUnixSeconds: Double?
        public let lastCapsuleAppliedAtUnixSeconds: Double?
        public let pendingTransferIDs: [String]

        public init(
            lastSnapshotAtUnixSeconds: Double? = nil,
            lastEventUploadAtUnixSeconds: Double? = nil,
            lastRPPArtifactUploadAtUnixSeconds: Double? = nil,
            lastCapsuleReceivedAtUnixSeconds: Double? = nil,
            lastCapsuleAppliedAtUnixSeconds: Double? = nil,
            pendingTransferIDs: [String] = []
        ) {
            self.lastSnapshotAtUnixSeconds = lastSnapshotAtUnixSeconds
            self.lastEventUploadAtUnixSeconds = lastEventUploadAtUnixSeconds
            self.lastRPPArtifactUploadAtUnixSeconds = lastRPPArtifactUploadAtUnixSeconds
            self.lastCapsuleReceivedAtUnixSeconds = lastCapsuleReceivedAtUnixSeconds
            self.lastCapsuleAppliedAtUnixSeconds = lastCapsuleAppliedAtUnixSeconds
            self.pendingTransferIDs = pendingTransferIDs
        }

        private enum CodingKeys: String, CodingKey {
            case lastSnapshotAtUnixSeconds = "last_snapshot_at_unix_seconds"
            case lastEventUploadAtUnixSeconds = "last_event_upload_at_unix_seconds"
            case lastRPPArtifactUploadAtUnixSeconds = "last_rpp_artifact_upload_at_unix_seconds"
            case lastCapsuleReceivedAtUnixSeconds = "last_capsule_received_at_unix_seconds"
            case lastCapsuleAppliedAtUnixSeconds = "last_capsule_applied_at_unix_seconds"
            case pendingTransferIDs = "pending_transfer_ids"
        }
    }
}

public protocol LearningStatusProvider {
    func makeDeviceLearningSnapshot() async throws -> DeviceLearningSnapshot
}

public struct DeviceStateSnapshotMessage: Codable, Equatable, Sendable {
    public static let operation = "device_state_snapshot"

    public let op: String
    public let payload: DeviceLearningSnapshot

    public init(payload: DeviceLearningSnapshot) {
        self.op = Self.operation
        self.payload = payload
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendDeviceStateSnapshot(_ snapshot: DeviceLearningSnapshot) throws {
        try sendJSON(DeviceStateSnapshotMessage(payload: snapshot))
    }
}
