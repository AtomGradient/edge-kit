// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

#if canImport(Darwin)
import Darwin
#endif

public enum DeviceLearningSnapshotBuilder {
    public struct ModelState: Sendable, Equatable {
        public let isLoaded: Bool
        public let loadedModelID: String?
        public let loadError: String?
        public let activeNeuralImprintPrefixTokenCount: Int?
        public let activeNeuralImprintArtifactSHA256: String?

        public init(
            isLoaded: Bool,
            loadedModelID: String? = nil,
            loadError: String? = nil,
            activeNeuralImprintPrefixTokenCount: Int? = nil,
            activeNeuralImprintArtifactSHA256: String? = nil
        ) {
            self.isLoaded = isLoaded
            self.loadedModelID = loadedModelID
            self.loadError = loadError
            self.activeNeuralImprintPrefixTokenCount = activeNeuralImprintPrefixTokenCount
            self.activeNeuralImprintArtifactSHA256 = activeNeuralImprintArtifactSHA256
        }

        public var loadState: String {
            if isLoaded { return "loaded" }
            if loadError != nil { return "error" }
            return "not_loaded"
        }

        public var hasActiveNeuralImprintCache: Bool {
            activeNeuralImprintPrefixTokenCount != nil || activeNeuralImprintArtifactSHA256 != nil
        }
    }

    public struct DataCounts: Sendable, Equatable {
        public let eventStoreTotal: Int?
        public let factsTotal: Int?
        public let factsClassified: Int?
        public let factsRawUnclassified: Int?

        public init(
            eventStoreTotal: Int? = nil,
            factsTotal: Int? = nil,
            factsClassified: Int? = nil,
            factsRawUnclassified: Int? = nil
        ) {
            self.eventStoreTotal = eventStoreTotal
            self.factsTotal = factsTotal
            self.factsClassified = factsClassified
            self.factsRawUnclassified = factsRawUnclassified
        }

        public var readiness: String {
            max(factsClassified ?? 0, eventStoreTotal ?? 0) >= 10 ? "enough" : "insufficient"
        }
    }

    public struct RPPState: Sendable, Equatable {
        public let runID: String?
        public let targetLayer: Int?
        public let aLibraryID: String?
        public let aLibrarySHA256: String?

        public init(
            runID: String? = nil,
            targetLayer: Int? = nil,
            aLibraryID: String? = nil,
            aLibrarySHA256: String? = nil
        ) {
            self.runID = runID
            self.targetLayer = targetLayer
            self.aLibraryID = aLibraryID
            self.aLibrarySHA256 = aLibrarySHA256
        }

        public var isRPPBacked: Bool {
            runID != nil
        }
    }

    public struct AppIdentity: Sendable, Equatable {
        static let embeddedBuildCommitInfoKey = "EdgeBuildCommit"

        public let peerID: String
        public let displayName: String
        public let appID: String?
        public let bundleIdentifier: String?
        public let bundleVersion: String?
        public let buildNumber: String?
        public let gitCommit: String?
        public let edgeHaloVersion: String?

        public init(
            peerID: String,
            displayName: String,
            appID: String? = Bundle.main.bundleIdentifier,
            bundleIdentifier: String? = Bundle.main.bundleIdentifier,
            bundleVersion: String? = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String,
            buildNumber: String? = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String,
            gitCommit: String? = nil,
            edgeHaloVersion: String? = nil
        ) {
            self.peerID = peerID
            self.displayName = displayName
            self.appID = appID
            self.bundleIdentifier = bundleIdentifier
            self.bundleVersion = bundleVersion
            self.buildNumber = buildNumber
            self.gitCommit = Self.normalizedBuildCommit(gitCommit) ?? Self.embeddedBuildGitCommit()
            self.edgeHaloVersion = edgeHaloVersion
        }

        static func embeddedBuildGitCommit(
            from infoDictionary: [String: Any]? = Bundle.main.infoDictionary
        ) -> String? {
            normalizedBuildCommit(infoDictionary?[embeddedBuildCommitInfoKey] as? String)
        }

        private static func normalizedBuildCommit(_ value: String?) -> String? {
            guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !trimmed.isEmpty,
                  trimmed.lowercased() != "unknown"
            else {
                return nil
            }
            return trimmed
        }
    }

    public struct ModelConfig: Sendable, Equatable {
        public let selectedModelID: String
        public let displayName: String?
        public let family: String?
        public let quantization: String?
        public let documentsDirectory: URL?

        public init(
            selectedModelID: String,
            displayName: String? = nil,
            family: String? = nil,
            quantization: String? = nil,
            documentsDirectory: URL? = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        ) {
            self.selectedModelID = selectedModelID
            self.displayName = displayName
            self.family = family
            self.quantization = quantization
            self.documentsDirectory = documentsDirectory
        }
    }

    public static func makeSnapshot(
        identity: AppIdentity,
        modelConfig: ModelConfig,
        modelState: ModelState,
        dataCounts: DataCounts,
        rppState: RPPState,
        neuralImprintDirectoryExists: Bool,
        toolSchemaSHA256: String?,
        capturedAtUnixSeconds: Double = Date().timeIntervalSince1970
    ) -> DeviceLearningSnapshot {
        DeviceLearningSnapshot(
            capturedAtUnixSeconds: capturedAtUnixSeconds,
            identity: .init(
                peerID: identity.peerID,
                displayName: identity.displayName,
                appID: identity.appID,
                bundleIdentifier: identity.bundleIdentifier,
                bundleVersion: identity.bundleVersion,
                buildNumber: identity.buildNumber,
                gitCommit: identity.gitCommit,
                edgeKitVersion: EdgeKitRuntime.version,
                edgeHaloVersion: identity.edgeHaloVersion,
                edgeEngineVersion: EdgeKitRuntime.nativeRuntimeVersion
            ),
            device: .init(
                platform: platformName,
                modelIdentifier: deviceModelIdentifier,
                osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                physicalMemoryBytes: Int64(ProcessInfo.processInfo.physicalMemory),
                thermalState: thermalState
            ),
            model: .init(
                selectedModelID: modelConfig.selectedModelID,
                loadedModelID: modelState.isLoaded ? (modelState.loadedModelID ?? modelConfig.selectedModelID) : nil,
                loadState: modelState.loadState,
                loadError: modelState.loadError,
                installedModels: installedModels(modelConfig: modelConfig)
            ),
            data: .init(
                eventStoreTotal: dataCounts.eventStoreTotal,
                factsTotal: dataCounts.factsTotal,
                factsClassified: dataCounts.factsClassified,
                factsRawUnclassified: dataCounts.factsRawUnclassified,
                readiness: dataCounts.readiness
            ),
            learning: .init(
                toolsOnly: toolsArtifact(modelState: modelState, rppState: rppState),
                rpp: rppArtifact(rppState: rppState),
                neuralImprint: neuralImprintArtifact(
                    modelState: modelState,
                    rppState: rppState,
                    neuralImprintDirectoryExists: neuralImprintDirectoryExists
                ),
                activeArtifactKind: activeArtifactKind(modelState: modelState, rppState: rppState),
                targetLayer: rppState.targetLayer,
                aLibraryID: rppState.aLibraryID,
                aLibrarySHA256: rppState.aLibrarySHA256,
                toolSchemaSHA256: toolSchemaSHA256
            )
        )
    }

    public static func readJSONFile(named name: String, documentsDirectory: URL? = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first) -> [String: Any]? {
        guard let documentsDirectory else { return nil }
        let url = documentsDirectory.appendingPathComponent(name)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    public static func quantizationLabel(for modelID: String) -> String? {
        let lower = modelID.lowercased()
        if lower.contains("4bit") { return "4bit" }
        if lower.contains("6bit") { return "6bit" }
        if lower.contains("8bit") { return "8bit" }
        return nil
    }

    private static func installedModels(modelConfig: ModelConfig) -> [DeviceLearningSnapshot.InstalledModel] {
        guard let documentsDirectory = modelConfig.documentsDirectory else { return [] }
        let modelDirectory = documentsDirectory.appendingPathComponent(modelConfig.selectedModelID, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelDirectory.path) else { return [] }
        return [
            .init(
                modelID: modelConfig.selectedModelID,
                displayName: modelConfig.displayName,
                family: modelConfig.family,
                quantization: modelConfig.quantization ?? quantizationLabel(for: modelConfig.selectedModelID),
                pathHint: "Documents/\(modelConfig.selectedModelID)",
                isSelected: true
            )
        ]
    }

    private static func neuralImprintArtifact(
        modelState: ModelState,
        rppState: RPPState,
        neuralImprintDirectoryExists: Bool
    ) -> DeviceLearningSnapshot.Artifact {
        if modelState.hasActiveNeuralImprintCache, rppState.isRPPBacked {
            return DeviceLearningSnapshot.Artifact(
                status: .active,
                prefixTokenCount: modelState.activeNeuralImprintPrefixTokenCount,
                artifactSHA256: modelState.activeNeuralImprintArtifactSHA256
            )
        }
        return DeviceLearningSnapshot.Artifact(
            status: neuralImprintDirectoryExists ? .presentInactive : .missing
        )
    }

    private static func rppArtifact(rppState: RPPState) -> DeviceLearningSnapshot.Artifact {
        DeviceLearningSnapshot.Artifact(
            status: rppState.runID == nil ? .missing : .active,
            runID: rppState.runID
        )
    }

    private static func toolsArtifact(
        modelState: ModelState,
        rppState: RPPState
    ) -> DeviceLearningSnapshot.Artifact {
        DeviceLearningSnapshot.Artifact(
            status: modelState.hasActiveNeuralImprintCache && !rppState.isRPPBacked ? .active : .unknown,
            prefixTokenCount: modelState.activeNeuralImprintPrefixTokenCount
        )
    }

    private static func activeArtifactKind(
        modelState: ModelState,
        rppState: RPPState
    ) -> String? {
        guard modelState.hasActiveNeuralImprintCache else { return nil }
        return rppState.isRPPBacked ? "combined_kv" : "tools_only_kv"
    }

    private static var platformName: String {
        #if os(iOS)
        "iOS"
        #elseif os(macOS)
        "macOS"
        #else
        ProcessInfo.processInfo.operatingSystemVersionString
        #endif
    }

    private static var deviceModelIdentifier: String? {
        #if canImport(Darwin)
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        #else
        return nil
        #endif
    }

    private static var thermalState: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
