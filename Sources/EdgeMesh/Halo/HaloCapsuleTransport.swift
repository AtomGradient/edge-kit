// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public enum HaloCapsuleTransportError: Error, Equatable, Sendable {
    case unsupportedSchemaVersion(String)
    case unsupportedMessageKind(String)
    case emptyTransferID
    case emptyCapsuleID
    case emptyBaseModelID
    case emptyArtifactID
    case emptyArtifactFiles
    case emptyCacheSnapshotID
    case emptyCacheTensors
    case emptyTensorName
    case emptyTensorDType
    case invalidArtifactByteCount(expected: Int, actual: Int)
    case invalidByteCount(field: String, value: Int)
    case invalidPrefixTokenCount(Int)
    case emptySHA256(field: String)
    case requirementsHashMismatch(expected: String, actual: String)
    case requirementsMismatch(key: String, expected: String, actual: String)
    case runtimeVersionUnsupported(minimum: String, actual: String)
    case unexpectedTransferFrame
    case unexpectedBinaryFrame
    case missingTransferOffer
    case transferIDMismatch(expected: String, actual: String)
    case unknownArtifactFile(String)
    case unsafeArtifactFileName(String)
    case missingDownloadURL(name: String)
    case downloadHTTPStatus(name: String, statusCode: Int)
    case artifactFileReadFailed(name: String, reason: String)
    case fileHashMismatch(name: String, expected: String, actual: String)
    case chunkHashMismatch(expected: String, actual: String)
    case invalidChunkOrder(name: String, expectedOffset: Int, actualOffset: Int)
    case incompleteChunk(name: String)
    case packageIncomplete(name: String, expected: Int, actual: Int)
}

public struct HaloCapsuleMeshMessage: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.halo_capsule_mesh_message.v1"
    public static let offerKind = "halo_capsule_offer"

    public var schemaVersion: String
    public var kind: String
    public var transferID: String
    public var capsule: HaloCapsuleDescriptor

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case transferID = "transfer_id"
        case capsule
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        kind: String = Self.offerKind,
        transferID: String,
        capsule: HaloCapsuleDescriptor
    ) {
        self.schemaVersion = schemaVersion
        self.kind = kind
        self.transferID = transferID
        self.capsule = capsule
    }

    public func validate(
        expectedBaseModelID: String? = nil,
        expectedModelFamily: String? = nil,
        expectedHiddenSize: Int? = nil,
        expectedLayerCount: Int? = nil,
        expectedToolSchemaSHA256: String? = nil,
        currentRuntimeVersion: String? = nil
    ) throws {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw HaloCapsuleTransportError.unsupportedSchemaVersion(schemaVersion)
        }
        guard kind == Self.offerKind else {
            throw HaloCapsuleTransportError.unsupportedMessageKind(kind)
        }
        guard !transferID.isEmpty else {
            throw HaloCapsuleTransportError.emptyTransferID
        }
        try capsule.validate(
            expectedBaseModelID: expectedBaseModelID,
            expectedModelFamily: expectedModelFamily,
            expectedHiddenSize: expectedHiddenSize,
            expectedLayerCount: expectedLayerCount,
            expectedToolSchemaSHA256: expectedToolSchemaSHA256,
            currentRuntimeVersion: currentRuntimeVersion
        )
    }

    public func canonicalData() throws -> Data {
        try HaloCapsuleCanonicalJSON.encode(normalizedForCanonicalEncoding())
    }

    public func canonicalSHA256() throws -> String {
        try HaloCapsuleCanonicalJSON.sha256Hex(canonicalData())
    }

    private func normalizedForCanonicalEncoding() -> HaloCapsuleMeshMessage {
        var copy = self
        copy.capsule = copy.capsule.normalizedForCanonicalEncoding()
        return copy
    }
}

public struct HaloCapsuleDescriptor: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.halo_capsule_descriptor.v1"

    public var schemaVersion: String
    public var capsuleID: String
    public var createdAt: Date
    public var baseModelID: String
    public var minRuntimeVersion: String
    public var requirementsSHA256: String
    public var requirements: HaloCapsuleRequirementsDescriptor
    public var cacheSnapshot: HaloCacheSnapshotDescriptor
    public var artifact: HaloCapsuleArtifactDescriptor

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case capsuleID = "capsule_id"
        case createdAt = "created_at"
        case baseModelID = "base_model_id"
        case minRuntimeVersion = "min_runtime_version"
        case requirementsSHA256 = "requirements_sha256"
        case requirements
        case cacheSnapshot = "cache_snapshot"
        case artifact
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        capsuleID: String,
        createdAt: Date,
        baseModelID: String,
        minRuntimeVersion: String,
        requirementsSHA256: String,
        requirements: HaloCapsuleRequirementsDescriptor,
        cacheSnapshot: HaloCacheSnapshotDescriptor,
        artifact: HaloCapsuleArtifactDescriptor
    ) {
        self.schemaVersion = schemaVersion
        self.capsuleID = capsuleID
        self.createdAt = createdAt
        self.baseModelID = baseModelID
        self.minRuntimeVersion = minRuntimeVersion
        self.requirementsSHA256 = requirementsSHA256
        self.requirements = requirements
        self.cacheSnapshot = cacheSnapshot
        self.artifact = artifact
    }

    public static func make(
        capsuleID: String,
        createdAt: Date,
        baseModelID: String,
        minRuntimeVersion: String,
        requirements: HaloCapsuleRequirementsDescriptor,
        cacheSnapshot: HaloCacheSnapshotDescriptor,
        artifact: HaloCapsuleArtifactDescriptor
    ) throws -> HaloCapsuleDescriptor {
        HaloCapsuleDescriptor(
            capsuleID: capsuleID,
            createdAt: createdAt,
            baseModelID: baseModelID,
            minRuntimeVersion: minRuntimeVersion,
            requirementsSHA256: try requirements.canonicalSHA256(),
            requirements: requirements,
            cacheSnapshot: cacheSnapshot,
            artifact: artifact
        )
    }

    public func validate(
        expectedBaseModelID: String? = nil,
        expectedModelFamily: String? = nil,
        expectedHiddenSize: Int? = nil,
        expectedLayerCount: Int? = nil,
        expectedToolSchemaSHA256: String? = nil,
        currentRuntimeVersion: String? = nil
    ) throws {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw HaloCapsuleTransportError.unsupportedSchemaVersion(schemaVersion)
        }
        guard !capsuleID.isEmpty else {
            throw HaloCapsuleTransportError.emptyCapsuleID
        }
        guard !baseModelID.isEmpty else {
            throw HaloCapsuleTransportError.emptyBaseModelID
        }
        try requirements.validate()
        try requireSHA256(requirementsSHA256, field: "requirements_sha256")
        let actualRequirementsSHA256 = try requirements.canonicalSHA256()
        guard requirementsSHA256 == actualRequirementsSHA256 else {
            throw HaloCapsuleTransportError.requirementsHashMismatch(
                expected: requirementsSHA256,
                actual: actualRequirementsSHA256
            )
        }
        if let expectedBaseModelID, expectedBaseModelID != baseModelID {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "base_model_id",
                expected: expectedBaseModelID,
                actual: baseModelID
            )
        }
        if let expectedModelFamily, expectedModelFamily != requirements.modelFamily {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "model_family",
                expected: expectedModelFamily,
                actual: requirements.modelFamily
            )
        }
        if let expectedHiddenSize, expectedHiddenSize != requirements.hiddenSize {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "hidden_size",
                expected: String(expectedHiddenSize),
                actual: String(requirements.hiddenSize)
            )
        }
        if let expectedLayerCount, expectedLayerCount != requirements.layerCount {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "layer_count",
                expected: String(expectedLayerCount),
                actual: String(requirements.layerCount)
            )
        }
        if let expectedToolSchemaSHA256,
           expectedToolSchemaSHA256 != requirements.toolSchemaSHA256 {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "tool_schema_sha256",
                expected: expectedToolSchemaSHA256,
                actual: requirements.toolSchemaSHA256
            )
        }
        if let currentRuntimeVersion,
           HaloCapsuleVersion.compare(currentRuntimeVersion, minRuntimeVersion) == .orderedAscending {
            throw HaloCapsuleTransportError.runtimeVersionUnsupported(
                minimum: minRuntimeVersion,
                actual: currentRuntimeVersion
            )
        }
        try cacheSnapshot.validate()
        try artifact.validate()
    }

    fileprivate func normalizedForCanonicalEncoding() -> HaloCapsuleDescriptor {
        var copy = self
        copy.cacheSnapshot = copy.cacheSnapshot.normalizedForCanonicalEncoding()
        copy.artifact = copy.artifact.normalizedForCanonicalEncoding()
        return copy
    }
}

public struct HaloCapsuleRequirementsDescriptor: Codable, Equatable, Sendable {
    public var modelFamily: String
    public var hiddenSize: Int
    public var layerCount: Int
    public var modelConfigSHA256: String
    public var modelWeightsSHA256: String
    public var tokenizerJSONSHA256: String
    public var tokenizerConfigSHA256: String
    public var chatTemplateSHA256: String
    public var systemPromptSHA256: String
    public var renderedPrefixSHA256: String
    public var prefixTokenCount: Int
    public var toolSchemaSHA256: String
    public var profileBodySHA256: String
    public var enableThinking: Bool
    public var cacheBackend: String
    public var cacheBackendVersion: String
    public var cacheTopologySHA256: String

    enum CodingKeys: String, CodingKey {
        case modelFamily = "model_family"
        case hiddenSize = "hidden_size"
        case layerCount = "layer_count"
        case modelConfigSHA256 = "model_config_sha256"
        case modelWeightsSHA256 = "model_weights_sha256"
        case tokenizerJSONSHA256 = "tokenizer_json_sha256"
        case tokenizerConfigSHA256 = "tokenizer_config_sha256"
        case chatTemplateSHA256 = "chat_template_sha256"
        case systemPromptSHA256 = "system_prompt_sha256"
        case renderedPrefixSHA256 = "rendered_prefix_sha256"
        case prefixTokenCount = "prefix_token_count"
        case toolSchemaSHA256 = "tool_schema_sha256"
        case profileBodySHA256 = "profile_body_sha256"
        case enableThinking = "enable_thinking"
        case cacheBackend = "cache_backend"
        case cacheBackendVersion = "cache_backend_version"
        case cacheTopologySHA256 = "cache_topology_sha256"
    }

    public init(
        modelConfigSHA256: String,
        modelWeightsSHA256: String,
        tokenizerJSONSHA256: String,
        tokenizerConfigSHA256: String,
        chatTemplateSHA256: String,
        systemPromptSHA256: String,
        renderedPrefixSHA256: String,
        prefixTokenCount: Int,
        toolSchemaSHA256: String,
        profileBodySHA256: String,
        enableThinking: Bool,
        cacheBackend: String,
        cacheBackendVersion: String,
        cacheTopologySHA256: String,
        modelFamily: String = "",
        hiddenSize: Int = 0,
        layerCount: Int = 0
    ) {
        self.modelFamily = modelFamily
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.modelConfigSHA256 = modelConfigSHA256
        self.modelWeightsSHA256 = modelWeightsSHA256
        self.tokenizerJSONSHA256 = tokenizerJSONSHA256
        self.tokenizerConfigSHA256 = tokenizerConfigSHA256
        self.chatTemplateSHA256 = chatTemplateSHA256
        self.systemPromptSHA256 = systemPromptSHA256
        self.renderedPrefixSHA256 = renderedPrefixSHA256
        self.prefixTokenCount = prefixTokenCount
        self.toolSchemaSHA256 = toolSchemaSHA256
        self.profileBodySHA256 = profileBodySHA256
        self.enableThinking = enableThinking
        self.cacheBackend = cacheBackend
        self.cacheBackendVersion = cacheBackendVersion
        self.cacheTopologySHA256 = cacheTopologySHA256
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelFamily = try container.decodeIfPresent(String.self, forKey: .modelFamily) ?? ""
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 0
        layerCount = try container.decodeIfPresent(Int.self, forKey: .layerCount) ?? 0
        modelConfigSHA256 = try container.decode(String.self, forKey: .modelConfigSHA256)
        modelWeightsSHA256 = try container.decode(String.self, forKey: .modelWeightsSHA256)
        tokenizerJSONSHA256 = try container.decode(String.self, forKey: .tokenizerJSONSHA256)
        tokenizerConfigSHA256 = try container.decode(String.self, forKey: .tokenizerConfigSHA256)
        chatTemplateSHA256 = try container.decode(String.self, forKey: .chatTemplateSHA256)
        systemPromptSHA256 = try container.decode(String.self, forKey: .systemPromptSHA256)
        renderedPrefixSHA256 = try container.decode(String.self, forKey: .renderedPrefixSHA256)
        prefixTokenCount = try container.decode(Int.self, forKey: .prefixTokenCount)
        toolSchemaSHA256 = try container.decode(String.self, forKey: .toolSchemaSHA256)
        profileBodySHA256 = try container.decode(String.self, forKey: .profileBodySHA256)
        enableThinking = try container.decode(Bool.self, forKey: .enableThinking)
        cacheBackend = try container.decode(String.self, forKey: .cacheBackend)
        cacheBackendVersion = try container.decode(String.self, forKey: .cacheBackendVersion)
        cacheTopologySHA256 = try container.decode(String.self, forKey: .cacheTopologySHA256)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelFamily, forKey: .modelFamily)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(layerCount, forKey: .layerCount)
        try container.encode(modelConfigSHA256, forKey: .modelConfigSHA256)
        try container.encode(modelWeightsSHA256, forKey: .modelWeightsSHA256)
        try container.encode(tokenizerJSONSHA256, forKey: .tokenizerJSONSHA256)
        try container.encode(tokenizerConfigSHA256, forKey: .tokenizerConfigSHA256)
        try container.encode(chatTemplateSHA256, forKey: .chatTemplateSHA256)
        try container.encode(systemPromptSHA256, forKey: .systemPromptSHA256)
        try container.encode(renderedPrefixSHA256, forKey: .renderedPrefixSHA256)
        try container.encode(prefixTokenCount, forKey: .prefixTokenCount)
        try container.encode(toolSchemaSHA256, forKey: .toolSchemaSHA256)
        try container.encode(profileBodySHA256, forKey: .profileBodySHA256)
        try container.encode(enableThinking, forKey: .enableThinking)
        try container.encode(cacheBackend, forKey: .cacheBackend)
        try container.encode(cacheBackendVersion, forKey: .cacheBackendVersion)
        try container.encode(cacheTopologySHA256, forKey: .cacheTopologySHA256)
    }

    public func canonicalData() throws -> Data {
        try HaloCapsuleCanonicalJSON.encode(self)
    }

    public func canonicalSHA256() throws -> String {
        try HaloCapsuleCanonicalJSON.sha256Hex(canonicalData())
    }

    public func validate() throws {
        guard !modelFamily.isEmpty else {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "model_family",
                expected: "non_empty",
                actual: modelFamily
            )
        }
        guard hiddenSize > 0 else {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "hidden_size",
                expected: ">0",
                actual: String(hiddenSize)
            )
        }
        guard layerCount > 0 else {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "layer_count",
                expected: ">0",
                actual: String(layerCount)
            )
        }
        try requireSHA256(modelConfigSHA256, field: "model_config_sha256")
        try requireSHA256(modelWeightsSHA256, field: "model_weights_sha256")
        try requireSHA256(tokenizerJSONSHA256, field: "tokenizer_json_sha256")
        try requireSHA256(tokenizerConfigSHA256, field: "tokenizer_config_sha256")
        try requireSHA256(chatTemplateSHA256, field: "chat_template_sha256")
        try requireSHA256(systemPromptSHA256, field: "system_prompt_sha256")
        try requireSHA256(renderedPrefixSHA256, field: "rendered_prefix_sha256")
        guard prefixTokenCount > 0 else {
            throw HaloCapsuleTransportError.invalidPrefixTokenCount(prefixTokenCount)
        }
        try requireSHA256(toolSchemaSHA256, field: "tool_schema_sha256")
        try requireSHA256(profileBodySHA256, field: "profile_body_sha256")
        try requireSHA256(cacheTopologySHA256, field: "cache_topology_sha256")
    }
}

public struct HaloCacheSnapshotDescriptor: Codable, Equatable, Sendable {
    public var snapshotID: String
    public var createdAt: Date
    public var tokenCount: Int
    public var tokenIDsSHA256: String
    public var cacheBackend: String
    public var cacheBackendVersion: String
    public var tensors: [HaloCacheTensorDescriptor]

    enum CodingKeys: String, CodingKey {
        case snapshotID = "snapshot_id"
        case createdAt = "created_at"
        case tokenCount = "token_count"
        case tokenIDsSHA256 = "token_ids_sha256"
        case cacheBackend = "cache_backend"
        case cacheBackendVersion = "cache_backend_version"
        case tensors
    }

    public init(
        snapshotID: String,
        createdAt: Date,
        tokenCount: Int,
        tokenIDsSHA256: String,
        cacheBackend: String,
        cacheBackendVersion: String,
        tensors: [HaloCacheTensorDescriptor]
    ) {
        self.snapshotID = snapshotID
        self.createdAt = createdAt
        self.tokenCount = tokenCount
        self.tokenIDsSHA256 = tokenIDsSHA256
        self.cacheBackend = cacheBackend
        self.cacheBackendVersion = cacheBackendVersion
        self.tensors = tensors
    }

    fileprivate func normalizedForCanonicalEncoding() -> HaloCacheSnapshotDescriptor {
        var copy = self
        copy.tensors.sort { $0.name < $1.name }
        return copy
    }

    public func validate() throws {
        guard !snapshotID.isEmpty else {
            throw HaloCapsuleTransportError.emptyCacheSnapshotID
        }
        try requireSHA256(tokenIDsSHA256, field: "cache_snapshot.token_ids_sha256")
        guard !cacheBackend.isEmpty else {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "cache_backend",
                expected: "non_empty",
                actual: cacheBackend
            )
        }
        guard !cacheBackendVersion.isEmpty else {
            throw HaloCapsuleTransportError.requirementsMismatch(
                key: "cache_backend_version",
                expected: "non_empty",
                actual: cacheBackendVersion
            )
        }
        guard !tensors.isEmpty else {
            throw HaloCapsuleTransportError.emptyCacheTensors
        }
        for tensor in tensors {
            try tensor.validate()
        }
    }
}

public struct HaloCacheTensorDescriptor: Codable, Equatable, Sendable {
    public var name: String
    public var shape: [Int]
    public var dtype: String
    public var byteCount: Int
    public var sha256: String?

    enum CodingKeys: String, CodingKey {
        case name
        case shape
        case dtype
        case byteCount = "byte_count"
        case sha256
    }

    public init(
        name: String,
        shape: [Int],
        dtype: String,
        byteCount: Int,
        sha256: String? = nil
    ) {
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.byteCount = byteCount
        self.sha256 = sha256
    }

    public func validate() throws {
        guard !name.isEmpty else {
            throw HaloCapsuleTransportError.emptyTensorName
        }
        guard !dtype.isEmpty else {
            throw HaloCapsuleTransportError.emptyTensorDType
        }
        guard byteCount >= 0 else {
            throw HaloCapsuleTransportError.invalidByteCount(
                field: "cache_snapshot.tensors[].byte_count",
                value: byteCount
            )
        }
        if let sha256 {
            try requireSHA256(sha256, field: "cache_snapshot.tensors[].sha256")
        }
    }
}

public struct HaloCapsuleArtifactDescriptor: Codable, Equatable, Sendable {
    public var artifactID: String
    public var mediaType: String
    public var totalBytes: Int
    public var sha256: String
    public var files: [HaloCapsuleArtifactFile]

    enum CodingKeys: String, CodingKey {
        case artifactID = "artifact_id"
        case mediaType = "media_type"
        case totalBytes = "total_bytes"
        case sha256
        case files
    }

    public init(
        artifactID: String,
        mediaType: String = "application/vnd.edgestudio.halo-capsule",
        totalBytes: Int,
        sha256: String,
        files: [HaloCapsuleArtifactFile]
    ) {
        self.artifactID = artifactID
        self.mediaType = mediaType
        self.totalBytes = totalBytes
        self.sha256 = sha256
        self.files = files
    }

    public func validate() throws {
        guard !artifactID.isEmpty else {
            throw HaloCapsuleTransportError.emptyArtifactID
        }
        try requireSHA256(sha256, field: "artifact.sha256")
        guard !files.isEmpty else {
            throw HaloCapsuleTransportError.emptyArtifactFiles
        }
        var actualBytes = 0
        for file in files {
            try file.validate()
            actualBytes += file.byteCount
        }
        guard actualBytes == totalBytes else {
            throw HaloCapsuleTransportError.invalidArtifactByteCount(
                expected: totalBytes,
                actual: actualBytes
            )
        }
    }

    fileprivate func normalizedForCanonicalEncoding() -> HaloCapsuleArtifactDescriptor {
        var copy = self
        copy.files = copy.files
            .map { $0.normalizedForCanonicalEncoding() }
            .sorted { $0.name < $1.name }
        return copy
    }
}

public struct HaloCapsuleArtifactFile: Codable, Equatable, Sendable {
    public var name: String
    public var byteCount: Int
    public var sha256: String
    public var downloadURL: URL?

    enum CodingKeys: String, CodingKey {
        case name
        case byteCount = "byte_count"
        case sha256
        case downloadURL = "download_url"
    }

    public init(name: String, byteCount: Int, sha256: String, downloadURL: URL? = nil) {
        self.name = name
        self.byteCount = byteCount
        self.sha256 = sha256
        self.downloadURL = downloadURL
    }

    public func validate() throws {
        guard byteCount >= 0 else {
            throw HaloCapsuleTransportError.invalidByteCount(
                field: "artifact.files[].byte_count",
                value: byteCount
            )
        }
        try requireSHA256(sha256, field: "artifact.files[].sha256")
    }

    fileprivate func normalizedForCanonicalEncoding() -> HaloCapsuleArtifactFile {
        var copy = self
        copy.downloadURL = nil
        return copy
    }
}

private func requireSHA256(_ value: String, field: String) throws {
    guard !value.isEmpty else {
        throw HaloCapsuleTransportError.emptySHA256(field: field)
    }
}

private enum HaloCapsuleCanonicalJSON {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(value)
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private enum HaloCapsuleVersion {
    static func compare(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let left = ParsedVersion(lhs)
        let right = ParsedVersion(rhs)
        let count = max(left.core.count, right.core.count)
        for index in 0..<count {
            let a = index < left.core.count ? left.core[index] : 0
            let b = index < right.core.count ? right.core[index] : 0
            if a < b { return .orderedAscending }
            if a > b { return .orderedDescending }
        }
        switch (left.prerelease, right.prerelease) {
        case (nil, nil):
            return .orderedSame
        case (nil, .some):
            return .orderedDescending
        case (.some, nil):
            return .orderedAscending
        case let (.some(leftTokens), .some(rightTokens)):
            return comparePrerelease(leftTokens, rightTokens)
        }
    }

    private struct ParsedVersion {
        let core: [Int]
        let prerelease: [PrereleaseToken]?

        init(_ value: String) {
            let parts = value.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            core = parts[0]
                .split(separator: ".")
                .map { part in
                    let digits = part.prefix { $0.isNumber }
                    return Int(digits) ?? 0
                }
            if parts.count > 1 {
                prerelease = HaloCapsuleVersion.prereleaseTokens(String(parts[1]))
            } else {
                prerelease = nil
            }
        }
    }

    private enum PrereleaseToken: Equatable {
        case number(Int)
        case text(String)
    }

    private static func prereleaseTokens(_ value: String) -> [PrereleaseToken] {
        value
            .split(separator: ".")
            .flatMap { segment in splitPrereleaseSegment(String(segment)) }
    }

    private static func splitPrereleaseSegment(_ segment: String) -> [PrereleaseToken] {
        var tokens: [PrereleaseToken] = []
        var buffer = ""
        var bufferIsNumber: Bool?

        func flush() {
            guard !buffer.isEmpty else { return }
            if bufferIsNumber == true {
                tokens.append(.number(Int(buffer) ?? 0))
            } else {
                tokens.append(.text(buffer.lowercased()))
            }
            buffer = ""
            bufferIsNumber = nil
        }

        for character in segment {
            let isNumber = character.isNumber
            if bufferIsNumber == nil || bufferIsNumber == isNumber {
                buffer.append(character)
                bufferIsNumber = isNumber
            } else {
                flush()
                buffer.append(character)
                bufferIsNumber = isNumber
            }
        }
        flush()
        return tokens
    }

    private static func comparePrerelease(
        _ lhs: [PrereleaseToken],
        _ rhs: [PrereleaseToken]
    ) -> ComparisonResult {
        let count = max(lhs.count, rhs.count)
        for index in 0..<count {
            if index >= lhs.count { return .orderedAscending }
            if index >= rhs.count { return .orderedDescending }
            let result = comparePrereleaseToken(lhs[index], rhs[index])
            if result != .orderedSame { return result }
        }
        return .orderedSame
    }

    private static func comparePrereleaseToken(
        _ lhs: PrereleaseToken,
        _ rhs: PrereleaseToken
    ) -> ComparisonResult {
        switch (lhs, rhs) {
        case let (.number(a), .number(b)):
            if a < b { return .orderedAscending }
            if a > b { return .orderedDescending }
            return .orderedSame
        case let (.text(a), .text(b)):
            return a.compare(b)
        case (.number, .text):
            return .orderedAscending
        case (.text, .number):
            return .orderedDescending
        }
    }
}
