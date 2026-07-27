// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import EdgeEngine
import Foundation
import Tokenizers

public struct NeuralImprintRuntimeCacheStatus: Equatable, Sendable {
    public var directory: URL
    public var artifactURL: URL
    public var metadataURL: URL
    public var artifactSHA256: String
    public var prefixTokenCount: Int
    public var modelID: String
    public var enableThinking: Bool
    public var cacheBackend: String
    public var cacheBackendVersion: String

    public init(
        directory: URL,
        artifactURL: URL,
        metadataURL: URL,
        artifactSHA256: String,
        prefixTokenCount: Int,
        modelID: String,
        enableThinking: Bool,
        cacheBackend: String,
        cacheBackendVersion: String
    ) {
        self.directory = directory
        self.artifactURL = artifactURL
        self.metadataURL = metadataURL
        self.artifactSHA256 = artifactSHA256
        self.prefixTokenCount = prefixTokenCount
        self.modelID = modelID
        self.enableThinking = enableThinking
        self.cacheBackend = cacheBackend
        self.cacheBackendVersion = cacheBackendVersion
    }
}

struct NeuralImprintRuntimeModelIdentityHashes {
    let modelDirectoryPath: String
    let modelArchitectureID: String
    let modelConfigSHA256: String
    let modelWeightsFingerprint: String
    let tokenizerJSONSHA256: String
    let tokenizerConfigSHA256: String
    let chatTemplateSHA256: String
}

enum NeuralImprintRuntimeSupport {
    static let artifactFileName = "neural_imprint.safetensors"
    static let legacyArtifactFileName = "persona_kv.safetensors"
    static let metadataFileName = "neural_imprint_metadata.json"
    static let legacyMetadataFileName = "persona_kv_metadata.json"
    static let prefixSplitSentinel = "__NEURAL_IMPRINT_TOOLS_SPLIT_SENTINEL__"
    static let prefixUserMarker = "<|im_start|>user\n"

    static func renderPrefix(
        profileBody: String,
        tools: [ToolSpec],
        parameters requestedParameters: EdgeGenerateParameters,
        tokenizer: Tokenizer,
        additionalContext: (EdgeGenerateParameters) -> [String: any Sendable]
    ) throws -> NeuralImprintPrefixRender {
        var parameters = requestedParameters
        parameters.enableThinking = false
        parameters.preserveThinking = false

        let systemPrompt = neuralImprintSystemPrompt(profileBody: profileBody)
        let messages: [ChatMessage] = [
            .system(systemPrompt),
            .user(prefixSplitSentinel),
        ]
        let promptTools = tools.isEmpty ? nil : tools
        let allTokenIDs = try tokenizer.applyChatTemplate(
            messages: messages.chatTemplateMessages(
                preserveThinking: parameters.preserveThinking
            ),
            tools: promptTools,
            additionalContext: additionalContext(parameters)
        )
        let markerTokenIDs = tokenizer.encode(
            text: prefixUserMarker,
            addSpecialTokens: false
        )
        guard let splitIndex = neuralImprintPrefixSplitIndex(
            fullTokenIDs: allTokenIDs,
            markerTokenIDs: markerTokenIDs
        ) else {
            throw EdgeRuntimeError.loadFailed("Neural Imprint prefix split marker not found")
        }

        let prefixTokenIDs = Array(allTokenIDs[..<splitIndex])
        guard !prefixTokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        let renderedPrefix = tokenizer.decode(
            tokens: prefixTokenIDs,
            skipSpecialTokens: false
        )
        return NeuralImprintPrefixRender(
            systemPrompt: systemPrompt,
            renderedPrefix: renderedPrefix,
            prefixTokenIDs: prefixTokenIDs,
            enableThinking: parameters.enableThinking
        )
    }

    static func renderPromptTokenIDs(
        promptMessages: [ChatMessage],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters,
        tokenizer: Tokenizer,
        additionalContext: (EdgeGenerateParameters) -> [String: any Sendable]
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: promptMessages.chatTemplateMessages(
                preserveThinking: parameters.preserveThinking
            ),
            tools: tools,
            additionalContext: additionalContext(parameters)
        )
    }

    static func loadCacheStatus(
        directory: URL,
        modelDirectory: URL,
        architecture: QwenHybridArchitecture
    ) throws -> NeuralImprintRuntimeCacheStatus {
        let metadataURL = neuralImprintMetadataURL(directory: directory)
        let sidecar = try NeuralImprintSidecar.load(from: metadataURL)
        let artifactURL = neuralImprintArtifactURL(
            sidecarArtifactPath: sidecar.artifact,
            directory: directory
        )
        let artifact = try SafeTensorsShardFile(url: artifactURL)
        let requirements = try neuralImprintCompatibilityRequirements(
            modelDirectory: modelDirectory,
            architecture: architecture,
            artifactHeader: artifact.metadata
        )
        try NeuralImprintArtifactValidator.validate(
            artifact: artifact,
            sidecar: sidecar,
            requirements: requirements
        )

        return NeuralImprintRuntimeCacheStatus(
            directory: directory,
            artifactURL: artifactURL,
            metadataURL: metadataURL,
            artifactSHA256: sidecar.artifactSHA256,
            prefixTokenCount: sidecar.prefix.tokenCount,
            modelID: sidecar.model.id,
            enableThinking: try requiredNeuralImprintHeader("enable_thinking", in: artifact.metadata) == "true",
            cacheBackend: try requiredNeuralImprintHeader("cache_backend", in: artifact.metadata),
            cacheBackendVersion: try requiredNeuralImprintHeader("cache_backend_version", in: artifact.metadata)
        )
    }

    static func neuralImprintSystemPrompt(profileBody: String) -> String {
        profileBody.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func neuralImprintPrefixSplitIndex(
        fullTokenIDs: [Int],
        markerTokenIDs: [Int]
    ) -> Int? {
        guard !markerTokenIDs.isEmpty,
              markerTokenIDs.count <= fullTokenIDs.count
        else {
            return nil
        }

        let lastStartIndex = fullTokenIDs.count - markerTokenIDs.count
        let matches = (0...lastStartIndex).filter { index in
            Array(fullTokenIDs[index..<index + markerTokenIDs.count]) == markerTokenIDs
        }
        guard matches.count == 1 else { return nil }
        return matches[0]
    }

    static func neuralImprintCompatibleParameters(
        _ parameters: EdgeGenerateParameters
    ) -> EdgeGenerateParameters {
        var updated = parameters
        updated.useDSR = false
        updated.dsrMaxCritical = nil
        updated.dsrHeavyBudget = nil
        updated.dsrRecentBudget = nil
        updated.dsrEvictionInterval = 0
        updated.kvBits = nil
        updated.quantizedKVStart = 0
        updated.maxKVSize = nil
        updated.frogJumpEnabled = false
        return updated
    }

    static func neuralImprintCapturePrefillStep(
        prefixTokenCount: Int,
        planPrefillStepSize: Int?,
        syncPrefill: Bool = false
    ) -> Int {
        let planned = planPrefillStepSize ?? 128
        let requested = NativeEnvironment.int(
            ["EDGE_NEURAL_IMPRINT_CAPTURE_PREFILL_STEP", "EDGE_CMLX_NEURAL_IMPRINT_CAPTURE_PREFILL_STEP"],
            defaultValue: 0,
            range: 0...4096
        )
        if requested > 0 {
            return max(1, min(max(1, prefixTokenCount), requested))
        }
        let hardLimit = syncPrefill ? 32 : 128
        return max(1, min(max(1, prefixTokenCount), min(max(1, planned), hardLimit)))
    }

    static func neuralImprintCaptureUsesSyncPrefill(
        memorySnapshot: DeviceProfile.MemorySnapshot,
        planSyncEval: Bool?
    ) -> Bool {
        if let override = NativeEnvironment.boolOverride([
            "EDGE_NEURAL_IMPRINT_CAPTURE_SYNC_PREFILL",
            "EDGE_CMLX_NEURAL_IMPRINT_CAPTURE_SYNC_PREFILL",
        ]) {
            return override
        }
        if planSyncEval == true {
            return true
        }
        return memorySnapshot.jetsamLimitMB <= 6_144 || memorySnapshot.availableMB < 1_536
    }

    static func canReuseCmlxSessionForCleanCapture(
        dsrPolicyCount: Int,
        hasAttentionCacheQuantization: Bool,
        frogJumpLayerMask: UInt64
    ) -> Bool {
        dsrPolicyCount == 0 &&
            !hasAttentionCacheQuantization &&
            frogJumpLayerMask == 0
    }

    static func prefillNeuralImprintCapture(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        chunkSize: Int,
        syncPrefill: Bool,
        diagnosticSink: ((String) -> Void)? = nil
    ) throws {
        let step = max(1, chunkSize)
        let chunkCount = Int(ceil(Double(tokenIDs.count) / Double(step)))
        diagnosticSink?(
            "neural_imprint_capture_prefill_begin total=\(tokenIDs.count) step=\(step) chunks=\(chunkCount) mode=\(syncPrefill ? "sync" : "async")"
        )

        var offset = 0
        var chunkIndex = 0
        while offset < tokenIDs.count {
            let end = min(offset + step, tokenIDs.count)
            let chunk = Array(tokenIDs[offset..<end])
            chunkIndex += 1
            if syncPrefill {
                _ = try session.prefill(tokenIDs: chunk)
            } else {
                try session.prefillAsync(tokenIDs: chunk)
            }
            diagnosticSink?(
                "neural_imprint_capture_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=\(end == tokenIDs.count)"
            )
            offset = end
        }
    }

    static func captureArtifact(
        request: NeuralImprintArtifactCaptureRequest,
        modelDirectory: URL,
        architecture: QwenHybridArchitecture,
        modelID: String,
        modelIdentityHashes: NeuralImprintRuntimeModelIdentityHashes,
        session: QwenCmlxLazyDecodeSession,
        capturePrefillStep: Int,
        captureUsesSyncPrefill: Bool,
        diagnosticPrefix: String = "",
        diagnosticSink: ((String) -> Void)? = nil
    ) throws -> NeuralImprintRuntimeCacheStatus {
        func emit(_ marker: String) {
            diagnosticSink?("\(diagnosticPrefix)\(marker)")
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: request.outputDirectory,
            withIntermediateDirectories: true
        )
        let artifactURL = request.outputDirectory.appendingPathComponent(artifactFileName)
        let metadataURL = request.outputDirectory.appendingPathComponent(metadataFileName)
        let profileBodyURL = request.outputDirectory.appendingPathComponent(request.profileBodyFileName)
        let toolSpecsURL = request.outputDirectory.appendingPathComponent(request.toolSpecsFileName)
        emit("neural_imprint_capture_source_write_begin")
        try Data(request.profileBody.utf8).write(to: profileBodyURL, options: [.atomic])
        try request.toolSchemaSnapshot.jsonData.write(to: toolSpecsURL, options: [.atomic])
        emit("neural_imprint_capture_source_write_done")

        let prefixTokenCount = request.prefixTokenIDs.count
        let profileBodySHA256 = sha256Text(request.profileBody)
        let renderedPrefixSHA256 = sha256Text(request.renderedPrefix)
        let prefixTokenIDsSHA256 = neuralImprintTokenIDsSHA256(request.prefixTokenIDs)
        emit("neural_imprint_capture_header_begin")
        let header = try neuralImprintArtifactHeader(
            modelID: modelID,
            modelDirectory: modelDirectory,
            modelArchitectureID: modelIdentityHashes.modelArchitectureID,
            modelConfigSHA256: modelIdentityHashes.modelConfigSHA256,
            modelWeightsFingerprint: modelIdentityHashes.modelWeightsFingerprint,
            tokenizerJSONSHA256: modelIdentityHashes.tokenizerJSONSHA256,
            tokenizerConfigSHA256: modelIdentityHashes.tokenizerConfigSHA256,
            chatTemplateSHA256: modelIdentityHashes.chatTemplateSHA256,
            systemPromptSHA256: sha256Text(request.systemPrompt),
            renderedPrefixSHA256: renderedPrefixSHA256,
            prefixTokenIDsSHA256: prefixTokenIDsSHA256,
            prefixTokenCount: prefixTokenCount,
            toolSchemaSHA256: request.toolSchemaSnapshot.sha256,
            profileBodySHA256: profileBodySHA256,
            enableThinking: request.enableThinking,
            cacheBackendVersion: request.cacheBackendVersion,
            createdAt: request.createdAt,
            createdBy: request.createdBy,
            writerVersion: request.writerVersion,
            minReaderVersion: request.minReaderVersion
        )
        emit("neural_imprint_capture_header_done")

        try prefillNeuralImprintCapture(
            session: session,
            tokenIDs: request.prefixTokenIDs,
            chunkSize: capturePrefillStep,
            syncPrefill: captureUsesSyncPrefill,
            diagnosticSink: { emit($0) }
        )
        emit("neural_imprint_capture_save_begin")
        try session.session.saveNeuralImprintCache(
            artifactURL: artifactURL,
            metadata: header
        )
        emit("neural_imprint_capture_save_done")
        emit("neural_imprint_capture_session_reset_begin")
        try session.reset()
        emit("neural_imprint_capture_session_reset_done")

        emit("neural_imprint_capture_artifact_map_begin")
        let artifact = try SafeTensorsShardFile(url: artifactURL)
        emit("neural_imprint_capture_artifact_map_done")
        emit("neural_imprint_capture_sidecar_payload_begin")
        let sidecarPayload = try neuralImprintSidecarPayload(
            artifactURL: artifactURL,
            artifact: artifact,
            request: request,
            modelID: modelID,
            architecture: architecture,
            tokenizerJSONSHA256: modelIdentityHashes.tokenizerJSONSHA256,
            tokenizerConfigSHA256: modelIdentityHashes.tokenizerConfigSHA256,
            chatTemplateSHA256: modelIdentityHashes.chatTemplateSHA256,
            renderedPrefixSHA256: renderedPrefixSHA256,
            prefixTokenIDsSHA256: prefixTokenIDsSHA256,
            profileBodySHA256: profileBodySHA256
        )
        emit("neural_imprint_capture_sidecar_payload_done")
        emit("neural_imprint_capture_sidecar_encode_begin")
        let sidecarData = try JSONSerialization.data(
            withJSONObject: sidecarPayload,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        emit("neural_imprint_capture_sidecar_encode_done")
        emit("neural_imprint_capture_sidecar_write_begin")
        try sidecarData.write(to: metadataURL, options: [.atomic])
        emit("neural_imprint_capture_sidecar_write_done")

        emit("neural_imprint_capture_sidecar_load_begin")
        let sidecar = try NeuralImprintSidecar.load(from: metadataURL)
        emit("neural_imprint_capture_sidecar_load_done")
        emit("neural_imprint_capture_requirements_begin")
        let requirements = try neuralImprintCompatibilityRequirements(
            modelDirectory: modelDirectory,
            architecture: architecture,
            artifactHeader: artifact.metadata,
            modelArchitectureID: modelIdentityHashes.modelArchitectureID,
            modelConfigSHA256: modelIdentityHashes.modelConfigSHA256,
            modelWeightsFingerprint: modelIdentityHashes.modelWeightsFingerprint,
            tokenizerJSONSHA256: modelIdentityHashes.tokenizerJSONSHA256,
            tokenizerConfigSHA256: modelIdentityHashes.tokenizerConfigSHA256,
            chatTemplateSHA256: modelIdentityHashes.chatTemplateSHA256
        )
        emit("neural_imprint_capture_requirements_done")
        emit("neural_imprint_capture_validator_begin")
        try NeuralImprintArtifactValidator.validate(
            artifact: artifact,
            sidecar: sidecar,
            requirements: requirements
        )
        emit("neural_imprint_capture_validator_done")

        emit("neural_imprint_capture_status_begin")
        let status = NeuralImprintRuntimeCacheStatus(
            directory: request.outputDirectory,
            artifactURL: artifactURL,
            metadataURL: metadataURL,
            artifactSHA256: sidecar.artifactSHA256,
            prefixTokenCount: prefixTokenCount,
            modelID: modelID,
            enableThinking: request.enableThinking,
            cacheBackend: try requiredNeuralImprintHeader("cache_backend", in: artifact.metadata),
            cacheBackendVersion: try requiredNeuralImprintHeader("cache_backend_version", in: artifact.metadata)
        )
        emit("neural_imprint_capture_status_done")
        emit(
            "neural_imprint_capture_done prefix=\(status.prefixTokenCount) artifactSHA256=\(status.artifactSHA256)"
        )
        return status
    }

    static func neuralImprintArtifactURL(
        sidecarArtifactPath: String,
        directory: URL
    ) -> URL {
        let fileManager = FileManager.default
        let isAbsolutePath = (sidecarArtifactPath as NSString).isAbsolutePath
        let sidecarURL = URL(fileURLWithPath: sidecarArtifactPath)
        if isAbsolutePath,
           fileManager.fileExists(atPath: sidecarURL.path) {
            return sidecarURL
        }

        let relativeURL = isAbsolutePath
            ? directory.appendingPathComponent((sidecarArtifactPath as NSString).lastPathComponent)
            : directory.appendingPathComponent(sidecarArtifactPath)
        if fileManager.fileExists(atPath: relativeURL.path) {
            return relativeURL
        }

        let defaultURLs = [
            directory.appendingPathComponent(artifactFileName),
            directory.appendingPathComponent(legacyArtifactFileName),
        ]
        if let defaultURL = defaultURLs.first(where: { fileManager.fileExists(atPath: $0.path) }) {
            return defaultURL
        }
        return relativeURL
    }

    static func neuralImprintMetadataURL(directory: URL) -> URL {
        let fileManager = FileManager.default
        let candidates = [
            directory.appendingPathComponent(metadataFileName),
            directory.appendingPathComponent(legacyMetadataFileName),
        ]
        return candidates.first(where: { fileManager.fileExists(atPath: $0.path) }) ?? candidates[0]
    }

    static func neuralImprintCompatibilityRequirements(
        modelDirectory: URL,
        architecture: QwenHybridArchitecture,
        artifactHeader: [String: String],
        modelArchitectureID: String? = nil,
        modelConfigSHA256: String? = nil,
        modelWeightsFingerprint: String? = nil,
        tokenizerJSONSHA256: String? = nil,
        tokenizerConfigSHA256: String? = nil,
        chatTemplateSHA256: String? = nil
    ) throws -> NeuralImprintCompatibilityRequirements {
        let modelArchitecture: String
        if let modelArchitectureID {
            modelArchitecture = modelArchitectureID
        } else {
            modelArchitecture = try neuralImprintModelArchitectureIdentifier(modelDirectory: modelDirectory)
        }
        let cacheBackend = try requiredNeuralImprintHeader("cache_backend", in: artifactHeader)
        guard cacheBackend == "mlx-lm" else {
            throw EdgeRuntimeError.unsupportedFeature(
                "Unsupported Neural Imprint cache backend: \(cacheBackend)"
            )
        }
        return try NeuralImprintCompatibilityRequirements(
            modelArchitecture: modelArchitecture,
            modelConfigSHA256: modelConfigSHA256
                ?? sha256File(modelDirectory.appendingPathComponent("config.json")),
            modelWeightsFingerprint: modelWeightsFingerprint
                ?? neuralImprintModelWeightsFingerprint(modelDirectory: modelDirectory),
            tokenizerJSONSHA256: tokenizerJSONSHA256
                ?? sha256File(modelDirectory.appendingPathComponent("tokenizer.json")),
            tokenizerConfigSHA256: tokenizerConfigSHA256
                ?? sha256File(modelDirectory.appendingPathComponent("tokenizer_config.json")),
            chatTemplateSHA256: chatTemplateSHA256 ?? neuralImprintChatTemplateSHA256(
                modelDirectory: modelDirectory,
                expectedSHA256: requiredNeuralImprintHeader("chat_template_sha256", in: artifactHeader)
            ),
            renderedPrefixSHA256: requiredNeuralImprintHeader("rendered_prefix_sha256", in: artifactHeader),
            prefixTokenIDsSHA256: requiredNeuralImprintHeader("prefix_token_ids_sha256", in: artifactHeader),
            enableThinking: requiredNeuralImprintHeader("enable_thinking", in: artifactHeader),
            cacheBackend: cacheBackend,
            cacheBackendVersion: requiredNeuralImprintHeader("cache_backend_version", in: artifactHeader),
            cacheTopology: .qwen35(architecture: architecture)
        )
    }

    static func neuralImprintArtifactHeader(
        modelID: String,
        modelDirectory: URL,
        modelArchitectureID: String,
        modelConfigSHA256: String,
        modelWeightsFingerprint: String,
        tokenizerJSONSHA256: String,
        tokenizerConfigSHA256: String,
        chatTemplateSHA256: String,
        systemPromptSHA256: String,
        renderedPrefixSHA256: String,
        prefixTokenIDsSHA256: String,
        prefixTokenCount: Int,
        toolSchemaSHA256: String,
        profileBodySHA256: String,
        enableThinking: Bool,
        cacheBackendVersion: String,
        createdAt: Date,
        createdBy: String,
        writerVersion: String,
        minReaderVersion: String
    ) throws -> [String: String] {
        [
            "format": "mlx",
            "artifact_type": NeuralImprintArtifactValidator.artifactType,
            "artifact_version": NeuralImprintArtifactValidator.artifactVersion,
            "cache_schema": NeuralImprintArtifactValidator.cacheSchema,
            "model_id": modelID,
            "model_architecture": modelArchitectureID,
            "model_config_sha256": modelConfigSHA256,
            "model_weights_fingerprint": modelWeightsFingerprint,
            "tokenizer_json_sha256": tokenizerJSONSHA256,
            "tokenizer_config_sha256": tokenizerConfigSHA256,
            "chat_template_sha256": chatTemplateSHA256,
            "system_prompt_sha256": systemPromptSHA256,
            "rendered_prefix_sha256": renderedPrefixSHA256,
            "prefix_token_ids_sha256": prefixTokenIDsSHA256,
            "prefix_token_count": String(prefixTokenCount),
            "prefix_renderer_version": NeuralImprintArtifactValidator.prefixRendererVersion,
            "tool_schema_sha256": toolSchemaSHA256,
            "profile_body_sha256": profileBodySHA256,
            "enable_thinking": enableThinking ? "true" : "false",
            "cache_backend": "mlx-lm",
            "cache_backend_version": cacheBackendVersion,
            "created_at": neuralImprintIso8601String(createdAt),
            "created_by": createdBy,
            "writer_version": writerVersion,
            "min_reader_version": minReaderVersion,
        ]
    }

    static func neuralImprintSidecarPayload(
        artifactURL: URL,
        artifact: SafeTensorsShardFile,
        request: NeuralImprintArtifactCaptureRequest,
        modelID: String,
        architecture: QwenHybridArchitecture,
        tokenizerJSONSHA256: String,
        tokenizerConfigSHA256: String,
        chatTemplateSHA256: String,
        renderedPrefixSHA256: String,
        prefixTokenIDsSHA256: String,
        profileBodySHA256: String
    ) throws -> [String: Any] {
        [
            "schema": NeuralImprintArtifactValidator.sidecarSchema,
            "artifact": artifactURL.lastPathComponent,
            "artifact_sha256": try sha256File(artifactURL),
            "source": [
                "profile_body_path": request.profileBodyFileName,
                "profile_body_sha256": profileBodySHA256,
                "tool_specs_path": request.toolSpecsFileName,
                "tool_schema_sha256": request.toolSchemaSnapshot.sha256,
            ],
            "model": [
                "id": modelID,
                "architecture": artifact.metadata["model_architecture"] ?? "qwen3_5",
                "hidden_size": architecture.hiddenSize,
                "num_layers": architecture.layerCount,
                "quantization": neuralImprintQuantizationPayload(architecture.quantization),
                "layer_types": architecture.layerPlan.map { layer in
                    layer.kind == .fullAttention ? "full_attention" : "linear_attention"
                },
            ],
            "tokenizer": [
                "tokenizer_json_sha256": tokenizerJSONSHA256,
                "tokenizer_config_sha256": tokenizerConfigSHA256,
                "chat_template_sha256": chatTemplateSHA256,
                "enable_thinking": request.enableThinking,
            ],
            "prefix": [
                "token_count": request.prefixTokenIDs.count,
                "rendered_prefix_sha256": renderedPrefixSHA256,
                "token_ids_sha256": prefixTokenIDsSHA256,
            ],
            "cache_manifest": try neuralImprintCacheManifest(
                artifact: artifact,
                architecture: architecture,
                prefixTokenCount: request.prefixTokenIDs.count
            ),
        ]
    }

    static func neuralImprintCacheManifest(
        artifact: SafeTensorsShardFile,
        architecture: QwenHybridArchitecture,
        prefixTokenCount: Int
    ) throws -> [String: Any] {
        let layers = try architecture.layerPlan.map { layer -> [String: Any] in
            let cacheClass = layer.kind == .fullAttention ? "KVCache" : "ArraysCache"
            let stateContainer = layer.kind == .fullAttention ? "tuple" : "list"
            let states = try (0..<2).map { stateIndex -> [String: Any] in
                let name = String(format: "layer_%02d.state_%d", layer.index, stateIndex)
                let tensor = try artifact.metadata(named: name)
                return [
                    "name": name,
                    "shape": tensor.shape,
                    "dtype": neuralImprintManifestDType(tensor.dtype),
                ]
            }
            return [
                "layer": layer.index,
                "cache_class": cacheClass,
                "state_container": stateContainer,
                "state_count": states.count,
                "states": states,
                "offset": layer.kind == .fullAttention ? prefixTokenCount : NSNull(),
                "meta_state": "",
            ]
        }
        return [
            "layer_count": layers.count,
            "layers": layers,
        ]
    }

    static func neuralImprintTokenIDsSHA256(_ tokenIDs: [Int]) -> String {
        let material = "[\(tokenIDs.map(String.init).joined(separator: ","))]"
        return sha256Text(material)
    }

    static func requiredNeuralImprintHeader(
        _ key: String,
        in header: [String: String]
    ) throws -> String {
        guard let value = header[key], !value.isEmpty else {
            throw NeuralImprintArtifactError.missingHeaderField(key)
        }
        return value
    }

    static func neuralImprintModelArchitectureIdentifier(modelDirectory: URL) throws -> String {
        let config = try jsonDictionary(
            at: modelDirectory.appendingPathComponent("config.json")
        )
        if let modelType = config["model_type"] as? String, !modelType.isEmpty {
            return modelType
        }
        if let textConfig = config["text_config"] as? [String: Any],
           let modelType = textConfig["model_type"] as? String,
           !modelType.isEmpty {
            return modelType
        }
        throw EdgeRuntimeError.loadFailed("config.json is missing model_type")
    }

    static func computeModelIdentityHashes(
        modelDirectory: URL
    ) throws -> NeuralImprintRuntimeModelIdentityHashes {
        let chatTemplateSHA256 = try neuralImprintChatTemplateSHA256(modelDirectory: modelDirectory)
        return try NeuralImprintRuntimeModelIdentityHashes(
            modelDirectoryPath: modelDirectory.standardizedFileURL.path,
            modelArchitectureID: neuralImprintModelArchitectureIdentifier(modelDirectory: modelDirectory),
            modelConfigSHA256: sha256File(modelDirectory.appendingPathComponent("config.json")),
            modelWeightsFingerprint: neuralImprintModelWeightsFingerprint(modelDirectory: modelDirectory),
            tokenizerJSONSHA256: sha256File(modelDirectory.appendingPathComponent("tokenizer.json")),
            tokenizerConfigSHA256: sha256File(modelDirectory.appendingPathComponent("tokenizer_config.json")),
            chatTemplateSHA256: chatTemplateSHA256
        )
    }

    static func neuralImprintChatTemplateSHA256(
        modelDirectory: URL,
        expectedSHA256: String? = nil
    ) throws -> String {
        let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        let template: String
        if let tokenizerConfig = try? jsonDictionary(at: tokenizerConfigURL),
           let chatTemplate = tokenizerConfig["chat_template"] as? String,
           !chatTemplate.isEmpty {
            template = chatTemplate
        } else {
            let templateURL = modelDirectory.appendingPathComponent("chat_template.jinja")
            template = try String(contentsOf: templateURL, encoding: .utf8)
        }

        let directHash = sha256Text(template)
        guard let expectedSHA256, directHash != expectedSHA256 else {
            return directHash
        }
        let canonicalTemplate = qwenTransformersCanonicalChatTemplate(template)
        let canonicalHash = sha256Text(canonicalTemplate)
        return canonicalHash == expectedSHA256 ? canonicalHash : directHash
    }

    static func neuralImprintModelWeightsFingerprint(modelDirectory: URL) throws -> String {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        let indexSHA256 = FileManager.default.fileExists(atPath: indexURL.path)
            ? try sha256File(indexURL)
            : nil
        let shards = try neuralImprintModelWeightFiles(modelDirectory: modelDirectory)
            .map { url -> (name: String, sizeBytes: Int, sha256: String) in
                let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                return (url.lastPathComponent, size, try sha256File(url))
            }
        let material = neuralImprintWeightsFingerprintJSON(
            indexSHA256: indexSHA256,
            shards: shards
        )
        return "sha256:" + sha256Text(material)
    }

    static func neuralImprintModelWeightFiles(modelDirectory: URL) throws -> [URL] {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(
                SafetensorsWeightIndex.self,
                from: try Data(contentsOf: indexURL)
            )
            return Set(index.weightMap.values)
                .sorted()
                .map { modelDirectory.appendingPathComponent($0) }
        }

        return try FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )
        .filter { url in
            let name = url.lastPathComponent
            return name.hasPrefix("model") && name.hasSuffix(".safetensors")
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func neuralImprintWeightsFingerprintJSON(
        indexSHA256: String?,
        shards: [(name: String, sizeBytes: Int, sha256: String)]
    ) -> String {
        let indexValue = indexSHA256.map { "\"\(jsonEscaped($0))\"" } ?? "null"
        let shardObjects = shards.map { shard in
            "{\"name\":\"\(jsonEscaped(shard.name))\",\"sha256\":\"\(jsonEscaped(shard.sha256))\",\"size_bytes\":\(shard.sizeBytes)}"
        }
        return "{\"index_sha256\":\(indexValue),\"shards\":[\(shardObjects.joined(separator: ","))]}"
    }

    static func jsonDictionary(at url: URL) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
        guard let dictionary = object as? [String: Any] else {
            throw EdgeRuntimeError.loadFailed("Expected JSON object at \(url.lastPathComponent)")
        }
        return dictionary
    }

    static func sha256File(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let data = try autoreleasepool {
                try handle.read(upToCount: 1024 * 1024) ?? Data()
            }
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func sha256Text(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func neuralImprintQuantizationPayload(
        _ quantization: QwenQuantizationProfile?
    ) -> [String: Any] {
        guard let quantization else {
            return ["bits": 0, "group_size": 0, "mode": "none"]
        }
        return [
            "bits": quantization.bits,
            "group_size": quantization.groupSize,
            "mode": "affine",
        ]
    }

    private static func neuralImprintManifestDType(_ safetensorsDType: String) -> String {
        switch safetensorsDType.uppercased() {
        case "BF16":
            return "mlx.core.bfloat16"
        case "F16":
            return "mlx.core.float16"
        case "F32":
            return "mlx.core.float32"
        default:
            return safetensorsDType
        }
    }

    private static func neuralImprintIso8601String(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withColonSeparatorInTimeZone]
        return formatter.string(from: date)
    }

    private static func qwenTransformersCanonicalChatTemplate(_ template: String) -> String {
        let source = #"""
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
"""#
        let replacement = #"""
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
"""#
        return template.replacingOccurrences(of: source, with: replacement)
    }

    private static func jsonEscaped(_ value: String) -> String {
        var escaped = ""
        escaped.reserveCapacity(value.count)
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x22:
                escaped += "\\\""
            case 0x5C:
                escaped += "\\\\"
            case 0x08:
                escaped += "\\b"
            case 0x0C:
                escaped += "\\f"
            case 0x0A:
                escaped += "\\n"
            case 0x0D:
                escaped += "\\r"
            case 0x09:
                escaped += "\\t"
            default:
                if scalar.value < 0x20 {
                    escaped += String(format: "\\u%04x", scalar.value)
                } else {
                    escaped.unicodeScalars.append(scalar)
                }
            }
        }
        return escaped
    }

    private struct SafetensorsWeightIndex: Decodable {
        let weightMap: [String: String]

        private enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }
}
