// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
import EdgeEngine
@testable import EdgeInference

final class LLMEngineTests: XCTestCase {

    static let baselineGate = "EDGEKIT_RUN_NATIVE_LLM_BASELINE"
    static let baselineModelPath = "EDGEKIT_QWEN_BASELINE_MODEL"
    static let neuralImprintGate = "EDGEKIT_RUN_NEURAL_IMPRINT_REAL_ARTIFACT"
    static let neuralImprintModelPath = "EDGEKIT_NEURAL_IMPRINT_MODEL_PATH"
    static let neuralImprintArtifactDir = "EDGEKIT_NEURAL_IMPRINT_ARTIFACT_DIR"
    static let neuralImprintCaptureGate = "EDGEKIT_RUN_NEURAL_IMPRINT_CAPTURE_SMOKE"

    private func requireBaselineModel(requireGate: Bool = true) throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        if requireGate, environment[Self.baselineGate] != "1" {
            throw XCTSkip("set \(Self.baselineGate)=1 to run Qwen3.5-4B-6bit baseline smoke")
        }
        guard let modelPath = environment[Self.baselineModelPath],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("set \(Self.baselineModelPath) to a local Qwen baseline bundle")
        }
        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(
            atPath: modelURL.appendingPathComponent("config.json").path
        ) else {
            throw XCTSkip("local Qwen baseline fixture not present: \(modelURL.path)")
        }
        return modelURL
    }

    private func requireRealNeuralImprintFixture() throws -> (model: URL, artifactDirectory: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard environment[Self.neuralImprintGate] == "1" else {
            throw XCTSkip("set \(Self.neuralImprintGate)=1 to run real Neural Imprint restore smoke")
        }
        guard let modelPath = environment[Self.neuralImprintModelPath],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("set \(Self.neuralImprintModelPath) to a local model bundle")
        }
        guard let artifactPath = environment[Self.neuralImprintArtifactDir],
              !artifactPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("set \(Self.neuralImprintArtifactDir) to a local Neural Imprint artifact directory")
        }
        let model = URL(fileURLWithPath: modelPath)
        let artifactDirectory = URL(fileURLWithPath: artifactPath)
        for url in [
            model.appendingPathComponent("config.json"),
            model.appendingPathComponent("tokenizer.json"),
            model.appendingPathComponent("tokenizer_config.json"),
        ] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("real Neural Imprint fixture missing: \(url.path)")
            }
        }
        let artifactURL = LLMEngine.neuralImprintArtifactURL(
            sidecarArtifactPath: LLMEngine.neuralImprintArtifactFileName,
            directory: artifactDirectory
        )
        let metadataURL = LLMEngine.neuralImprintMetadataURL(directory: artifactDirectory)
        for url in [artifactURL, metadataURL] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("real Neural Imprint fixture missing: \(url.path)")
            }
        }
        return (model, artifactDirectory)
    }

    func testNeuralImprintCacheStatusConstructsStableValue() {
        let status = LLMEngine.NeuralImprintCacheStatus(
            directory: URL(fileURLWithPath: "/artifact"),
            artifactURL: URL(fileURLWithPath: "/artifact/neural_imprint.safetensors"),
            metadataURL: URL(fileURLWithPath: "/artifact/neural_imprint_metadata.json"),
            artifactSHA256: String(repeating: "a", count: 64),
            prefixTokenCount: 2_438,
            modelID: "Qwen3.5-9B-4bit",
            enableThinking: false,
            cacheBackend: "cmlx-full-cache",
            cacheBackendVersion: "edge-engine 1.0.0-rc138"
        )
        XCTAssertEqual(status.modelID, "Qwen3.5-9B-4bit")
    }

    func testDSREvictionIntervalEnvironmentOverrideParsesPositiveInteger() {
        XCTAssertEqual(
            LLMEngine.dsrEvictionIntervalEnvironmentOverride(
                from: ["EDGE_DSR_EVICTION_INTERVAL": "64"]
            ),
            64
        )
        XCTAssertEqual(
            LLMEngine.dsrEvictionIntervalEnvironmentOverride(
                from: ["EDGE_CMLX_DSR_EVICTION_INTERVAL": " 32 "]
            ),
            32
        )
        XCTAssertNil(
            LLMEngine.dsrEvictionIntervalEnvironmentOverride(
                from: ["EDGE_DSR_EVICTION_INTERVAL": "0"]
            )
        )
        XCTAssertNil(
            LLMEngine.dsrEvictionIntervalEnvironmentOverride(
                from: ["EDGE_DSR_EVICTION_INTERVAL": "abc"]
            )
        )
    }

    func testKVBitsEnvironmentOverrideParsesNonNegativeIntegerAndDisableAliases() {
        XCTAssertEqual(
            LLMEngine.kvBitsEnvironmentOverride(
                from: ["EDGE_KV_BITS": "0"]
            ),
            0
        )
        XCTAssertEqual(
            LLMEngine.kvBitsEnvironmentOverride(
                from: ["EDGE_CMLX_KV_BITS": " 8 "]
            ),
            8
        )
        XCTAssertEqual(
            LLMEngine.kvBitsEnvironmentOverride(
                from: ["EDGE_KV_BITS": "off"]
            ),
            0
        )
        XCTAssertNil(
            LLMEngine.kvBitsEnvironmentOverride(
                from: ["EDGE_KV_BITS": "-1"]
            )
        )
        XCTAssertNil(
            LLMEngine.kvBitsEnvironmentOverride(
                from: ["EDGE_KV_BITS": "abc"]
            )
        )
    }

    func testToolCallDetectionWindowParsesBoundedPositiveOverride() {
        XCTAssertEqual(
            LLMEngine.toolCallDetectionWindow(from: [:]),
            50
        )
        XCTAssertEqual(LLMEngine.defaultToolCallDetectionWindow, 50)
        XCTAssertEqual(
            LLMEngine.toolCallDetectionWindow(from: ["EDGE_TOOL_CALL_DETECTION_WINDOW": "80"]),
            80
        )
        XCTAssertEqual(
            LLMEngine.toolCallDetectionWindow(from: ["EDGE_CMLX_TOOL_CALL_DETECTION_WINDOW": " 100 "]),
            100
        )
        XCTAssertEqual(
            LLMEngine.toolCallDetectionWindow(from: ["EDGE_TOOL_CALL_DETECTION_WINDOW": "999"]),
            512
        )
        XCTAssertEqual(
            LLMEngine.toolCallDetectionWindow(from: ["EDGE_TOOL_CALL_DETECTION_WINDOW": "0"]),
            LLMEngine.defaultToolCallDetectionWindow
        )
    }

    func testDecodedTextContainsToolCallStartMatchesPartialToolCallPrefix() {
        XCTAssertTrue(LLMEngine.decodedTextContainsToolCallStart("<tool_call"))
        XCTAssertTrue(LLMEngine.decodedTextContainsToolCallStart("<tool_call>"))
        XCTAssertTrue(LLMEngine.decodedTextContainsToolCallStart("prefix <TOOL_CALL> suffix"))
        XCTAssertFalse(LLMEngine.decodedTextContainsToolCallStart("普通直接回答，不调用工具。"))
    }

    func testCmlxRepetitionContextIncludesPromptAndGeneratedTokens() {
        XCTAssertEqual(
            NativeCmlxSampling.contextTokenIds(
                promptSessionTokenIds: [10, 11, 12]
            ),
            [10, 11, 12]
        )
        XCTAssertEqual(
            NativeCmlxSampling.contextTokenIds(
                promptSessionTokenIds: [10, 11, 12],
                generatedTokenIds: [20, 21]
            ),
            [10, 11, 12, 20, 21]
        )
        XCTAssertEqual(
            NativeCmlxSampling.contextTokenIds(
                promptSessionTokenIds: [10, 11, 12],
                generatedTokenIds: [20, 21],
                contextSize: 3
            ),
            [12, 20, 21]
        )
        XCTAssertEqual(
            NativeCmlxSampling.contextTokenIds(
                promptSessionTokenIds: [10, 11, 12],
                generatedTokenIds: [20, 21],
                contextSize: 0
            ),
            []
        )
    }

    func testHiddenStateCaptureRequiresLoadedModel() async {
        let engine = LLMEngine()
        do {
            _ = try await engine.captureHiddenStates(tokens: [1], targetLayer: 0)
            XCTFail("captureHiddenStates should reject unloaded models")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("No LLM model loaded"))
        }
    }

    func testNeuralImprintCompatibleParametersDisableIncompatibleCacheFeatures() {
        let requested = EdgeGenerateParameters(
            temperature: 0,
            maxTokens: 64,
            quantizedKVStart: 128,
            kvBits: 4,
            maxKVSize: 2_048,
            useDSR: true,
            dsrMaxCritical: 1_024,
            dsrHeavyBudget: 256,
            dsrRecentBudget: 128,
            dsrEvictionInterval: 64,
            frogJumpEnabled: true
        )

        let updated = LLMEngine.neuralImprintCompatibleParameters(requested)

        XCTAssertEqual(updated.temperature, requested.temperature)
        XCTAssertEqual(updated.maxTokens, requested.maxTokens)
        XCTAssertFalse(updated.useDSR)
        XCTAssertNil(updated.dsrMaxCritical)
        XCTAssertNil(updated.dsrHeavyBudget)
        XCTAssertNil(updated.dsrRecentBudget)
        XCTAssertEqual(updated.dsrEvictionInterval, 0)
        XCTAssertNil(updated.kvBits)
        XCTAssertEqual(updated.quantizedKVStart, 0)
        XCTAssertNil(updated.maxKVSize)
        XCTAssertFalse(updated.frogJumpEnabled)
    }

    func testNeuralImprintCapturePrefillStepIsDeviceConservative() {
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 1_018,
                planPrefillStepSize: 512
            ),
            128
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 96,
                planPrefillStepSize: 512
            ),
            96
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 1_018,
                planPrefillStepSize: 64
            ),
            64
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 1_018,
                planPrefillStepSize: 128,
                syncPrefill: true
            ),
            32
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 24,
                planPrefillStepSize: 128,
                syncPrefill: true
            ),
            24
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintCapturePrefillStep(
                prefixTokenCount: 0,
                planPrefillStepSize: 0
            ),
            1
        )
    }

    func testNeuralImprintCaptureUsesSyncPrefillOnTightDeviceOrPlan() {
        let tightPhone = DeviceProfile.MemorySnapshot(
            totalPhysicalMB: 7_600,
            availableMB: 1_900,
            footprintMB: 4_200,
            jetsamLimitMB: 6_100,
            jetsamToPhysicalRatio: 0.8
        )
        XCTAssertTrue(
            LLMEngine.neuralImprintCaptureUsesSyncPrefill(
                memorySnapshot: tightPhone,
                planSyncEval: false
            )
        )

        let lowHeadroom = DeviceProfile.MemorySnapshot(
            totalPhysicalMB: 64_000,
            availableMB: 1_200,
            footprintMB: 4_200,
            jetsamLimitMB: 5_400,
            jetsamToPhysicalRatio: 0.08
        )
        XCTAssertTrue(
            LLMEngine.neuralImprintCaptureUsesSyncPrefill(
                memorySnapshot: lowHeadroom,
                planSyncEval: false
            )
        )

        let roomyMac = DeviceProfile.MemorySnapshot(
            totalPhysicalMB: 64_000,
            availableMB: 12_000,
            footprintMB: 4_200,
            jetsamLimitMB: 16_200,
            jetsamToPhysicalRatio: 0.25
        )
        XCTAssertFalse(
            LLMEngine.neuralImprintCaptureUsesSyncPrefill(
                memorySnapshot: roomyMac,
                planSyncEval: false
            )
        )
        XCTAssertTrue(
            LLMEngine.neuralImprintCaptureUsesSyncPrefill(
                memorySnapshot: roomyMac,
                planSyncEval: true
            )
        )
    }

    func testCleanCaptureSessionReuseRequiresBaseCmlxConfig() {
        XCTAssertTrue(
            LLMEngine.canReuseCmlxSessionForCleanCapture(
                dsrPolicyCount: 0,
                hasAttentionCacheQuantization: false,
                frogJumpLayerMask: 0
            )
        )
        XCTAssertFalse(
            LLMEngine.canReuseCmlxSessionForCleanCapture(
                dsrPolicyCount: 1,
                hasAttentionCacheQuantization: false,
                frogJumpLayerMask: 0
            )
        )
        XCTAssertFalse(
            LLMEngine.canReuseCmlxSessionForCleanCapture(
                dsrPolicyCount: 0,
                hasAttentionCacheQuantization: true,
                frogJumpLayerMask: 0
            )
        )
        XCTAssertFalse(
            LLMEngine.canReuseCmlxSessionForCleanCapture(
                dsrPolicyCount: 0,
                hasAttentionCacheQuantization: false,
                frogJumpLayerMask: 1 << 12
            )
        )
    }

    func testNeuralImprintTokenIDsHashMatchesCanonicalJSON() {
        XCTAssertEqual(
            LLMEngine.neuralImprintTokenIDsSHA256([3, 1, 4]),
            "17bd9d6d847aac8a9c60e9a8aee94635e0cd2aa70dca2fa8f6bcea00362e5391"
        )
    }

    func testNeuralImprintCaptureRequestDefaultBackendVersionMatchesPinnedEngine() {
        let request = NeuralImprintArtifactCaptureRequest(
            outputDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("neural-imprint-test", isDirectory: true),
            renderedPrefix: "",
            prefixTokenIDs: [],
            profileBody: "",
            toolSchemaSnapshot: ToolSchemaSnapshot(
                export: ToolSchemaExport(tools: []),
                jsonData: Data("{}".utf8),
                sha256: LLMEngine.sha256Text("{}")
            )
        )

        XCTAssertEqual(request.cacheBackendVersion, "edge-engine 1.0.0-rc138")
    }

    func testNeuralImprintSystemPromptEmbedsProfileBody() {
        let prompt = LLMEngine.neuralImprintSystemPrompt(
            profileBody: "用户偏好：常在周末买咖啡。"
        )

        XCTAssertEqual(prompt, "用户偏好：常在周末买咖啡。")
        XCTAssertFalse(prompt.contains("运行在用户私有设备上的端侧助手"))
        XCTAssertFalse(prompt.contains("本地 RPP 用户画像"))
        XCTAssertFalse(prompt.contains("不要调用工具"))
    }

    func testNeuralImprintPrefixSplitIndexFindsUserMarker() {
        XCTAssertEqual(
            LLMEngine.neuralImprintPrefixSplitIndex(
                fullTokenIDs: [10, 11, 20, 21, 30],
                markerTokenIDs: [20, 21]
            ),
            2
        )
        XCTAssertEqual(
            LLMEngine.neuralImprintPrefixSplitIndex(
                fullTokenIDs: [20, 21, 10],
                markerTokenIDs: [20, 21]
            ),
            0
        )
        XCTAssertNil(
            LLMEngine.neuralImprintPrefixSplitIndex(
                fullTokenIDs: [10, 20, 30],
                markerTokenIDs: [20, 21]
            )
        )
        XCTAssertNil(
            LLMEngine.neuralImprintPrefixSplitIndex(
                fullTokenIDs: [20, 21, 10, 20, 21],
                markerTokenIDs: [20, 21]
            )
        )
        XCTAssertNil(
            LLMEngine.neuralImprintPrefixSplitIndex(
                fullTokenIDs: [10, 20, 30],
                markerTokenIDs: []
            )
        )
    }

    func testNeuralImprintCacheManifestUsesSavedTensorMetadata() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("neural_imprint_manifest_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let artifactURL = directory.appendingPathComponent("neural_imprint.safetensors")
        try Self.writeSafeTensorsFixture(
            to: artifactURL,
            tensors: [
                SafeTensorFixtureTensor(
                    name: "layer_00.state_0",
                    dtype: "BF16",
                    shape: [1, 4, 8]
                ),
                SafeTensorFixtureTensor(
                    name: "layer_00.state_1",
                    dtype: "BF16",
                    shape: [1, 2, 3, 4]
                ),
                SafeTensorFixtureTensor(
                    name: "layer_01.state_0",
                    dtype: "F16",
                    shape: [1, 2, 5, 2]
                ),
                SafeTensorFixtureTensor(
                    name: "layer_01.state_1",
                    dtype: "F16",
                    shape: [1, 2, 5, 2]
                ),
            ]
        )

        let artifact = try SafeTensorsShardFile(url: artifactURL)
        let architecture = try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: 128,
            hiddenSize: 8,
            intermediateSize: 16,
            attentionHeadCount: 4,
            keyValueHeadCount: 2,
            contextLength: 64,
            rmsNormEpsilon: 1e-6,
            ropeTheta: 1_000_000,
            layerKinds: [.gdn, .fullAttention]
        )

        let manifest = try LLMEngine.neuralImprintCacheManifest(
            artifact: artifact,
            architecture: architecture,
            prefixTokenCount: 5
        )
        let layers = try XCTUnwrap(manifest["layers"] as? [[String: Any]])
        XCTAssertEqual(manifest["layer_count"] as? Int, 2)

        XCTAssertEqual(layers[0]["layer"] as? Int, 0)
        XCTAssertEqual(layers[0]["cache_class"] as? String, "ArraysCache")
        XCTAssertEqual(layers[0]["state_container"] as? String, "list")
        XCTAssertTrue(layers[0]["offset"] is NSNull)
        let gdnStates = try XCTUnwrap(layers[0]["states"] as? [[String: Any]])
        XCTAssertEqual(gdnStates[0]["shape"] as? [Int], [1, 4, 8])
        XCTAssertEqual(gdnStates[0]["dtype"] as? String, "mlx.core.bfloat16")

        XCTAssertEqual(layers[1]["layer"] as? Int, 1)
        XCTAssertEqual(layers[1]["cache_class"] as? String, "KVCache")
        XCTAssertEqual(layers[1]["state_container"] as? String, "tuple")
        XCTAssertEqual(layers[1]["offset"] as? Int, 5)
        let attentionStates = try XCTUnwrap(layers[1]["states"] as? [[String: Any]])
        XCTAssertEqual(attentionStates[0]["shape"] as? [Int], [1, 2, 5, 2])
        XCTAssertEqual(attentionStates[0]["dtype"] as? String, "mlx.core.float16")
    }

    func testNeuralImprintArtifactAndMetadataURLsFallbackToLegacyNeuralImprintNames() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("neural_imprint_url_fallback_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let legacyArtifact = directory.appendingPathComponent("persona_kv.safetensors")
        let legacyMetadata = directory.appendingPathComponent("persona_kv_metadata.json")
        try Data("legacy".utf8).write(to: legacyArtifact)
        try Data("legacy".utf8).write(to: legacyMetadata)

        XCTAssertEqual(
            LLMEngine.neuralImprintArtifactURL(
                sidecarArtifactPath: "neural_imprint.safetensors",
                directory: directory
            ),
            legacyArtifact
        )
        XCTAssertEqual(LLMEngine.neuralImprintMetadataURL(directory: directory), legacyMetadata)

        let newArtifact = directory.appendingPathComponent("neural_imprint.safetensors")
        let newMetadata = directory.appendingPathComponent("neural_imprint_metadata.json")
        try Data("new".utf8).write(to: newArtifact)
        try Data("new".utf8).write(to: newMetadata)

        XCTAssertEqual(
            LLMEngine.neuralImprintArtifactURL(
                sidecarArtifactPath: "neural_imprint.safetensors",
                directory: directory
            ),
            newArtifact
        )
        XCTAssertEqual(LLMEngine.neuralImprintMetadataURL(directory: directory), newMetadata)
    }

    func testNeuralImprintModelWeightsFingerprintIncludesIndexAndShards() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("neural_imprint_fingerprint_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let firstShard = directory.appendingPathComponent("a.safetensors")
        let secondShard = directory.appendingPathComponent("b.safetensors")
        try Data(#"{"weight_map":{"model.b":"b.safetensors","model.a":"a.safetensors"}}"#.utf8)
            .write(to: indexURL)
        try Data("aaa".utf8).write(to: firstShard)
        try Data("bbb".utf8).write(to: secondShard)

        let expectedPayload = "{\"index_sha256\":\"\(try LLMEngine.sha256File(indexURL))\",\"shards\":[{\"name\":\"a.safetensors\",\"sha256\":\"\(try LLMEngine.sha256File(firstShard))\",\"size_bytes\":3},{\"name\":\"b.safetensors\",\"sha256\":\"\(try LLMEngine.sha256File(secondShard))\",\"size_bytes\":3}]}"

        let fingerprint = try LLMEngine.neuralImprintModelWeightsFingerprint(
            modelDirectory: directory
        )
        XCTAssertEqual(fingerprint, "sha256:" + LLMEngine.sha256Text(expectedPayload))

        try Data("changed".utf8).write(to: secondShard)
        XCTAssertNotEqual(
            fingerprint,
            try LLMEngine.neuralImprintModelWeightsFingerprint(modelDirectory: directory)
        )
    }

    func testNeuralImprintChatTemplateHashCanMatchTransformersCanonicalTemplate() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("neural_imprint_chat_template_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let rawTemplate = #"""
prefix
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
suffix
"""#
        let canonicalTemplate = #"""
prefix
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
suffix
"""#
        let configData = try JSONSerialization.data(
            withJSONObject: ["chat_template": rawTemplate],
            options: [.sortedKeys]
        )
        try configData.write(to: directory.appendingPathComponent("tokenizer_config.json"))

        let expected = LLMEngine.sha256Text(canonicalTemplate)
        XCTAssertEqual(
            try LLMEngine.neuralImprintChatTemplateSHA256(
                modelDirectory: directory,
                expectedSHA256: expected
            ),
            expected
        )
    }

    func testGenerationConfigEndTokenIdsParsesArrayValue() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("generation_config_array_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = Data(#"{"eos_token_id":[151643,151645]}"#.utf8)
        try data.write(to: directory.appendingPathComponent("generation_config.json"))

        XCTAssertEqual(
            LLMEngine.generationConfigEndTokenIds(modelDirectory: directory),
            [151643, 151645]
        )
    }

    func testGenerationConfigEndTokenIdsParsesSingleValue() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("generation_config_single_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = Data(#"{"eos_token_id":151645}"#.utf8)
        try data.write(to: directory.appendingPathComponent("generation_config.json"))

        XCTAssertEqual(
            LLMEngine.generationConfigEndTokenIds(modelDirectory: directory),
            [151645]
        )
    }

    func testDefaultEndTokenIdsDoesNotTreatPadEndOfTextAsStopToken() {
        let ids = LLMEngine.defaultEndTokenIds(
            eosTokenId: 248046,
            tokenIdForToken: { token in
                switch token {
                case "<|im_end|>":
                    return 248046
                case "<|endoftext|>":
                    return 248044
                default:
                    return nil
                }
            }
        )

        XCTAssertEqual(ids, [248046])
        XCTAssertFalse(ids.contains(248044))
    }

    func testDefaultEndTokenIdsIncludesGenerationConfigTokensWhenExplicit() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("default_end_tokens_explicit_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let data = Data(#"{"eos_token_id":[248044,248046]}"#.utf8)
        try data.write(to: directory.appendingPathComponent("generation_config.json"))

        let ids = LLMEngine.defaultEndTokenIds(
            eosTokenId: 248046,
            tokenIdForToken: { token in
                token == "<|im_end|>" ? 248046 : nil
            },
            modelDirectory: directory
        )

        XCTAssertEqual(ids, [248044, 248046])
    }

    func testPruningMetadataOnStandardModel() throws {
        let modelURL = try requireBaselineModel(requireGate: false)
        let meta = try PruningMetadata.load(from: modelURL)
        XCTAssertFalse(meta.isEdgeOptimized, "Standard model should not be Edge Studio optimized")
        print("[PruningMetadata]", meta)
    }

    @MainActor
    func testRestoreNeuralImprintCacheValidatesRealArtifactWhenEnabled() async throws {
        let fixture = try requireRealNeuralImprintFixture()
        let sidecarURL = LLMEngine.neuralImprintMetadataURL(directory: fixture.artifactDirectory)
        let sidecar = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: sidecarURL)
        ) as? [String: Any]
        let prefix = sidecar?["prefix"] as? [String: Any]
        guard let expectedPrefixTokens = prefix?["token_count"] as? Int,
              let expectedArtifactSHA256 = sidecar?["artifact_sha256"] as? String else {
            XCTFail("Neural Imprint sidecar is missing prefix.token_count or artifact_sha256")
            return
        }

        let engine = LLMEngine()
        try await engine.loadLocal(
            directory: fixture.model,
            options: NativeRuntimeLoadOptions(
                cmlxLazyDecodeEnabled: true,
                greedyOutputHeadArgmaxEnabled: true
            )
        )
        defer { engine.unload() }

        let status = try engine.restoreNeuralImprintCache(from: fixture.artifactDirectory)

        XCTAssertEqual(status.prefixTokenCount, expectedPrefixTokens)
        XCTAssertEqual(status.artifactSHA256, expectedArtifactSHA256)
        XCTAssertEqual(engine.activeNeuralImprintCache, status)
    }

    @MainActor
    func testCaptureNeuralImprintArtifactWithSyntheticPrefixWhenEnabled() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment[Self.neuralImprintCaptureGate] == "1" else {
            throw XCTSkip("set \(Self.neuralImprintCaptureGate)=1 to run real Neural Imprint capture smoke")
        }
        let modelURL = try requireBaselineModel(requireGate: false)

        let outputDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("neural_imprint_capture_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: outputDirectory) }

        let engine = LLMEngine()
        var diagnostics: [String] = []
        engine.diagnosticSink = { message in
            diagnostics.append(message)
            print("[NeuralImprintCapture]", message)
        }
        try await engine.loadLocal(
            directory: modelURL,
            options: NativeRuntimeLoadOptions(
                cmlxLazyDecodeEnabled: true,
                greedyOutputHeadArgmaxEnabled: true
            )
        )
        defer { engine.unload() }

        let export = ToolSchemaExport(tools: [])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let toolSchemaData = try encoder.encode(export)
        let toolSchemaString = try XCTUnwrap(String(data: toolSchemaData, encoding: .utf8))
        let toolSchemaSnapshot = ToolSchemaSnapshot(
            export: export,
            jsonData: toolSchemaData,
            sha256: LLMEngine.sha256Text(toolSchemaString)
        )
        let tokenIDs = Array(repeating: 1, count: 1_018)
        let status = try await engine.captureNeuralImprintArtifact(
            request: NeuralImprintArtifactCaptureRequest(
                outputDirectory: outputDirectory,
                renderedPrefix: "synthetic capture prefix",
                prefixTokenIDs: tokenIDs,
                profileBody: "synthetic profile",
                toolSchemaSnapshot: toolSchemaSnapshot,
                modelID: "synthetic-qwen3.5-4b-6bit",
                systemPrompt: "synthetic system",
                createdBy: "edge-kit-tests",
                writerVersion: "edge-kit.neural_imprint_capture_smoke.v1"
            )
        )

        XCTAssertEqual(status.prefixTokenCount, tokenIDs.count)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: outputDirectory.appendingPathComponent("neural_imprint.safetensors").path
            )
        )
        XCTAssertTrue(
            diagnostics.contains {
                $0.contains("neural_imprint_capture_prefill_begin total=1018 step=128")
            }
        )
    }

    @MainActor
    func testLoadLocalBaselineModel() async throws {
        let modelURL = try requireBaselineModel()
        let engine = LLMEngine()

        do {
            try await engine.loadLocal(directory: modelURL)
        } catch {
            throw XCTSkip("local Qwen baseline is not an edge-engine native bundle: \(error)")
        }
        XCTAssertEqual(engine.state, .ready)
        print("[Engine] loaded, pruning:", engine.pruningMetadata as Any)

        engine.unload()
        XCTAssertEqual(engine.state, .idle)
    }

    @MainActor
    func testGenerateOnceWithBaselineModel() async throws {
        let modelURL = try requireBaselineModel()
        let engine = LLMEngine()
        try await engine.loadLocal(directory: modelURL)
        defer { engine.unload() }

        var output = ""
        let stream = engine.generate(
            messages: [
                .system("你是 AtomGradient EdgeStudio 的本地推理助手。EdgeStudio 是 AtomGradient 面向 Apple 生态的端侧 AI 开发与运行时套件，不是 Apple 官方产品，也不是微软 Azure 产品。回答要简洁。"),
                .user("用一句中文介绍EdgeStudio。"),
            ],
            parameters: EdgeGenerateParameters(
                temperature: 0,
                topP: 1,
                maxTokens: 32,
                useDSR: false
            ),
            bypassPolicy: true
        )
        for try await chunk in stream {
            output += chunk.text
        }

        XCTAssertFalse(output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertTrue(output.localizedCaseInsensitiveContains("AtomGradient"))
        XCTAssertTrue(output.localizedCaseInsensitiveContains("Apple"))
        XCTAssertFalse(output.localizedCaseInsensitiveContains("Apple 推出"))
        XCTAssertFalse(output.localizedCaseInsensitiveContains("Apple推出"))
        XCTAssertFalse(output.localizedCaseInsensitiveContains("Azure"))
        XCTAssertNotNil(engine.lastMetrics)
        print("[Engine] output:", output)
        print("[Engine] metrics:", engine.lastMetrics as Any)
    }

    @MainActor
    func testGenerateOnceWithBaselineModelUsingCmlxLazyDecode() async throws {
        let modelURL = try requireBaselineModel()
        let engine = LLMEngine()
        var diagnostics: [String] = []
        engine.diagnosticSink = { diagnostics.append($0) }
        try await engine.loadLocal(
            directory: modelURL,
            options: NativeRuntimeLoadOptions(
                cmlxLazyDecodeEnabled: true,
                greedyOutputHeadArgmaxEnabled: true
            )
        )
        defer { engine.unload() }

        var output = ""
        let stream = engine.generate(
            messages: [
                .system("你是 AtomGradient EdgeStudio 的本地推理助手。回答要简洁。"),
                .user("用一句中文介绍EdgeStudio。"),
            ],
            parameters: EdgeGenerateParameters(
                temperature: 0,
                topP: 1,
                maxTokens: 16,
                useDSR: false
            ),
            bypassPolicy: true
        )
        for try await chunk in stream {
            output += chunk.text
        }

        XCTAssertFalse(output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertTrue(diagnostics.contains("cmlx_lazy_load_local deferred_swift_model=true"))
        XCTAssertTrue(diagnostics.contains { $0.hasPrefix("cmlx_lazy_session_init_done") })
        XCTAssertTrue(engine.lastMetrics?.policyReasoning.contains("cmlxLazyDecode=on") ?? false)

        engine.clearPromptCache()
        XCTAssertTrue(
            diagnostics.contains { $0 == "cmlx_lazy_session_release reason=clear_prompt_cache" },
            "clearPromptCache must release the CMLX lazy decode session, otherwise isolated smoke prompts inherit stale KV state"
        )
        XCTAssertTrue(
            diagnostics.contains { $0 == "cmlx_lazy_session_reset_done reason=clear_prompt_cache" },
            "clearPromptCache must reset the CMLX lazy decode session before release"
        )
        print("[Engine CMLX] output:", output)
        print("[Engine CMLX] metrics:", engine.lastMetrics as Any)
    }

    private struct SafeTensorFixtureTensor {
        var name: String
        var dtype: String
        var shape: [Int]

        var byteCount: Int {
            let elementCount = shape.reduce(1, *)
            switch dtype {
            case "F32":
                return elementCount * 4
            default:
                return elementCount * 2
            }
        }
    }

    private static func writeSafeTensorsFixture(
        to url: URL,
        tensors: [SafeTensorFixtureTensor]
    ) throws {
        var offset = 0
        var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for tensor in tensors {
            let end = offset + tensor.byteCount
            header[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [offset, end],
            ]
            offset = end
        }

        let headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys, .withoutEscapingSlashes]
        )
        var data = Data()
        var headerLength = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
        data.append(headerData)
        data.append(Data(repeating: 0, count: offset))
        try data.write(to: url)
    }
}

var stdout = StandardOutputStream()
struct StandardOutputStream: TextOutputStream {
    mutating func write(_ string: String) { Swift.print(string, terminator: "") }
}
