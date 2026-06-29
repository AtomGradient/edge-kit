// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import CoreImage
import EdgeEngine
import Foundation
import Tokenizers

public struct NeuralImprintArtifactCaptureRequest: Sendable {
    public var outputDirectory: URL
    public var renderedPrefix: String
    public var prefixTokenIDs: [Int]
    public var profileBody: String
    public var toolSchemaSnapshot: ToolSchemaSnapshot
    public var modelID: String?
    public var systemPrompt: String
    public var enableThinking: Bool
    public var profileBodyFileName: String
    public var toolSpecsFileName: String
    public var createdAt: Date
    public var createdBy: String
    public var writerVersion: String
    public var minReaderVersion: String
    public var cacheBackendVersion: String

    public init(
        outputDirectory: URL,
        renderedPrefix: String,
        prefixTokenIDs: [Int],
        profileBody: String,
        toolSchemaSnapshot: ToolSchemaSnapshot,
        modelID: String? = nil,
        systemPrompt: String = "",
        enableThinking: Bool = false,
        profileBodyFileName: String = "profile_body.txt",
        toolSpecsFileName: String = "tool_specs.json",
        createdAt: Date = Date(),
        createdBy: String = "edge-kit",
        writerVersion: String = "edge-kit.neural_imprint_capture.v1",
        minReaderVersion: String = "edge-kit.neural_imprint_reader.v1",
        cacheBackendVersion: String = "edge-engine 1.0.0-rc138"
    ) {
        self.outputDirectory = outputDirectory
        self.renderedPrefix = renderedPrefix
        self.prefixTokenIDs = prefixTokenIDs
        self.profileBody = profileBody
        self.toolSchemaSnapshot = toolSchemaSnapshot
        self.modelID = modelID
        self.systemPrompt = systemPrompt
        self.enableThinking = enableThinking
        self.profileBodyFileName = profileBodyFileName
        self.toolSpecsFileName = toolSpecsFileName
        self.createdAt = createdAt
        self.createdBy = createdBy
        self.writerVersion = writerVersion
        self.minReaderVersion = minReaderVersion
        self.cacheBackendVersion = cacheBackendVersion
    }
}

public struct NeuralImprintPrefixRender: Sendable, Equatable {
    public var systemPrompt: String
    public var renderedPrefix: String
    public var prefixTokenIDs: [Int]
    public var enableThinking: Bool

    public init(
        systemPrompt: String,
        renderedPrefix: String,
        prefixTokenIDs: [Int],
        enableThinking: Bool
    ) {
        self.systemPrompt = systemPrompt
        self.renderedPrefix = renderedPrefix
        self.prefixTokenIDs = prefixTokenIDs
        self.enableThinking = enableThinking
    }
}

public final class LLMEngine: ObservableObject {
    public static let neuralImprintArtifactFileName = NeuralImprintRuntimeSupport.artifactFileName
    public static let legacyPersonaKVArtifactFileName = NeuralImprintRuntimeSupport.legacyArtifactFileName
    public static let neuralImprintMetadataFileName = NeuralImprintRuntimeSupport.metadataFileName
    public static let legacyPersonaKVMetadataFileName = NeuralImprintRuntimeSupport.legacyMetadataFileName
    public static let neuralImprintPrefixSplitSentinel = NeuralImprintRuntimeSupport.prefixSplitSentinel
    static let neuralImprintPrefixUserMarker = NeuralImprintRuntimeSupport.prefixUserMarker

    @Published public private(set) var state: EngineState = .idle
    @Published public private(set) var loadedConfig: ModelConfig?
    @Published public private(set) var pruningMetadata: PruningMetadata?
    @Published public private(set) var downloadProgress: Double = 0
    @Published public private(set) var lastPolicy: InferencePolicy.Resolved?
    @Published public private(set) var lastMetrics: InferenceMetrics?

    public private(set) var memoryPolicy: KVCacheMemoryPolicy?
    public private(set) var archInfo: ModelArchInfo?
    public private(set) var currentPlan: MemoryBudgetPlanner.Plan?
    public let promptCache = PromptCacheManager()
    public private(set) var activeNeuralImprintCache: NeuralImprintCacheStatus?

    public typealias NeuralImprintCacheStatus = NeuralImprintRuntimeCacheStatus

    private var modelDirectory: URL?
    private var nativeRuntime: EdgeMetalRuntime?
    private var nativeExecutor: MetalKernelExecutor?
    private var nativeModel: QwenHybridModelReference?
    private var nativeArchitecture: QwenHybridArchitecture?
    private var nativeBundleIndex: QwenModelBundleIndex?
    private var nativeTokenizer: Tokenizer?
    private var nativeEndTokenIds: Set<Int> = []
    private var nativeDecodeSession: QwenGreedyDecodeSession?
    private var nativeCmlxLazyDecodeSession: QwenCmlxLazyDecodeSession?
    private var nativeCmlxLazyDecodeSessionDSRPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    private var nativeCmlxLazyDecodeSessionAttentionCacheQuantization: NativeCmlxAttentionCacheQuantization?
    private var nativeCmlxLazyDecodeSessionFrogJumpLayerMask: UInt64 = 0
    private var nativeCmlxLimitState = NativeCmlxCommandBufferLimitState()
    private var onlineCalibrator: OnlineCalibrator?
    private var activeOnlineCalibrationOverrides: OnlineCalibrator.CalibrationOverrides?
    private var nativeUseCmlxLazyDecode = false
    private var nativeUseCmlxLazyTextSuffixPromptCache = true
    private var nativeDecodeSessionTokenIds: [Int] = []
    private var nativeDecodeSessionCapacity: Int = 0
    private var nativeDecodeSessionDSRPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    private var nativeDecodeSessionPromptMessages: [ChatMessage] = []
    private var nativeDecodeSessionLastAssistantText: String = ""
    private var turnCounter = 0
    private var cachedModelIdentityHashes: ModelIdentityHashes?
    public var diagnosticSink: ((String) -> Void)?

    private struct ModelIdentityHashes {
        let modelDirectoryPath: String
        let modelArchitectureID: String
        let modelConfigSHA256: String
        let modelWeightsFingerprint: String
        let tokenizerJSONSHA256: String
        let tokenizerConfigSHA256: String
        let chatTemplateSHA256: String
    }

    public init() {}

    nonisolated static func chatTemplateContext(
        parameters: EdgeGenerateParameters
    ) -> [String: any Sendable] {
        [
            "enable_thinking": parameters.enableThinking,
            "preserve_thinking": parameters.preserveThinking,
        ]
    }

    private func emitDecodedTextIfNeeded(
        tokenizer: Tokenizer,
        generatedTokenIds: [Int],
        emittedText: inout String,
        emittedTokenCount: inout Int,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation,
        force: Bool = false
    ) {
        guard !generatedTokenIds.isEmpty else { return }

        let decodedText: String
        let delta: String
        if force {
            decodedText = tokenizer.decode(
                tokens: generatedTokenIds,
                skipSpecialTokens: true
            )
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                diagnosticSink?(
                    "decode_final_text_mismatch streamedChars=\(emittedText.count) finalChars=\(decodedText.count)"
                )
                emittedTokenCount = generatedTokenIds.count
                return
            }
        } else {
            guard emittedTokenCount < generatedTokenIds.count,
                  let lastTokenId = generatedTokenIds.last else {
                return
            }
            let decodeWindow = 32
            let previousCount = generatedTokenIds.count - 1
            let previousStart = max(0, previousCount - decodeWindow)
            let currentStart = max(0, generatedTokenIds.count - decodeWindow)
            let previousTail = previousCount > 0
                ? Array(generatedTokenIds[previousStart..<previousCount])
                : []
            let currentTail = Array(generatedTokenIds[currentStart..<generatedTokenIds.count])
            let previousTailText = previousTail.isEmpty
                ? ""
                : tokenizer.decode(tokens: previousTail, skipSpecialTokens: true)
            let currentTailText = tokenizer.decode(
                tokens: currentTail,
                skipSpecialTokens: true
            )
            if currentTailText.hasPrefix(previousTailText) {
                let start = currentTailText.index(
                    currentTailText.startIndex,
                    offsetBy: previousTailText.count
                )
                delta = String(currentTailText[start...])
            } else {
                delta = tokenizer.decode(tokens: [lastTokenId], skipSpecialTokens: true)
            }
            decodedText = emittedText + delta
        }
        emittedText = decodedText
        emittedTokenCount = generatedTokenIds.count
        if !delta.isEmpty {
            continuation.yield(
                GenerateChunk(
                    text: delta,
                    generatedTokenCount: generatedTokenIds.count
                )
            )
        }
    }

    private func emitFinalDecodedTextHandlingToolCalls(
        tokenizer: Tokenizer,
        generatedTokenIds: [Int],
        emittedText: inout String,
        emittedTokenCount: inout Int,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?
    ) async throws {
        guard !generatedTokenIds.isEmpty else { return }

        let decodedText = tokenizer.decode(
            tokens: generatedTokenIds,
            skipSpecialTokens: true
        )
        emittedText = decodedText
        emittedTokenCount = generatedTokenIds.count

        guard let onToolCall else {
            if !decodedText.isEmpty {
                continuation.yield(
                    GenerateChunk(
                        text: decodedText,
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
            }
            return
        }

        let toolCalls = ToolCallTextParser.toolCalls(in: decodedText)
        guard !toolCalls.isEmpty else {
            if !decodedText.isEmpty {
                continuation.yield(
                    GenerateChunk(
                        text: decodedText,
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
            }
            return
        }

        for toolCall in toolCalls {
            let result = try await onToolCall(toolCall)
            if !result.isEmpty {
                continuation.yield(GenerateChunk(text: result))
            }
        }
    }

    public func load(config: ModelConfig, onProgress: ((Double) -> Void)? = nil) async throws {
        loadedConfig = config
        onProgress?(0)
        throw EdgeRuntimeError.unsupportedFeature(
            "Remote model download is not part of the native default build; use ODR or loadLocal(directory:)."
        )
    }

    public func loadLocal(directory: URL, onProgress: ((Double) -> Void)? = nil) async throws {
        try await loadLocal(directory: directory, options: nil, onProgress: onProgress)
    }

    public func loadLocal(
        directory: URL,
        options: NativeRuntimeLoadOptions?,
        onProgress: ((Double) -> Void)? = nil
    ) async throws {
        guard state != .loading else { return }
        state = .loading
        downloadProgress = 0
        onProgress?(0)
        do {
            _ = try QwenBundlePreflightRunner.run(
                configuration: QwenBundlePreflightConfiguration(modelRootURL: directory)
            )
            modelDirectory = directory
            pruningMetadata = try? PruningMetadata.load(from: directory)
            archInfo = ModelArchInfo.load(from: directory)
            let benchmark = DeviceBenchmark.cachedOrCurrent()
            let profile = benchmark.profile
            let modelSizeGB = Self.estimateModelSizeGB(directory: directory)
            var plan = MemoryBudgetPlanner.plan(
                profile: profile,
                modelSizeGB: modelSizeGB,
                measuredBandwidthGBs: benchmark.measuredBandwidthGBs,
                intent: options?.memoryIntent ?? .balanced
            ).applying(options)
            onlineCalibrator = nil
            activeOnlineCalibrationOverrides = nil
            if options?.onlineCalibrationEnabled == true {
                let identity = OnlineCalibrator.storageIdentity(
                    deviceModel: profile.machineIdentifier,
                    modelIdentifier: directory.lastPathComponent,
                    quantization: String(format: "size%.1f", modelSizeGB)
                )
                let initial = OnlineCalibrator.CalibrationOverrides(
                    maxOpsPerBuffer: plan.maxOpsPerBuffer,
                    prefillStepSize: plan.prefillStepSize,
                    dynamicOpsFloor: plan.dynamicOpsFloor
                )
                let calibrator = OnlineCalibrator(storageIdentity: identity, initialOverrides: initial)
                onlineCalibrator = calibrator
                plan = plan.applying(calibrator.currentOverrides.nativeLoadOptions)
                activeOnlineCalibrationOverrides = calibrator.currentOverrides
                diagnosticSink?(
                    "online_calibration_enabled key=\(identity) maxOps=\(calibrator.currentOverrides.maxOpsPerBuffer) prefill=\(calibrator.currentOverrides.prefillStepSize) dynFloor=\(calibrator.currentOverrides.dynamicOpsFloor)"
                )
            }
            currentPlan = plan
            memoryPolicy = MemoryBudgetPlanner.toKVPolicy(plan)
            var metalConfiguration = NativeRuntimeBridge.metalConfiguration(for: plan)
            if let noCopy = options?.quantizedNoCopyBuffersEnabled {
                metalConfiguration.quantizedNoCopyBuffersEnabled = noCopy
            }
            if let qmm = options?.vendoredQuantizedMatmulEnabled {
                metalConfiguration.useMLXQuantizedMatmul = qmm
            }
            if let prefillQMM = options?.vendoredQuantizedPrefillMatmulEnabled {
                metalConfiguration.useMLXQuantizedPrefillMatmul = prefillQMM
            }
            if let commandBufferPrefillQMM = options?.vendoredCommandBufferPrefillQMMEnabled {
                metalConfiguration.useVendoredCommandBufferPrefillQMM = commandBufferPrefillQMM
            }
            if let singleCBPrefill = options?.singleCommandBufferPrefillEnabled {
                metalConfiguration.useSingleCommandBufferPrefill = singleCBPrefill
            }
            if let singleCBDecode = options?.singleCommandBufferDecodeEnabled {
                metalConfiguration.useSingleCommandBufferDecode = singleCBDecode
            }
            if let prefillLayerCB = options?.prefillLayerCommandBufferBatchingEnabled {
                metalConfiguration.usePrefillLayerCommandBufferBatching = prefillLayerCB
            }
            if let fusedGDNDecode = options?.fusedGDNDecodeEnabled {
                metalConfiguration.useFusedGDNDecode = fusedGDNDecode
            }
            if let cmlxFastRMSNorm = options?.cmlxFastRMSNormEnabled {
                metalConfiguration.useCmlxFastRMSNorm = cmlxFastRMSNorm
            }
            if let cmlxLazyOutputHead = options?.cmlxLazyOutputHeadEnabled {
                metalConfiguration.useCmlxLazyOutputHead = cmlxLazyOutputHead
            }
            nativeUseCmlxLazyDecode = options?.cmlxLazyDecodeEnabled
                ?? Self.shouldEnableCmlxLazyDecodeByDefault(plan: plan)
            nativeUseCmlxLazyTextSuffixPromptCache = NativeEnvironment.bool(
                ["EDGE_CMLX_TEXT_SUFFIX_PROMPT_CACHE", "EDGE_TEXT_SUFFIX_PROMPT_CACHE"],
                defaultValue: true
            )
            diagnosticSink?(
                "prompt_cache_text_suffix_config cmlxEnabled=\(nativeUseCmlxLazyTextSuffixPromptCache ? 1 : 0)"
            )
            if let greedyOutputHeadArgmax = options?.greedyOutputHeadArgmaxEnabled {
                metalConfiguration.useGreedyOutputHeadArgmax = greedyOutputHeadArgmax
            }
            if let maxInFlight = options?.maxInFlightCommandBuffers {
                metalConfiguration.maxInFlightCommandBuffers = maxInFlight
            }
            _ = NativeRuntimeBridge.applyMetalConfiguration(metalConfiguration)
            let runtime = try NativeRuntimeBridge.makeNativeMetalRuntime()
            let bundleIndex = try QwenModelBundleIndex.load(from: directory)
            let tokenizer = try await AutoTokenizer.from(
                modelFolder: directory,
                strict: false
            )
            nativeRuntime = runtime
            nativeArchitecture = bundleIndex.architecture
            nativeBundleIndex = bundleIndex
            nativeTokenizer = tokenizer
            nativeEndTokenIds = Self.defaultEndTokenIds(
                tokenizer: tokenizer,
                modelDirectory: directory
            )
            diagnosticSink?("model_identity_cache_begin")
            cachedModelIdentityHashes = try Self.computeModelIdentityHashes(modelDirectory: directory)
            diagnosticSink?("model_identity_cache_done")
            if nativeUseCmlxLazyDecode {
                nativeExecutor = nil
                nativeModel = nil
                diagnosticSink?("cmlx_lazy_load_local deferred_swift_model=true")
            } else {
                let loaded = try loadNativeModelFallback(runtime: runtime, bundleIndex: bundleIndex)
                nativeExecutor = loaded.executor
                nativeModel = loaded.model
            }
            clearNativeGreedyDecodeSession()
            downloadProgress = 1
            onProgress?(1)
            state = .ready
        } catch {
            unload()
            throw EdgeRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func generate(
        messages: [ChatMessage],
        tools: [ToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters = .default,
        bypassPolicy: Bool = false
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try await self.runNativeGenerate(
                        messages: messages,
                        tools: tools,
                        onToolCall: onToolCall,
                        parameters: parameters,
                        bypassPolicy: bypassPolicy,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public func generateStream(
        messages: [ChatMessage],
        parameters: EdgeGenerateParameters = .default
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        generate(messages: messages, parameters: parameters)
    }

    public func generateOnce(
        messages: [ChatMessage],
        parameters: EdgeGenerateParameters = .default
    ) async throws -> String {
        var output = ""
        for try await chunk in generate(messages: messages, parameters: parameters) {
            output += chunk.text
        }
        return output
    }

    public func tokenize(_ text: String) async throws -> [Int] {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native Qwen tokenizer is not initialized")
        }
        return tokenizer.encode(text: text)
    }

    public func renderNeuralImprintPrefix(
        profileBody: String,
        tools: [ToolSpec] = [],
        parameters requestedParameters: EdgeGenerateParameters = .default
    ) async throws -> NeuralImprintPrefixRender {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native Qwen tokenizer is not initialized")
        }
        return try NeuralImprintRuntimeSupport.renderPrefix(
            profileBody: profileBody,
            tools: tools,
            parameters: requestedParameters,
            tokenizer: tokenizer,
            additionalContext: Self.chatTemplateContext(parameters:)
        )
    }

    public func captureHiddenStates(
        tokens: [Int],
        targetLayer: Int
    ) async throws -> [Float] {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        guard !tokens.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard let runtime = nativeRuntime,
              let architecture = nativeArchitecture,
              let bundleIndex = nativeBundleIndex
        else {
            throw EdgeRuntimeError.loadFailed("Native Qwen runtime is not initialized")
        }
        guard targetLayer >= 0, targetLayer < architecture.layerCount else {
            throw QwenHybridModelReferenceError.invalidCaptureLayer(targetLayer)
        }

        if nativeUseCmlxLazyDecode {
            diagnosticSink?("cmlx_capture_session_prepare_begin")
            try applyCmlxCommandBufferLimits(contextLengthHint: tokens.count)
            let session = try prepareCmlxSessionForHiddenStateCapture(
                bundleIndex: bundleIndex,
                runtime: runtime
            )
            defer {
                do {
                    try session.reset()
                    diagnosticSink?("cmlx_capture_session_reset_done")
                } catch {
                    diagnosticSink?("cmlx_capture_session_reset_failed error=\(error)")
                }
            }
            let captured = try session.captureLastHidden(
                tokenIDs: tokens,
                targetLayer: targetLayer
            )
            diagnosticSink?(
                "cmlx_capture_hidden_done tokens=\(tokens.count) layer=\(targetLayer) hidden=\(captured.count)"
            )
            return captured
        }

        let loaded = try loadNativeModelFallback(
            runtime: runtime,
            bundleIndex: bundleIndex
        )
        let caches = try QwenHybridDecoderCaches(
            architecture: architecture,
            runtime: runtime,
            kvCapacity: max(tokens.count, 1)
        )
        let hidden = try loaded.model.lastTokenHiddenState(
            tokenIds: tokens,
            targetLayer: targetLayer,
            caches: caches,
            executor: loaded.executor,
            diagnosticSink: diagnosticSink
        )
        let captured = try hidden.readFloat32()
        diagnosticSink?(
            "native_capture_hidden_done tokens=\(tokens.count) layer=\(targetLayer) hidden=\(captured.count)"
        )
        return captured
    }

    /// Clears prompt reuse state for both native decode backends.
    ///
    /// Lifecycle vocabulary used by this engine:
    /// - reset: call the backend reset API on an existing session.
    /// - nil/drop: remove Swift references and sidecar bookkeeping.
    /// - release: reset first, then nil/drop the session and bookkeeping.
    ///
    /// `clearNativeGreedyDecodeSession()` only covers the older greedy decode
    /// session. CMLX lazy decode must be released separately because it owns a
    /// resident backend session and KV cache state.
    public func clearPromptCache(resetTurnCounter: Bool = false) {
        promptCache.clear()
        releaseNativeCmlxLazySession(reason: "clear_prompt_cache")
        clearNativeGreedyDecodeSession()
        if resetTurnCounter {
            turnCounter = 0
            lastPolicy = nil
            lastMetrics = nil
        }
    }

    @discardableResult
    public func restoreNeuralImprintCache(from directory: URL) throws -> NeuralImprintCacheStatus {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        guard let modelDirectory, let architecture = nativeArchitecture else {
            throw EdgeRuntimeError.loadFailed("Native Qwen runtime is not initialized")
        }

        let status = try NeuralImprintRuntimeSupport.loadCacheStatus(
            directory: directory,
            modelDirectory: modelDirectory,
            architecture: architecture
        )
        activeNeuralImprintCache = status
        nativeUseCmlxLazyDecode = true
        releaseNativeCmlxLazySession(reason: "neural_imprint_restore_configured")
        nativeExecutor = nil
        nativeModel = nil
        clearNativeGreedyDecodeSession()
        promptCache.clear()
        diagnosticSink?(
            "neural_imprint_restore_configured prefix=\(status.prefixTokenCount) artifactSHA256=\(status.artifactSHA256)"
        )
        return status
    }

    public func unloadNeuralImprintCache() {
        activeNeuralImprintCache = nil
        releaseNativeCmlxLazySession(reason: "neural_imprint_unloaded")
        clearNativeGreedyDecodeSession()
        promptCache.clear()
    }

    static func neuralImprintCapturePrefillStep(
        prefixTokenCount: Int,
        planPrefillStepSize: Int?,
        syncPrefill: Bool = false
    ) -> Int {
        NeuralImprintRuntimeSupport.neuralImprintCapturePrefillStep(
            prefixTokenCount: prefixTokenCount,
            planPrefillStepSize: planPrefillStepSize,
            syncPrefill: syncPrefill
        )
    }

    static func neuralImprintCaptureUsesSyncPrefill(
        memorySnapshot: DeviceProfile.MemorySnapshot,
        planSyncEval: Bool?
    ) -> Bool {
        NeuralImprintRuntimeSupport.neuralImprintCaptureUsesSyncPrefill(
            memorySnapshot: memorySnapshot,
            planSyncEval: planSyncEval
        )
    }

    static func canReuseCmlxSessionForCleanCapture(
        dsrPolicyCount: Int,
        hasAttentionCacheQuantization: Bool,
        frogJumpLayerMask: UInt64
    ) -> Bool {
        NeuralImprintRuntimeSupport.canReuseCmlxSessionForCleanCapture(
            dsrPolicyCount: dsrPolicyCount,
            hasAttentionCacheQuantization: hasAttentionCacheQuantization,
            frogJumpLayerMask: frogJumpLayerMask
        )
    }

    private func prepareCmlxSessionForNeuralImprintCapture(
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime
    ) throws -> QwenCmlxLazyDecodeSession {
        let canReuseResidentSession = Self.canReuseCmlxSessionForCleanCapture(
            dsrPolicyCount: nativeCmlxLazyDecodeSessionDSRPolicies.count,
            hasAttentionCacheQuantization: nativeCmlxLazyDecodeSessionAttentionCacheQuantization != nil,
            frogJumpLayerMask: nativeCmlxLazyDecodeSessionFrogJumpLayerMask
        )

        if let existing = nativeCmlxLazyDecodeSession, canReuseResidentSession {
            try existing.reset()
            diagnosticSink?("neural_imprint_capture_session_reuse")
            return existing
        }

        if nativeCmlxLazyDecodeSession != nil {
            releaseNativeCmlxLazySession(reason: "neural_imprint_capture_requires_clean_session")
        }

        let created = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )
        nativeCmlxLazyDecodeSession = created
        nativeCmlxLazyDecodeSessionDSRPolicies = [:]
        nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
        nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
        diagnosticSink?(
            "neural_imprint_capture_session_init_done floats=\(created.registeredFloatTensorCount) quantized=\(created.registeredQuantizedTensorCount)"
        )
        return created
    }

    private func prepareCmlxSessionForHiddenStateCapture(
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime
    ) throws -> QwenCmlxLazyDecodeSession {
        let canReuseResidentSession = Self.canReuseCmlxSessionForCleanCapture(
            dsrPolicyCount: nativeCmlxLazyDecodeSessionDSRPolicies.count,
            hasAttentionCacheQuantization: nativeCmlxLazyDecodeSessionAttentionCacheQuantization != nil,
            frogJumpLayerMask: nativeCmlxLazyDecodeSessionFrogJumpLayerMask
        )

        if let existing = nativeCmlxLazyDecodeSession, canReuseResidentSession {
            try existing.reset()
            diagnosticSink?("cmlx_capture_session_reuse")
            return existing
        }

        if nativeCmlxLazyDecodeSession != nil {
            releaseNativeCmlxLazySession(reason: "hidden_state_capture_requires_clean_session")
        }

        let created = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )
        nativeCmlxLazyDecodeSession = created
        nativeCmlxLazyDecodeSessionDSRPolicies = [:]
        nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
        nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
        diagnosticSink?(
            "cmlx_capture_session_init_done floats=\(created.registeredFloatTensorCount) quantized=\(created.registeredQuantizedTensorCount)"
        )
        return created
    }

    private func prefillNeuralImprintCapture(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        chunkSize: Int,
        syncPrefill: Bool
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

    @discardableResult
    public func captureNeuralImprintArtifact(
        request: NeuralImprintArtifactCaptureRequest
    ) async throws -> NeuralImprintCacheStatus {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        guard !request.prefixTokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard let modelDirectory,
              let runtime = nativeRuntime,
              let architecture = nativeArchitecture,
              let bundleIndex = nativeBundleIndex
        else {
            throw EdgeRuntimeError.loadFailed("Native Qwen runtime is not initialized")
        }

        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: request.outputDirectory,
            withIntermediateDirectories: true
        )
        let artifactURL = request.outputDirectory.appendingPathComponent(Self.neuralImprintArtifactFileName)
        let metadataURL = request.outputDirectory.appendingPathComponent(Self.neuralImprintMetadataFileName)
        let profileBodyURL = request.outputDirectory.appendingPathComponent(request.profileBodyFileName)
        let toolSpecsURL = request.outputDirectory.appendingPathComponent(request.toolSpecsFileName)
        diagnosticSink?("neural_imprint_capture_source_write_begin")
        try Data(request.profileBody.utf8).write(to: profileBodyURL, options: [.atomic])
        try request.toolSchemaSnapshot.jsonData.write(to: toolSpecsURL, options: [.atomic])
        diagnosticSink?("neural_imprint_capture_source_write_done")

        let prefixTokenCount = request.prefixTokenIDs.count
        let profileBodySHA256 = Self.sha256Text(request.profileBody)
        let modelID = request.modelID ?? loadedConfig?.modelID ?? modelDirectory.lastPathComponent
        diagnosticSink?("neural_imprint_capture_model_identity_begin")
        let modelIdentityHashes = try modelIdentityHashes(for: modelDirectory)
        diagnosticSink?("neural_imprint_capture_model_identity_done")
        let renderedPrefixSHA256 = Self.sha256Text(request.renderedPrefix)
        let prefixTokenIDsSHA256 = Self.neuralImprintTokenIDsSHA256(request.prefixTokenIDs)
        diagnosticSink?("neural_imprint_capture_header_begin")
        let header = try Self.neuralImprintArtifactHeader(
            modelID: modelID,
            modelDirectory: modelDirectory,
            modelArchitectureID: modelIdentityHashes.modelArchitectureID,
            modelConfigSHA256: modelIdentityHashes.modelConfigSHA256,
            modelWeightsFingerprint: modelIdentityHashes.modelWeightsFingerprint,
            tokenizerJSONSHA256: modelIdentityHashes.tokenizerJSONSHA256,
            tokenizerConfigSHA256: modelIdentityHashes.tokenizerConfigSHA256,
            chatTemplateSHA256: modelIdentityHashes.chatTemplateSHA256,
            systemPromptSHA256: Self.sha256Text(request.systemPrompt),
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
        diagnosticSink?("neural_imprint_capture_header_done")

        let captureMemorySnapshot = DeviceProfile.captureMemorySnapshot()
        let captureUsesSyncPrefill = Self.neuralImprintCaptureUsesSyncPrefill(
            memorySnapshot: captureMemorySnapshot,
            planSyncEval: currentPlan?.syncEval
        )
        let capturePrefillStep = Self.neuralImprintCapturePrefillStep(
            prefixTokenCount: prefixTokenCount,
            planPrefillStepSize: currentPlan?.prefillStepSize,
            syncPrefill: captureUsesSyncPrefill
        )
        diagnosticSink?(
            "neural_imprint_capture_session_init_begin prefix=\(prefixTokenCount) prefillStep=\(capturePrefillStep) syncPrefill=\(captureUsesSyncPrefill) availableMB=\(captureMemorySnapshot.availableMB) footprintMB=\(captureMemorySnapshot.footprintMB) jetsamLimitMB=\(captureMemorySnapshot.jetsamLimitMB)"
        )
        try applyCmlxCommandBufferLimits(contextLengthHint: capturePrefillStep)
        let session = try prepareCmlxSessionForNeuralImprintCapture(
            bundleIndex: bundleIndex,
            runtime: runtime
        )
        try prefillNeuralImprintCapture(
            session: session,
            tokenIDs: request.prefixTokenIDs,
            chunkSize: capturePrefillStep,
            syncPrefill: captureUsesSyncPrefill
        )
        diagnosticSink?("neural_imprint_capture_save_begin")
        try session.session.saveNeuralImprintCache(
            artifactURL: artifactURL,
            metadata: header
        )
        diagnosticSink?("neural_imprint_capture_save_done")
        diagnosticSink?("neural_imprint_capture_session_reset_begin")
        try session.reset()
        diagnosticSink?("neural_imprint_capture_session_reset_done")

        diagnosticSink?("neural_imprint_capture_artifact_map_begin")
        let artifact = try SafeTensorsShardFile(url: artifactURL)
        diagnosticSink?("neural_imprint_capture_artifact_map_done")
        diagnosticSink?("neural_imprint_capture_sidecar_payload_begin")
        let sidecarPayload = try Self.neuralImprintSidecarPayload(
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
        diagnosticSink?("neural_imprint_capture_sidecar_payload_done")
        diagnosticSink?("neural_imprint_capture_sidecar_encode_begin")
        let sidecarData = try JSONSerialization.data(
            withJSONObject: sidecarPayload,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        diagnosticSink?("neural_imprint_capture_sidecar_encode_done")
        diagnosticSink?("neural_imprint_capture_sidecar_write_begin")
        try sidecarData.write(to: metadataURL, options: [.atomic])
        diagnosticSink?("neural_imprint_capture_sidecar_write_done")

        diagnosticSink?("neural_imprint_capture_sidecar_load_begin")
        let sidecar = try NeuralImprintSidecar.load(from: metadataURL)
        diagnosticSink?("neural_imprint_capture_sidecar_load_done")
        diagnosticSink?("neural_imprint_capture_requirements_begin")
        let requirements = try Self.neuralImprintCompatibilityRequirements(
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
        diagnosticSink?("neural_imprint_capture_requirements_done")
        diagnosticSink?("neural_imprint_capture_validator_begin")
        try NeuralImprintArtifactValidator.validate(
            artifact: artifact,
            sidecar: sidecar,
            requirements: requirements
        )
        diagnosticSink?("neural_imprint_capture_validator_done")

        diagnosticSink?("neural_imprint_capture_status_begin")
        let status = NeuralImprintCacheStatus(
            directory: request.outputDirectory,
            artifactURL: artifactURL,
            metadataURL: metadataURL,
            artifactSHA256: sidecar.artifactSHA256,
            prefixTokenCount: prefixTokenCount,
            modelID: modelID,
            enableThinking: request.enableThinking,
            cacheBackend: try Self.requiredNeuralImprintHeader("cache_backend", in: artifact.metadata),
            cacheBackendVersion: try Self.requiredNeuralImprintHeader("cache_backend_version", in: artifact.metadata)
        )
        diagnosticSink?("neural_imprint_capture_status_done")
        diagnosticSink?(
            "neural_imprint_capture_done prefix=\(status.prefixTokenCount) artifactSHA256=\(status.artifactSHA256)"
        )
        return status
    }

    public func unload() {
        state = .idle
        loadedConfig = nil
        pruningMetadata = nil
        archInfo = nil
        currentPlan = nil
        memoryPolicy = nil
        cachedModelIdentityHashes = nil
        modelDirectory = nil
        nativeRuntime = nil
        nativeExecutor = nil
        nativeModel = nil
        nativeArchitecture = nil
        nativeBundleIndex = nil
        nativeTokenizer = nil
        nativeEndTokenIds = []
        nativeUseCmlxLazyDecode = false
        nativeUseCmlxLazyTextSuffixPromptCache = true
        nativeCmlxLazyDecodeSession = nil
        nativeCmlxLazyDecodeSessionDSRPolicies = [:]
        nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
        nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
        nativeCmlxLimitState.reset()
        onlineCalibrator = nil
        activeOnlineCalibrationOverrides = nil
        clearNativeGreedyDecodeSession()
        turnCounter = 0
        lastPolicy = nil
        lastMetrics = nil
        activeNeuralImprintCache = nil
        promptCache.clear()
        downloadProgress = 0
    }

    public nonisolated static func estimateModelSizeGB(directory: URL) -> Double {
        if let indexedBytes = estimateIndexedSafetensorsBytes(directory: directory), indexedBytes > 0 {
            return Double(indexedBytes) / (1024 * 1024 * 1024)
        }

        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey]
        )) ?? []
        let totalBytes = files
            .filter { $0.pathExtension == "safetensors" }
            .compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }
            .reduce(0, +)
        return totalBytes > 0 ? Double(totalBytes) / (1024 * 1024 * 1024) : 4.0
    }

    private nonisolated static func shouldEnableCmlxLazyDecodeByDefault(
        plan: MemoryBudgetPlanner.Plan
    ) -> Bool {
        plan.fusedGDNDecodeEnabled || plan.vendoredCommandBufferPrefillQMMEnabled
    }

    public nonisolated static func neuralImprintSystemPrompt(profileBody: String) -> String {
        NeuralImprintRuntimeSupport.neuralImprintSystemPrompt(profileBody: profileBody)
    }

    private static func computeModelIdentityHashes(
        modelDirectory: URL
    ) throws -> ModelIdentityHashes {
        let chatTemplateSHA256 = try neuralImprintChatTemplateSHA256(modelDirectory: modelDirectory)
        return try ModelIdentityHashes(
            modelDirectoryPath: modelDirectory.standardizedFileURL.path,
            modelArchitectureID: neuralImprintModelArchitectureIdentifier(modelDirectory: modelDirectory),
            modelConfigSHA256: sha256File(modelDirectory.appendingPathComponent("config.json")),
            modelWeightsFingerprint: neuralImprintModelWeightsFingerprint(modelDirectory: modelDirectory),
            tokenizerJSONSHA256: sha256File(modelDirectory.appendingPathComponent("tokenizer.json")),
            tokenizerConfigSHA256: sha256File(modelDirectory.appendingPathComponent("tokenizer_config.json")),
            chatTemplateSHA256: chatTemplateSHA256
        )
    }

    private func modelIdentityHashes(
        for modelDirectory: URL
    ) throws -> ModelIdentityHashes {
        let standardizedPath = modelDirectory.standardizedFileURL.path
        if let cachedModelIdentityHashes,
           cachedModelIdentityHashes.modelDirectoryPath == standardizedPath {
            diagnosticSink?("neural_imprint_capture_model_identity_cache_hit")
            return cachedModelIdentityHashes
        }
        diagnosticSink?("neural_imprint_capture_model_identity_cache_miss")
        let computed = try Self.computeModelIdentityHashes(modelDirectory: modelDirectory)
        cachedModelIdentityHashes = computed
        return computed
    }

    nonisolated static func neuralImprintPrefixSplitIndex(
        fullTokenIDs: [Int],
        markerTokenIDs: [Int]
    ) -> Int? {
        NeuralImprintRuntimeSupport.neuralImprintPrefixSplitIndex(
            fullTokenIDs: fullTokenIDs,
            markerTokenIDs: markerTokenIDs
        )
    }

    private nonisolated static func estimateIndexedSafetensorsBytes(directory: URL) -> Int? {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(SafetensorsWeightIndex.self, from: data)
        else {
            return nil
        }

        let shardNames = Set(index.weightMap.values.filter { $0.hasSuffix(".safetensors") })
        guard !shardNames.isEmpty else { return 0 }

        return shardNames.reduce(0) { total, shardName in
            let shardURL = directory.appendingPathComponent(shardName)
            let size = (try? shardURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return total + size
        }
    }

    private struct SafetensorsWeightIndex: Decodable {
        let weightMap: [String: String]

        private enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    static func neuralImprintCompatibleParameters(
        _ parameters: EdgeGenerateParameters
    ) -> EdgeGenerateParameters {
        NeuralImprintRuntimeSupport.neuralImprintCompatibleParameters(parameters)
    }

    static func neuralImprintArtifactURL(
        sidecarArtifactPath: String,
        directory: URL
    ) -> URL {
        NeuralImprintRuntimeSupport.neuralImprintArtifactURL(
            sidecarArtifactPath: sidecarArtifactPath,
            directory: directory
        )
    }

    static func neuralImprintMetadataURL(directory: URL) -> URL {
        NeuralImprintRuntimeSupport.neuralImprintMetadataURL(directory: directory)
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
        try NeuralImprintRuntimeSupport.neuralImprintCompatibilityRequirements(
            modelDirectory: modelDirectory,
            architecture: architecture,
            artifactHeader: artifactHeader,
            modelArchitectureID: modelArchitectureID,
            modelConfigSHA256: modelConfigSHA256,
            modelWeightsFingerprint: modelWeightsFingerprint,
            tokenizerJSONSHA256: tokenizerJSONSHA256,
            tokenizerConfigSHA256: tokenizerConfigSHA256,
            chatTemplateSHA256: chatTemplateSHA256
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
        try NeuralImprintRuntimeSupport.neuralImprintArtifactHeader(
            modelID: modelID,
            modelDirectory: modelDirectory,
            modelArchitectureID: modelArchitectureID,
            modelConfigSHA256: modelConfigSHA256,
            modelWeightsFingerprint: modelWeightsFingerprint,
            tokenizerJSONSHA256: tokenizerJSONSHA256,
            tokenizerConfigSHA256: tokenizerConfigSHA256,
            chatTemplateSHA256: chatTemplateSHA256,
            systemPromptSHA256: systemPromptSHA256,
            renderedPrefixSHA256: renderedPrefixSHA256,
            prefixTokenIDsSHA256: prefixTokenIDsSHA256,
            prefixTokenCount: prefixTokenCount,
            toolSchemaSHA256: toolSchemaSHA256,
            profileBodySHA256: profileBodySHA256,
            enableThinking: enableThinking,
            cacheBackendVersion: cacheBackendVersion,
            createdAt: createdAt,
            createdBy: createdBy,
            writerVersion: writerVersion,
            minReaderVersion: minReaderVersion
        )
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
        try NeuralImprintRuntimeSupport.neuralImprintSidecarPayload(
            artifactURL: artifactURL,
            artifact: artifact,
            request: request,
            modelID: modelID,
            architecture: architecture,
            tokenizerJSONSHA256: tokenizerJSONSHA256,
            tokenizerConfigSHA256: tokenizerConfigSHA256,
            chatTemplateSHA256: chatTemplateSHA256,
            renderedPrefixSHA256: renderedPrefixSHA256,
            prefixTokenIDsSHA256: prefixTokenIDsSHA256,
            profileBodySHA256: profileBodySHA256
        )
    }

    static func neuralImprintCacheManifest(
        artifact: SafeTensorsShardFile,
        architecture: QwenHybridArchitecture,
        prefixTokenCount: Int
    ) throws -> [String: Any] {
        try NeuralImprintRuntimeSupport.neuralImprintCacheManifest(
            artifact: artifact,
            architecture: architecture,
            prefixTokenCount: prefixTokenCount
        )
    }

    static func neuralImprintTokenIDsSHA256(_ tokenIDs: [Int]) -> String {
        NeuralImprintRuntimeSupport.neuralImprintTokenIDsSHA256(tokenIDs)
    }

    static func requiredNeuralImprintHeader(
        _ key: String,
        in header: [String: String]
    ) throws -> String {
        try NeuralImprintRuntimeSupport.requiredNeuralImprintHeader(key, in: header)
    }

    static func neuralImprintModelArchitectureIdentifier(modelDirectory: URL) throws -> String {
        try NeuralImprintRuntimeSupport.neuralImprintModelArchitectureIdentifier(
            modelDirectory: modelDirectory
        )
    }

    static func neuralImprintChatTemplateSHA256(
        modelDirectory: URL,
        expectedSHA256: String? = nil
    ) throws -> String {
        try NeuralImprintRuntimeSupport.neuralImprintChatTemplateSHA256(
            modelDirectory: modelDirectory,
            expectedSHA256: expectedSHA256
        )
    }

    static func neuralImprintModelWeightsFingerprint(modelDirectory: URL) throws -> String {
        try NeuralImprintRuntimeSupport.neuralImprintModelWeightsFingerprint(
            modelDirectory: modelDirectory
        )
    }

    static func neuralImprintModelWeightFiles(modelDirectory: URL) throws -> [URL] {
        try NeuralImprintRuntimeSupport.neuralImprintModelWeightFiles(
            modelDirectory: modelDirectory
        )
    }

    static func neuralImprintWeightsFingerprintJSON(
        indexSHA256: String?,
        shards: [(name: String, sizeBytes: Int, sha256: String)]
    ) -> String {
        NeuralImprintRuntimeSupport.neuralImprintWeightsFingerprintJSON(
            indexSHA256: indexSHA256,
            shards: shards
        )
    }

    static func jsonDictionary(at url: URL) throws -> [String: Any] {
        try NeuralImprintRuntimeSupport.jsonDictionary(at: url)
    }

    static func sha256File(_ url: URL) throws -> String {
        try NeuralImprintRuntimeSupport.sha256File(url)
    }

    static func sha256Text(_ text: String) -> String {
        NeuralImprintRuntimeSupport.sha256Text(text)
    }

    private func releaseNativeCmlxLazySession(reason: String) {
        if let session = nativeCmlxLazyDecodeSession {
            diagnosticSink?("cmlx_lazy_session_release reason=\(reason)")
            do {
                try session.reset()
                diagnosticSink?("cmlx_lazy_session_reset_done reason=\(reason)")
            } catch {
                diagnosticSink?("cmlx_lazy_session_reset_failed reason=\(reason) error=\(error)")
            }
        }
        nativeCmlxLazyDecodeSession = nil
        nativeCmlxLazyDecodeSessionDSRPolicies = [:]
        nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
        nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
        nativeCmlxLimitState.reset()
    }

    private func loadNativeModelFallback(
        runtime: EdgeMetalRuntime,
        bundleIndex: QwenModelBundleIndex
    ) throws -> (model: QwenHybridModelReference, executor: MetalKernelExecutor) {
        if let model = nativeModel, let executor = nativeExecutor {
            return (model, executor)
        }
        nativeCmlxLazyDecodeSession = nil
        nativeCmlxLazyDecodeSessionDSRPolicies = [:]
        nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
        nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
        nativeCmlxLimitState.reset()
        clearNativeGreedyDecodeSession()
        diagnosticSink?("native_fallback_model_load_begin")
        let executor = try MetalKernelExecutor(runtime: runtime)
        let model = try QwenHybridModelReference.loadHuggingFaceLayout(
            weightStore: QwenModelWeightStore(bundleIndex: bundleIndex),
            runtime: runtime
        )
        nativeExecutor = executor
        nativeModel = model
        diagnosticSink?("native_fallback_model_load_done")
        return (model, executor)
    }

    private func runNativeGenerate(
        messages: [ChatMessage],
        tools: [ToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters requestedParameters: EdgeGenerateParameters,
        bypassPolicy: Bool,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No LLM model loaded")
        }
        let generateEnteredAt = Date()
        guard let runtime = nativeRuntime,
              let architecture = nativeArchitecture,
              let bundleIndex = nativeBundleIndex,
              let tokenizer = nativeTokenizer
        else {
            throw EdgeRuntimeError.loadFailed("Native Qwen runtime is not initialized")
        }

        state = .generating
        var generateSucceeded = false
        defer {
            if !generateSucceeded {
                clearNativeGreedyDecodeSession()
            }
            if state == .generating {
                state = .ready
            }
        }

        var parameters = requestedParameters
        let neuralImprintStatus = activeNeuralImprintCache
        if let neuralImprintStatus,
           parameters.enableThinking != neuralImprintStatus.enableThinking {
            throw EdgeRuntimeError.unsupportedFeature(
                "Neural Imprint enableThinking mismatch: artifact=\(neuralImprintStatus.enableThinking) request=\(parameters.enableThinking)"
            )
        }
        let promptTools = neuralImprintStatus == nil ? tools : nil
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: parameters.preserveThinking
        )
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: promptMessages.chatTemplateMessages(
                preserveThinking: parameters.preserveThinking
            ),
            tools: promptTools,
            additionalContext: Self.chatTemplateContext(parameters: parameters)
        )
        let inputPreparedAt = Date()
        let arch = archInfo ?? ModelArchInfo.fallback(
            modelSizeGB: Self.estimateModelSizeGB(directory: modelDirectory ?? URL(fileURLWithPath: "."))
        )
        let snapshot = InferencePolicy.DeviceSnapshot.capture()
        applyOnlineCalibrationBeforeTurn(contextLengthHint: promptTokens.count)
        let resolvedPolicy: InferencePolicy.Resolved?
        let turn = turnCounter + 1
        if bypassPolicy {
            resolvedPolicy = nil
        } else {
            let resolved = InferencePolicy.resolve(
                snapshot: snapshot,
                context: InferencePolicy.TurnContext(
                    turn: turn,
                    cachedTokenCount: promptCache.tokenCount,
                    archInfo: arch,
                    scene: parameters.dsrScene,
                    requestedMaxTokens: parameters.maxTokens,
                    planPrefillStepSize: currentPlan?.prefillStepSize,
                    memoryIntent: currentPlan?.memoryIntent ?? memoryPolicy?.memoryIntent ?? .balanced
                ),
                staticPolicy: memoryPolicy,
                dsrMaxCriticalOverride: requestedParameters.dsrMaxCritical
            )
            resolved.apply(to: &parameters)
            lastPolicy = resolved
            resolvedPolicy = resolved
            if resolved.shouldPause {
                throw EdgeRuntimeError.thermalPause
            }
        }
        var policyReasonSuffix = ""
        if let evictOverride = Self.dsrEvictionIntervalEnvironmentOverride(),
           parameters.useDSR {
            parameters.dsrEvictionInterval = evictOverride
            policyReasonSuffix = " | DSR evict override \(evictOverride) (env)"
        }
        if let kvBitsOverride = Self.kvBitsEnvironmentOverride() {
            parameters.kvBits = kvBitsOverride > 0 ? kvBitsOverride : 0
            policyReasonSuffix += " | KV bits override \(kvBitsOverride) (env)"
        }
        if neuralImprintStatus != nil {
            parameters = Self.neuralImprintCompatibleParameters(parameters)
            policyReasonSuffix += " | neuralImprint full-cache restore disables DSR/KV-quant/FrogJump"
        }

        diagnosticSink?("tokenize_done tokens=\(promptTokens.count)")
        let useGreedyTokenPath = parameters.temperature == 0
        let dsrPolicies = try Self.makeDSRPolicies(
            parameters: parameters,
            architecture: architecture
        )
        let kvCapacity = Self.kvCapacity(
            promptTokenCount: promptTokens.count,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let policyReason = (resolvedPolicy?.reasoning ?? "bypassed") + policyReasonSuffix
        diagnosticSink?(
            "edgekit_policy_effective useDSR=\(parameters.useDSR) dsrMax=\(parameters.dsrMaxCritical.map(String.init) ?? "nil") dsrPolicy=\(Self.dsrPolicySummary(dsrPolicies)) scoreActivation=\(dsrPolicies.values.sorted { $0.maxSize < $1.maxSize }.first.map { String(format: "%.2f", $0.scoreActivationRatio) } ?? "nil") policyReason=\(policyReason)"
        )
        let planningDoneAt = Date()

        let memoryBefore = DeviceProfile.captureMemorySnapshot().footprintMB
        let startedAt = Date()
        try Task.checkCancellation()
        if let rejection = cmlxLazyDecodeRejectionReason(
            tools: tools,
            parameters: parameters,
            promptTokenCount: promptTokens.count,
            neuralImprintPrefixTokenCount: neuralImprintStatus?.prefixTokenCount ?? 0,
            architecture: architecture,
            useGreedyTokenPath: useGreedyTokenPath
        ) {
            if nativeUseCmlxLazyDecode {
                diagnosticSink?("cmlx_lazy_decode_reject reason=\(rejection)")
            }
            if neuralImprintStatus != nil {
                throw EdgeRuntimeError.unsupportedFeature(
                    "Neural Imprint restore requires Cmlx lazy decode: \(rejection)"
                )
            }
            if nativeCmlxLazyDecodeSession != nil {
                releaseNativeCmlxLazySession(reason: rejection)
            }
        } else {
            try await runCmlxLazyGenerate(
                messages: promptMessages,
                promptTokens: promptTokens,
                parameters: parameters,
                tokenizer: tokenizer,
                bundleIndex: bundleIndex,
                runtime: runtime,
                continuation: continuation,
                startedAt: startedAt,
                inputPreparationMs: inputPreparedAt.timeIntervalSince(generateEnteredAt) * 1000,
                planningMs: planningDoneAt.timeIntervalSince(inputPreparedAt) * 1000,
                memoryBefore: memoryBefore,
                dsrPolicies: dsrPolicies,
                resolvedPolicy: resolvedPolicy,
                neuralImprintStatus: neuralImprintStatus,
                onToolCall: onToolCall
            )
            generateSucceeded = true
            return
        }
        let fallbackSessionStartedAt = Date()
        let loadedFallback = try loadNativeModelFallback(
            runtime: runtime,
            bundleIndex: bundleIndex
        )
        let preparedSession = try prepareNativeDecodeSession(
            messages: promptMessages,
            promptTokens: promptTokens,
            kvCapacity: kvCapacity,
            dsrPolicies: dsrPolicies,
            skippableCachedTokenSequences: Self.qwenThinkingSentinelTokenSequences(tokenizer: tokenizer),
            tokenizer: tokenizer,
            enableThinking: parameters.enableThinking,
            toolsAreEmpty: tools?.isEmpty ?? true,
            model: loadedFallback.model,
            runtime: runtime,
            executor: loadedFallback.executor,
            architecture: architecture,
            useGreedyTokenPath: useGreedyTokenPath
        )
        let fallbackSessionReadyAt = Date()
        let session = preparedSession.session
        let cachedTokensReused = preparedSession.cachedTokensReused
        let promptCacheHit = cachedTokensReused > 0
        promptCache.update(
            cache: [],
            totalTokenCount: nativeDecodeSessionTokenIds.count,
            tokenPrefix: promptTokens.map { Int32($0) }
        )

        var rng = EdgeSeededRandomNumberGenerator(seed: UInt64.random(in: 1...UInt64.max))
        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(parameters.maxTokens)
        var emittedText = ""
        var emittedTokenCount = 0
        var buffersToolCallText = onToolCall != nil
        var toolCallDetectionResolved = !buffersToolCallText
        let toolCallDetectionWindow = Self.toolCallDetectionWindow()
        var firstTokenAt: Date?
        var firstTokenSelectionStartedAt: Date?
        var firstTokenSelectionMs: Double?

        while generatedTokenIds.count < parameters.maxTokens {
            try Task.checkCancellation()
            let tokenId: Int
            if generatedTokenIds.isEmpty {
                firstTokenSelectionStartedAt = Date()
                diagnosticSink?("select_first_token_begin")
            }
            let sampling = NativeCmlxSampling.qwenSamplingConfiguration(
                parameters: parameters,
                promptSessionTokenIds: nativeDecodeSessionTokenIds,
                generatedTokenIds: generatedTokenIds,
                endTokenIds: nativeEndTokenIds
            )
            if sampling.temperature == 0,
               parameters.minimumGeneratedTokens == 0,
               parameters.eosPenaltyUntilToken == 0 {
                tokenId = try session.selectNextToken().tokenId
            } else {
                tokenId = try session.selectSampledToken(
                    configuration: sampling,
                    rng: &rng
                ).tokenId
            }
            if parameters.stopOnEndToken,
               nativeEndTokenIds.contains(tokenId),
               generatedTokenIds.count >= parameters.minimumGeneratedTokens {
                session.invalidateCurrentLogits()
                break
            }
            if firstTokenAt == nil {
                firstTokenAt = Date()
                firstTokenSelectionMs = firstTokenAt.map {
                    $0.timeIntervalSince(firstTokenSelectionStartedAt ?? startedAt) * 1000
                }
                diagnosticSink?("select_first_token_done token=\(tokenId)")
            }
            generatedTokenIds.append(tokenId)
            if buffersToolCallText {
                continuation.yield(
                    GenerateChunk(
                        text: "",
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
                if !toolCallDetectionResolved,
                   generatedTokenIds.count >= toolCallDetectionWindow {
                    toolCallDetectionResolved = true
                    let bufferedText = tokenizer.decode(
                        tokens: generatedTokenIds,
                        skipSpecialTokens: true
                    )
                    if Self.decodedTextContainsToolCallStart(bufferedText) {
                        diagnosticSink?("tool_call_buffer_keep detected=true window=\(toolCallDetectionWindow) generated=\(generatedTokenIds.count)")
                    } else {
                        buffersToolCallText = false
                        emittedText = bufferedText
                        emittedTokenCount = generatedTokenIds.count
                        diagnosticSink?("tool_call_buffer_early_flush window=\(toolCallDetectionWindow) generated=\(generatedTokenIds.count) chars=\(bufferedText.count)")
                        if !bufferedText.isEmpty {
                            continuation.yield(
                                GenerateChunk(
                                    text: bufferedText,
                                    generatedTokenCount: generatedTokenIds.count
                                )
                            )
                        }
                    }
                }
            } else {
                emitDecodedTextIfNeeded(
                    tokenizer: tokenizer,
                    generatedTokenIds: generatedTokenIds,
                    emittedText: &emittedText,
                    emittedTokenCount: &emittedTokenCount,
                    continuation: continuation
                )
            }
            if useGreedyTokenPath {
                try session.advanceGreedy(with: tokenId)
            } else {
                try session.advance(with: tokenId)
            }
            nativeDecodeSessionTokenIds.append(tokenId)
        }
        if buffersToolCallText {
            try await emitFinalDecodedTextHandlingToolCalls(
                tokenizer: tokenizer,
                generatedTokenIds: generatedTokenIds,
                emittedText: &emittedText,
                emittedTokenCount: &emittedTokenCount,
                continuation: continuation,
                onToolCall: onToolCall
            )
        } else {
            emitDecodedTextIfNeeded(
                tokenizer: tokenizer,
                generatedTokenIds: generatedTokenIds,
                emittedText: &emittedText,
                emittedTokenCount: &emittedTokenCount,
                continuation: continuation,
                force: true
            )
        }

        let endedAt = Date()
        let first = firstTokenAt ?? endedAt
        let decodeSeconds = max(endedAt.timeIntervalSince(first), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        promptCache.update(
            cache: [],
            totalTokenCount: nativeDecodeSessionTokenIds.count,
            tokenPrefix: promptTokens.map { Int32($0) }
        )
        nativeDecodeSessionPromptMessages = promptMessages
        nativeDecodeSessionLastAssistantText = NativePromptSessionReuse.normalizeAssistantText(emittedText)
        turnCounter = turn
        let metrics = InferenceMetrics(
            ttftMs: first.timeIntervalSince(startedAt) * 1000,
            decodeTPS: Double(generatedTokenIds.count) / decodeSeconds,
            deferredLoadMs: nil,
            phaseTimings: InferencePhaseTimings(
                inputPreparationMs: inputPreparedAt.timeIntervalSince(generateEnteredAt) * 1000,
                planningMs: planningDoneAt.timeIntervalSince(inputPreparedAt) * 1000,
                warmupMs: fallbackSessionReadyAt.timeIntervalSince(fallbackSessionStartedAt) * 1000,
                sessionSetupMs: fallbackSessionReadyAt.timeIntervalSince(fallbackSessionStartedAt) * 1000,
                firstTokenSelectionMs: firstTokenSelectionMs,
                firstDecodeTokenMs: firstTokenSelectionMs,
                decodeMs: endedAt.timeIntervalSince(first) * 1000
            ),
            promptTokenCount: promptTokens.count,
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: resolvedPolicy?.reasoning ?? "policy bypassed",
            promptCacheHit: promptCacheHit,
            cachedTokensReused: cachedTokensReused,
            thermalState: ThermalManager().level.rawValue,
            turn: turn
        )
        lastMetrics = metrics
        recordOnlineCalibration(metrics: metrics)
        generateSucceeded = true
    }

    private func cmlxLazyDecodeRejectionReason(
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters,
        promptTokenCount: Int,
        neuralImprintPrefixTokenCount: Int,
        architecture: QwenHybridArchitecture,
        useGreedyTokenPath: Bool
    ) -> String? {
        guard nativeUseCmlxLazyDecode else {
            return "disabled"
        }
        guard parameters.temperature >= 0, parameters.temperature.isFinite else {
            return "invalid_temperature"
        }
        guard parameters.topP.isFinite else {
            return "invalid_top_p"
        }
        guard parameters.minP >= 0, parameters.minP.isFinite else {
            return "invalid_min_p"
        }
        if let topK = parameters.topK, topK <= 0 {
            return "invalid_top_k"
        }
        guard parameters.repetitionPenalty > 0,
              parameters.repetitionPenalty.isFinite
        else {
            return "invalid_repetition_penalty"
        }
        guard parameters.repetitionContextSize >= 0 else {
            return "invalid_repetition_context_size"
        }
        guard parameters.presencePenalty.isFinite else {
            return "invalid_presence_penalty"
        }
        guard parameters.presenceContextSize >= 0 else {
            return "invalid_presence_context_size"
        }
        guard parameters.frequencyPenalty.isFinite else {
            return "invalid_frequency_penalty"
        }
        guard parameters.frequencyContextSize >= 0 else {
            return "invalid_frequency_context_size"
        }
        guard parameters.minimumGeneratedTokens >= 0 else {
            return "invalid_minimum_generated_tokens"
        }
        guard parameters.eosPenaltyUntilToken >= 0 else {
            return "invalid_eos_penalty_until_token"
        }
        _ = useGreedyTokenPath
        let requestedContext = neuralImprintPrefixTokenCount + promptTokenCount + parameters.maxTokens
        guard requestedContext <= architecture.contextLength else {
            return "context_overflow"
        }
        #if os(iOS)
        let memory = DeviceProfile.captureMemorySnapshot()
        let modelSizeMB = Int(
            (Self.estimateModelSizeGB(directory: modelDirectory ?? URL(fileURLWithPath: ".")) * 1024)
                .rounded(.up)
        )
        let cmlxSessionAlreadyResident = nativeCmlxLazyDecodeSession != nil
        let requiredHeadroomMB = cmlxSessionAlreadyResident
            ? 256
            : Int((Double(modelSizeMB) * 0.9).rounded(.up)) + 512
        guard memory.availableMB >= requiredHeadroomMB else {
            return "insufficient_headroom available=\(memory.availableMB)MB required=\(requiredHeadroomMB)MB"
        }
        #endif
        return nil
    }

    private func applyCmlxCommandBufferLimits(contextLengthHint: Int) throws {
        try NativeCmlxCommandBufferLimitApplier.apply(
            contextLengthHint: contextLengthHint,
            state: &nativeCmlxLimitState,
            commandBufferDiagnosticName: "cmlx_lazy_command_buffer_limits",
            memoryLimitDiagnosticName: "cmlx_lazy_memory_limit",
            emitDiagnostic: diagnosticSink
        )
    }

    private func runCmlxLazyGenerate(
        messages: [ChatMessage],
        promptTokens: [Int],
        parameters: EdgeGenerateParameters,
        tokenizer: Tokenizer,
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation,
        startedAt: Date,
        inputPreparationMs: Double,
        planningMs: Double,
        memoryBefore: Int,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        resolvedPolicy: InferencePolicy.Resolved?,
        neuralImprintStatus: NeuralImprintCacheStatus?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?
    ) async throws {
        guard !promptTokens.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        nativeDecodeSession = nil
        nativeDecodeSessionCapacity = 0
        nativeDecodeSessionDSRPolicies = [:]
        let sessionSetupStartedAt = Date()

        let cachedDSRPolicies = nativeCmlxLazyDecodeSessionDSRPolicies
        let requestedAttentionCacheQuantization =
            Self.cmlxAttentionCacheQuantization(
                parameters: parameters,
                dsrPolicies: dsrPolicies
            )
        let requestedFrogJumpPlan = QwenFrogJumpPlan.compute(
            architecture: bundleIndex.architecture,
            requestedEnabled: parameters.frogJumpEnabled,
            thinkingEnabled: parameters.enableThinking
        )
        let requestedFrogJumpMask = requestedFrogJumpPlan.layerMask
        let requestedFrogJumpSummary = requestedFrogJumpPlan.layerSummary
        let cachedAttentionCacheQuantization =
            nativeCmlxLazyDecodeSessionAttentionCacheQuantization
        let canUpdateResidentDSRPolicies =
            nativeCmlxLazyDecodeSession != nil &&
            cachedAttentionCacheQuantization == requestedAttentionCacheQuantization &&
            nativeCmlxLazyDecodeSessionFrogJumpLayerMask == requestedFrogJumpMask &&
            Self.canUpdateResidentDSRPoliciesInPlace(
                requested: dsrPolicies,
                cached: cachedDSRPolicies
            )
        let canReuseResidentSession =
            nativeCmlxLazyDecodeSession != nil &&
            cachedAttentionCacheQuantization == requestedAttentionCacheQuantization &&
            nativeCmlxLazyDecodeSessionFrogJumpLayerMask == requestedFrogJumpMask &&
            (cachedDSRPolicies == dsrPolicies || canUpdateResidentDSRPolicies)
        let hadResidentSession = canReuseResidentSession
        let attentionCacheLimit = parameters.maxKVSize ?? parameters.dsrMaxCritical
        let session: QwenCmlxLazyDecodeSession
        if let existing = nativeCmlxLazyDecodeSession, canReuseResidentSession {
            session = existing
            try session.setAttentionCacheLimit(attentionCacheLimit)
            if canUpdateResidentDSRPolicies {
                try session.updateDSRPoliciesInPlace(dsrPolicies)
                nativeCmlxLazyDecodeSessionDSRPolicies = dsrPolicies
                diagnosticSink?(
                    "cmlx_lazy_dsr_policy_updated requested=\(Self.dsrPolicySummary(dsrPolicies)) cached=\(Self.dsrPolicySummary(cachedDSRPolicies))"
                )
            }
        } else {
            if nativeCmlxLazyDecodeSession != nil {
                diagnosticSink?(
                    "cmlx_lazy_session_release reason=dsr_kv_or_frog_jump_changed requestedLayers=\(dsrPolicies.count) cachedLayers=\(nativeCmlxLazyDecodeSessionDSRPolicies.count) requestedKV=\(NativeCmlxAttentionCacheQuantization.summary(requestedAttentionCacheQuantization)) cachedKV=\(NativeCmlxAttentionCacheQuantization.summary(cachedAttentionCacheQuantization)) requestedFrog=0x\(String(requestedFrogJumpMask, radix: 16)) cachedFrog=0x\(String(nativeCmlxLazyDecodeSessionFrogJumpLayerMask, radix: 16))"
                )
            }
            nativeCmlxLazyDecodeSession = nil
            nativeCmlxLazyDecodeSessionDSRPolicies = [:]
            nativeCmlxLazyDecodeSessionAttentionCacheQuantization = nil
            nativeCmlxLazyDecodeSessionFrogJumpLayerMask = 0
            nativeCmlxLimitState.reset()
            diagnosticSink?("cmlx_lazy_session_init_begin")
            let created = try QwenCmlxLazyDecodeSession(
                bundleIndex: bundleIndex,
                runtime: runtime,
                attentionCacheLimit: attentionCacheLimit,
                dsrPolicies: dsrPolicies,
                attentionCacheQuantizationGroupSize: requestedAttentionCacheQuantization?.groupSize,
                attentionCacheQuantizationBits: requestedAttentionCacheQuantization?.bits,
                frogJumpLayerMask: requestedFrogJumpMask
            )
            nativeCmlxLazyDecodeSession = created
            nativeCmlxLazyDecodeSessionDSRPolicies = dsrPolicies
            nativeCmlxLazyDecodeSessionAttentionCacheQuantization =
                requestedAttentionCacheQuantization
            nativeCmlxLazyDecodeSessionFrogJumpLayerMask = requestedFrogJumpMask
            session = created
            diagnosticSink?(
                "cmlx_lazy_session_init_done floats=\(created.registeredFloatTensorCount) quantized=\(created.registeredQuantizedTensorCount) attentionKV=\(NativeCmlxAttentionCacheQuantization.summary(requestedAttentionCacheQuantization)) frogJump=\(requestedFrogJumpSummary)"
            )
        }
        let sessionReadyAt = Date()
        let cmlxEvalProfileEnabled = NativeEnvironment.bool(
            ["EDGE_CMLX_EVAL_PROFILE", "CMLX_EVAL_PROFILE", "EDGE_CMLX_METAL_PROFILE", "CMLX_METAL_PROFILE"],
            defaultValue: false
        )
        let cmlxPrefillChunkDiagnosticsEnabled = NativeEnvironment.bool(
            ["EDGE_CMLX_PREFILL_CHUNK_DIAGNOSTICS", "CMLX_PREFILL_CHUNK_DIAGNOSTICS"],
            defaultValue: false
        )
        let cmlxPrefillChunkProfileEnabled = NativeEnvironment.bool(
            ["EDGE_CMLX_PREFILL_CHUNK_PROFILE", "CMLX_PREFILL_CHUNK_PROFILE"],
            defaultValue: false
        )
        let cmlxPrefillBarrierAvailableMB = NativeEnvironment.int(
            ["EDGE_CMLX_PREFILL_BARRIER_AVAIL_MB", "CMLX_PREFILL_BARRIER_AVAIL_MB"],
            defaultValue: 0,
            range: 0...65_536
        )
        let cmlxPrefillBarrierMaxChunks = NativeEnvironment.int(
            ["EDGE_CMLX_PREFILL_BARRIER_MAX_CHUNKS", "CMLX_PREFILL_BARRIER_MAX_CHUNKS"],
            defaultValue: 0,
            range: 0...4_096
        )
        let cmlxTokenTimingEnabled = NativeEnvironment.bool(
            ["EDGE_CMLX_TOKEN_TIMING", "CMLX_TOKEN_TIMING"],
            defaultValue: false
        )

        var prefillTokens = promptTokens
        var cachedTokensReused = 0
        var promptCacheHit = false
        if let neuralImprintStatus {
            try session.restoreNeuralImprintCache(
                artifactURL: neuralImprintStatus.artifactURL,
                prefixTokenCount: neuralImprintStatus.prefixTokenCount
            )
            nativeDecodeSessionTokenIds = []
            diagnosticSink?(
                "cmlx_neural_imprint_restore prefix=\(neuralImprintStatus.prefixTokenCount) artifactSHA256=\(neuralImprintStatus.artifactSHA256)"
            )
        } else if hadResidentSession {
            if let reusablePrefix = NativePromptSessionReuse.reusablePrefixMatch(
                cachedTokenIds: nativeDecodeSessionTokenIds,
                promptTokenIds: promptTokens,
                skippableCachedTokenSequences: Self.qwenThinkingSentinelTokenSequences(tokenizer: tokenizer)
            ) {
                let suffix = Array(promptTokens.dropFirst(reusablePrefix.promptTokenLength))
                if suffix.isEmpty {
                    diagnosticSink?("cmlx_lazy_prompt_cache_token_reject reason=empty_suffix reused=\(reusablePrefix.cachedTokenLength) promptReused=\(reusablePrefix.promptTokenLength)")
                } else {
                    cachedTokensReused = reusablePrefix.cachedTokenLength
                    promptCacheHit = true
                    prefillTokens = suffix
                    diagnosticSink?("cmlx_lazy_prompt_cache_token_hit reused=\(cachedTokensReused) promptReused=\(reusablePrefix.promptTokenLength) suffix=\(suffix.count)")
                }
            }
            if !promptCacheHit, nativeUseCmlxLazyTextSuffixPromptCache {
                switch NativePromptSessionReuse.qwenIncrementalSuffix(
                    previousPromptMessages: nativeDecodeSessionPromptMessages,
                    lastAssistantText: nativeDecodeSessionLastAssistantText,
                    currentMessages: messages,
                    enableThinking: parameters.enableThinking,
                    matchPath: "text_suffix"
                ) {
                case let .match(suffixText):
                    let cachedCount = nativeDecodeSessionTokenIds.count
                    guard cachedCount > 0 else {
                        diagnosticSink?("cmlx_lazy_prompt_cache_suffix_reject reason=empty_cache")
                        break
                    }
                    let suffix = tokenizer.encode(text: suffixText, addSpecialTokens: false)
                    if suffix.isEmpty {
                        diagnosticSink?("cmlx_lazy_prompt_cache_suffix_reject reason=text_suffix_empty cached=\(cachedCount) promptTokens=\(promptTokens.count)")
                    } else {
                        cachedTokensReused = cachedCount
                        promptCacheHit = true
                        prefillTokens = suffix
                        diagnosticSink?("cmlx_lazy_prompt_cache_suffix_hit reused=\(cachedTokensReused) suffix=\(suffix.count) promptTokens=\(promptTokens.count)")
                    }
                case let .reject(reason):
                    diagnosticSink?("cmlx_lazy_prompt_cache_suffix_reject reason=\(reason)")
                }
            } else if !promptCacheHit {
                diagnosticSink?("cmlx_lazy_prompt_cache_suffix_reject reason=text_suffix_cache_disabled")
            }
        } else {
            diagnosticSink?("cmlx_lazy_prompt_cache_incremental_reject reason=no_session")
        }

        if neuralImprintStatus == nil && !promptCacheHit {
            try session.reset()
            nativeDecodeSessionTokenIds = []
        }
        let cacheLookupDoneAt = Date()
        let dsrActivationThreshold = dsrPolicies.values
            .map { Int(Float($0.maxSize) * $0.scoreActivationRatio) }
            .min()
        let dsrSummary = Self.dsrPolicySummary(dsrPolicies)
        let useBatchedDSRSuffix = promptCacheHit &&
            dsrActivationThreshold.map { cachedTokensReused >= $0 } == true
        let prefillMode = neuralImprintStatus != nil
            ? "persona_full"
            : (useBatchedDSRSuffix
                ? "incremental_dsr_batch"
                : (promptCacheHit ? "incremental" : "full"))
        diagnosticSink?(
            "cmlx_lazy_prefill_begin tokens=\(prefillTokens.count) mode=\(prefillMode) cached=\(cachedTokensReused) neuralImprintPrefix=\(neuralImprintStatus?.prefixTokenCount ?? 0) dsrThreshold=\(dsrActivationThreshold.map(String.init) ?? "nil") dsrPolicy=\(dsrSummary)"
        )
        if cmlxEvalProfileEnabled {
            try? session.resetEvalProfile()
        }
        try Task.checkCancellation()
        var timingApplyLimitsCount = 0
        var timingApplyLimitsTotalMs = 0.0
        var timingApplyLimitsMaxMs = 0.0
        var timingEmitCount = 0
        var timingEmitTotalMs = 0.0
        var timingEmitMaxMs = 0.0
        var timingNextTokenCount = 0
        var timingNextTokenTotalMs = 0.0
        var timingNextTokenMaxMs = 0.0

        func recordTiming(_ elapsedMs: Double, count: inout Int, totalMs: inout Double, maxMs: inout Double) {
            count += 1
            totalMs += elapsedMs
            maxMs = max(maxMs, elapsedMs)
        }

        func timedApplyCmlxCommandBufferLimits(contextLengthHint: Int) throws {
            guard cmlxTokenTimingEnabled else {
                try applyCmlxCommandBufferLimits(contextLengthHint: contextLengthHint)
                return
            }
            let startedAt = Date()
            try applyCmlxCommandBufferLimits(contextLengthHint: contextLengthHint)
            recordTiming(
                Date().timeIntervalSince(startedAt) * 1000,
                count: &timingApplyLimitsCount,
                totalMs: &timingApplyLimitsTotalMs,
                maxMs: &timingApplyLimitsMaxMs
            )
        }

        func timedNextToken() throws -> Int {
            guard cmlxTokenTimingEnabled else {
                return try session.nextToken()
            }
            let startedAt = Date()
            let token = try session.nextToken()
            recordTiming(
                Date().timeIntervalSince(startedAt) * 1000,
                count: &timingNextTokenCount,
                totalMs: &timingNextTokenTotalMs,
                maxMs: &timingNextTokenMaxMs
            )
            return token
        }

        try timedApplyCmlxCommandBufferLimits(
            contextLengthHint: (neuralImprintStatus?.prefixTokenCount ?? 0) + promptTokens.count
        )
        let useSampledCmlxPath = parameters.temperature > 0
        let cmlxTopK = parameters.topK
        let cmlxTopP = (parameters.topP > 0 && parameters.topP <= 1) ? parameters.topP : 1
        let cmlxMinP = parameters.minP
        var cmlxSamplingRNG = EdgeSeededRandomNumberGenerator(seed: UInt64.random(in: 1...UInt64.max))
        try? session.clearRepetitionPenalty()
        try? session.clearEOSSamplingBias()
        let samplingPenaltyApplier = NativeCmlxSampling.PenaltyApplier(
            parameters: parameters,
            endTokenIds: nativeEndTokenIds,
            setSamplingPenalties: { repetitionPenalty, repetitionTokenIds, presencePenalty, presenceTokenIds, frequencyPenalty, frequencyTokenIds in
                try session.setSamplingPenalties(
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextTokenIds: repetitionTokenIds,
                    presencePenalty: presencePenalty,
                    presenceContextTokenIds: presenceTokenIds,
                    frequencyPenalty: frequencyPenalty,
                    frequencyContextTokenIds: frequencyTokenIds
                )
            },
            setEOSSamplingBias: { tokenIds, suppress, logitPenalty in
                try session.setEOSSamplingBias(
                    tokenIds: tokenIds,
                    suppress: suppress,
                    logitPenalty: logitPenalty
                )
            },
            clearEOSSamplingBias: {
                try session.clearEOSSamplingBias()
            }
        )
        let useSamplingPenalties = useSampledCmlxPath &&
            samplingPenaltyApplier.samplingPenaltiesAreActive
        let useEOSSamplingBias = useSampledCmlxPath &&
            samplingPenaltyApplier.eosSamplingBiasRequested &&
            !nativeEndTokenIds.isEmpty
        if useSamplingPenalties {
            try samplingPenaltyApplier.applySamplingPenalties(
                promptSessionTokenIds: nativeDecodeSessionTokenIds + prefillTokens
            )
        }
        if useEOSSamplingBias {
            try samplingPenaltyApplier.applyEOSSamplingBias(generatedTokenCount: 0)
        }
        if samplingPenaltyApplier.samplingPenaltiesAreActive && !useSampledCmlxPath {
            diagnosticSink?("sampling_penalties_not_supported backend=cmlx_greedy")
        }
        if samplingPenaltyApplier.eosSamplingBiasRequested && !useSampledCmlxPath {
            diagnosticSink?("eos_sampling_bias_not_supported backend=cmlx_greedy")
        }
        defer {
            if useSamplingPenalties { try? session.clearRepetitionPenalty() }
            if useEOSSamplingBias { try? session.clearEOSSamplingBias() }
        }
        var nextTokenID: Int?
        let prefillStartedAt = Date()
        nextTokenID = try runCmlxLazyPrefill(
            session: session,
            tokenIDs: prefillTokens,
            parameters: parameters,
            useSampledPath: useSampledCmlxPath,
            topK: cmlxTopK,
            topP: cmlxTopP,
            minP: cmlxMinP,
            rng: &cmlxSamplingRNG,
            chunkDiagnosticsEnabled: cmlxPrefillChunkDiagnosticsEnabled,
            chunkProfileEnabled: cmlxEvalProfileEnabled && cmlxPrefillChunkProfileEnabled,
            barrierAvailableMB: cmlxPrefillBarrierAvailableMB,
            barrierMaxChunks: cmlxPrefillBarrierMaxChunks
        )
        let prefillDoneAt = Date()
        if promptCacheHit {
            nativeDecodeSessionTokenIds.append(contentsOf: prefillTokens)
        } else {
            nativeDecodeSessionTokenIds = prefillTokens
        }
        let firstTokenAt = prefillDoneAt
        let firstDecodeTokenStartedAt = prefillDoneAt
        var firstDecodeTokenMs: Double?
        if let nextTokenID {
            let mode = useSampledCmlxPath ? "sampled" : "greedy"
            diagnosticSink?("cmlx_lazy_prefill_done firstToken=\(nextTokenID) decodeMode=\(mode)")
        } else {
            diagnosticSink?("cmlx_lazy_prefill_done firstToken=nil")
        }

        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(parameters.maxTokens)
        var stoppedTokenId: Int?
        var emittedText = ""
        var emittedTokenCount = 0
        var buffersToolCallText = onToolCall != nil
        var toolCallDetectionResolved = !buffersToolCallText
        let toolCallDetectionWindow = Self.toolCallDetectionWindow()
        while generatedTokenIds.count < parameters.maxTokens, let tokenID = nextTokenID {
            try Task.checkCancellation()
            if parameters.stopOnEndToken,
               nativeEndTokenIds.contains(tokenID),
               (!useEOSSamplingBias || generatedTokenIds.count >= parameters.minimumGeneratedTokens) {
                stoppedTokenId = tokenID
                let sampleDiagnostics = try? session.lastSampleDiagnostics()
                diagnosticSink?("cmlx_lazy_stop token=\(tokenID) generated=\(generatedTokenIds.count) cached=\(nativeDecodeSessionTokenIds.count) prompt=\(promptTokens.count)\(sampleDiagnostics.map { " \($0)" } ?? "")")
                break
            }
            generatedTokenIds.append(tokenID)
            if generatedTokenIds.count == 1 {
                firstDecodeTokenMs = Date().timeIntervalSince(firstDecodeTokenStartedAt) * 1000
                diagnosticSink?("select_first_token_done token=\(tokenID) backend=cmlx_lazy")
            }
            if buffersToolCallText {
                continuation.yield(
                    GenerateChunk(
                        text: "",
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
                if !toolCallDetectionResolved,
                   generatedTokenIds.count >= toolCallDetectionWindow {
                    toolCallDetectionResolved = true
                    let bufferedText = tokenizer.decode(
                        tokens: generatedTokenIds,
                        skipSpecialTokens: true
                    )
                    if Self.decodedTextContainsToolCallStart(bufferedText) {
                        diagnosticSink?("tool_call_buffer_keep detected=true window=\(toolCallDetectionWindow) generated=\(generatedTokenIds.count)")
                    } else {
                        buffersToolCallText = false
                        emittedText = bufferedText
                        emittedTokenCount = generatedTokenIds.count
                        diagnosticSink?("tool_call_buffer_early_flush window=\(toolCallDetectionWindow) generated=\(generatedTokenIds.count) chars=\(bufferedText.count)")
                        if !bufferedText.isEmpty {
                            continuation.yield(
                                GenerateChunk(
                                    text: bufferedText,
                                    generatedTokenCount: generatedTokenIds.count
                                )
                            )
                        }
                    }
                }
            } else if cmlxTokenTimingEnabled {
                let emitStartedAt = Date()
                emitDecodedTextIfNeeded(
                    tokenizer: tokenizer,
                    generatedTokenIds: generatedTokenIds,
                    emittedText: &emittedText,
                    emittedTokenCount: &emittedTokenCount,
                    continuation: continuation
                )
                recordTiming(
                    Date().timeIntervalSince(emitStartedAt) * 1000,
                    count: &timingEmitCount,
                    totalMs: &timingEmitTotalMs,
                    maxMs: &timingEmitMaxMs
                )
            } else {
                emitDecodedTextIfNeeded(
                    tokenizer: tokenizer,
                    generatedTokenIds: generatedTokenIds,
                    emittedText: &emittedText,
                    emittedTokenCount: &emittedTokenCount,
                    continuation: continuation
                )
            }
            if generatedTokenIds.count < parameters.maxTokens {
                let contextHint = (neuralImprintStatus?.prefixTokenCount ?? 0) +
                    nativeDecodeSessionTokenIds.count +
                    generatedTokenIds.count
                try timedApplyCmlxCommandBufferLimits(contextLengthHint: contextHint)
                if useSampledCmlxPath {
                    if useSamplingPenalties {
                        try samplingPenaltyApplier.applySamplingPenalties(
                            promptSessionTokenIds: nativeDecodeSessionTokenIds,
                            generatedTokenIds: generatedTokenIds
                        )
                    }
                    if useEOSSamplingBias {
                        try samplingPenaltyApplier.applyEOSSamplingBias(
                            generatedTokenCount: generatedTokenIds.count
                        )
                    }
                    nextTokenID = try session.nextSampledToken(
                        temperature: parameters.temperature,
                        topK: cmlxTopK,
                        topP: cmlxTopP,
                        minP: cmlxMinP,
                        seed: cmlxSamplingRNG.next()
                    )
                } else {
                    nextTokenID = try timedNextToken()
                }
            } else {
                nextTokenID = nil
            }
        }
        if buffersToolCallText {
            try await emitFinalDecodedTextHandlingToolCalls(
                tokenizer: tokenizer,
                generatedTokenIds: generatedTokenIds,
                emittedText: &emittedText,
                emittedTokenCount: &emittedTokenCount,
                continuation: continuation,
                onToolCall: onToolCall
            )
        } else if cmlxTokenTimingEnabled {
            let emitStartedAt = Date()
            emitDecodedTextIfNeeded(
                tokenizer: tokenizer,
                generatedTokenIds: generatedTokenIds,
                emittedText: &emittedText,
                emittedTokenCount: &emittedTokenCount,
                continuation: continuation,
                force: true
            )
            recordTiming(
                Date().timeIntervalSince(emitStartedAt) * 1000,
                count: &timingEmitCount,
                totalMs: &timingEmitTotalMs,
                maxMs: &timingEmitMaxMs
            )
        } else {
            emitDecodedTextIfNeeded(
                tokenizer: tokenizer,
                generatedTokenIds: generatedTokenIds,
                emittedText: &emittedText,
                emittedTokenCount: &emittedTokenCount,
                continuation: continuation,
                force: true
            )
        }

        let endedAt = Date()
        let decodeSeconds = max(endedAt.timeIntervalSince(firstTokenAt), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        nativeDecodeSessionTokenIds.append(contentsOf: generatedTokenIds)
        if let stoppedTokenId {
            nativeDecodeSessionTokenIds.append(stoppedTokenId)
        }
        promptCache.update(
            cache: [],
            totalTokenCount: nativeDecodeSessionTokenIds.count,
            tokenPrefix: promptTokens.map { Int32($0) }
        )
        nativeDecodeSessionPromptMessages = messages
        nativeDecodeSessionLastAssistantText = NativePromptSessionReuse.normalizeAssistantText(emittedText)
        turnCounter += 1
        let metrics = InferenceMetrics(
            ttftMs: firstTokenAt.timeIntervalSince(startedAt) * 1000,
            decodeTPS: Double(generatedTokenIds.count) / decodeSeconds,
            deferredLoadMs: nil,
            phaseTimings: InferencePhaseTimings(
                inputPreparationMs: inputPreparationMs,
                planningMs: planningMs,
                warmupMs: sessionReadyAt.timeIntervalSince(sessionSetupStartedAt) * 1000,
                sessionSetupMs: sessionReadyAt.timeIntervalSince(sessionSetupStartedAt) * 1000,
                promptCacheLookupMs: cacheLookupDoneAt.timeIntervalSince(sessionReadyAt) * 1000,
                prefillMs: prefillDoneAt.timeIntervalSince(prefillStartedAt) * 1000,
                fullPrefillMs: promptCacheHit ? nil : prefillDoneAt.timeIntervalSince(prefillStartedAt) * 1000,
                cacheHitSuffixPrefillMs: promptCacheHit ? prefillDoneAt.timeIntervalSince(prefillStartedAt) * 1000 : nil,
                firstTokenSelectionMs: nil,
                firstDecodeTokenMs: firstDecodeTokenMs,
                decodeMs: endedAt.timeIntervalSince(firstTokenAt) * 1000,
                prefillTokenCount: prefillTokens.count,
                prefillMode: prefillMode
            ),
            promptTokenCount: promptTokens.count,
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: (resolvedPolicy?.reasoning ?? "policy bypassed") + " | cmlxLazyDecode=on",
            promptCacheHit: promptCacheHit,
            cachedTokensReused: cachedTokensReused,
            thermalState: ThermalManager().level.rawValue,
            turn: turnCounter
        )
        lastMetrics = metrics
        diagnosticSink?(
            "cmlx_lazy_decode_done generated=\(generatedTokenIds.count) ttftMs=\(Int(firstTokenAt.timeIntervalSince(startedAt) * 1000)) tps=\(String(format: "%.1f", Double(generatedTokenIds.count) / decodeSeconds))"
        )
        diagnosticSink?(
            "cmlx_lazy_phase_timings inputMs=\(Int(inputPreparationMs)) planningMs=\(Int(planningMs)) warmupMs=\(Int(sessionReadyAt.timeIntervalSince(sessionSetupStartedAt) * 1000)) sessionMs=\(Int(sessionReadyAt.timeIntervalSince(sessionSetupStartedAt) * 1000)) cacheMs=\(Int(cacheLookupDoneAt.timeIntervalSince(sessionReadyAt) * 1000)) prefillMs=\(Int(prefillDoneAt.timeIntervalSince(prefillStartedAt) * 1000)) firstDecodeMs=\(firstDecodeTokenMs.map { String(Int($0)) } ?? "nil") decodeMs=\(Int(endedAt.timeIntervalSince(firstTokenAt) * 1000)) prefillTokens=\(prefillTokens.count) mode=\(prefillMode)"
        )
        if cmlxTokenTimingEnabled {
            func timingField(_ name: String, count: Int, totalMs: Double, maxMs: Double) -> String {
                let averageMs = count > 0 ? totalMs / Double(count) : 0
                return "\(name)=count:\(count),totalMs:\(String(format: "%.2f", totalMs)),avgMs:\(String(format: "%.4f", averageMs)),maxMs:\(String(format: "%.4f", maxMs))"
            }
            diagnosticSink?(
                "cmlx_lazy_token_timing "
                    + timingField("limits", count: timingApplyLimitsCount, totalMs: timingApplyLimitsTotalMs, maxMs: timingApplyLimitsMaxMs)
                    + " "
                    + timingField("emit", count: timingEmitCount, totalMs: timingEmitTotalMs, maxMs: timingEmitMaxMs)
                    + " "
                    + timingField("nextToken", count: timingNextTokenCount, totalMs: timingNextTokenTotalMs, maxMs: timingNextTokenMaxMs)
            )
        }
        if cmlxEvalProfileEnabled, let evalProfile = try? session.evalProfileSummary() {
            diagnosticSink?("cmlx_eval_profile \(evalProfile)")
        }
        recordOnlineCalibration(metrics: metrics)
    }

    private func applyOnlineCalibrationBeforeTurn(contextLengthHint: Int) {
        guard let calibrator = onlineCalibrator,
              let plan = currentPlan else {
            return
        }
        let overrides = calibrator.currentOverrides
        guard overrides != activeOnlineCalibrationOverrides else { return }
        let adjusted = plan.applying(overrides.nativeLoadOptions)
        currentPlan = adjusted
        memoryPolicy = MemoryBudgetPlanner.toKVPolicy(adjusted)
        _ = NativeRuntimeBridge.applyMetalConfiguration(
            NativeRuntimeBridge.metalConfiguration(for: adjusted, contextLengthHint: contextLengthHint)
        )
        nativeCmlxLimitState.reset()
        activeOnlineCalibrationOverrides = overrides
        diagnosticSink?(
            "online_calibration_apply maxOps=\(overrides.maxOpsPerBuffer) prefill=\(overrides.prefillStepSize) dynFloor=\(overrides.dynamicOpsFloor) ctx=\(contextLengthHint)"
        )
    }

    private func recordOnlineCalibration(metrics: InferenceMetrics) {
        guard let calibrator = onlineCalibrator else { return }
        let memory = DeviceProfile.captureMemorySnapshot()
        let sample = OnlineCalibrator.Sample(
            turnNumber: metrics.turn,
            tps: metrics.decodeTPS,
            contextLength: metrics.promptTokenCount + metrics.generationTokenCount,
            thermalState: metrics.thermalState,
            availableMemoryMB: memory.availableMB
        )
        guard (2...4).contains(sample.turnNumber) else {
            diagnosticSink?(
                "online_calibration_skip turn=\(sample.turnNumber) tps=\(String(format: "%.2f", sample.tps)) ctx=\(sample.contextLength)"
            )
            return
        }
        if let decision = calibrator.record(sample) {
            diagnosticSink?(
                "online_calibration_decision avgTps=\(String(format: "%.2f", decision.averageTps)) maxOps=\(decision.overrides.maxOpsPerBuffer) prefill=\(decision.overrides.prefillStepSize) dynFloor=\(decision.overrides.dynamicOpsFloor) reason=\(decision.reason)"
            )
        } else {
            diagnosticSink?(
                "online_calibration_record turn=\(sample.turnNumber) tps=\(String(format: "%.2f", sample.tps)) ctx=\(sample.contextLength) availMB=\(sample.availableMemoryMB)"
            )
        }
    }

    private func runCmlxLazyPrefill(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        parameters: EdgeGenerateParameters,
        useSampledPath: Bool,
        topK: Int?,
        topP: Float,
        minP: Float,
        rng: inout EdgeSeededRandomNumberGenerator,
        chunkDiagnosticsEnabled: Bool = false,
        chunkProfileEnabled: Bool = false,
        barrierAvailableMB: Int = 0,
        barrierMaxChunks: Int = 0
    ) throws -> Int {
        let chunkSize = max(1, parameters.prefillStepSize)
        func chunkMemoryFields(_ memory: DeviceProfile.MemorySnapshot) -> String {
            "footprintMB=\(memory.footprintMB) availableMB=\(memory.availableMB) jetsamLimitMB=\(memory.jetsamLimitMB)"
        }
        func logChunkProbe(
            phase: String,
            index: Int,
            tokens: Int,
            offset: Int,
            final: Bool,
            async: Bool,
            elapsedMs: Double? = nil
        ) {
            guard chunkDiagnosticsEnabled else { return }
            let memory = DeviceProfile.captureMemorySnapshot()
            var fields = [
                "phase=\(phase)",
                "index=\(index)",
                "tokens=\(tokens)",
                "offset=\(offset)",
                "final=\(final)",
                "async=\(async)",
                chunkMemoryFields(memory),
            ]
            if let elapsedMs {
                fields.append("elapsedMs=\(String(format: "%.2f", elapsedMs))")
            }
            diagnosticSink?("cmlx_lazy_prefill_chunk_probe \(fields.joined(separator: " "))")
            guard chunkProfileEnabled, phase == "after",
                  let profile = try? session.evalProfileSummary()
            else { return }
            diagnosticSink?(
                "cmlx_lazy_prefill_chunk_profile index=\(index) final=\(final) \(profile)"
            )
        }
        func maybeApplyPrefillBarrier(index: Int, chunksSinceBarrier: Int) throws -> Bool {
            guard barrierAvailableMB > 0 || barrierMaxChunks > 0 else {
                return false
            }
            let before = DeviceProfile.captureMemorySnapshot()
            let reason: String?
            if barrierAvailableMB > 0, before.availableMB < barrierAvailableMB {
                reason = "low_available"
            } else if barrierMaxChunks > 0, chunksSinceBarrier >= barrierMaxChunks {
                reason = "max_chunks"
            } else {
                reason = nil
            }
            guard let reason else {
                return false
            }

            let barrierStartedAt = Date()
            try session.synchronize()
            let elapsedMs = Date().timeIntervalSince(barrierStartedAt) * 1000
            let after = DeviceProfile.captureMemorySnapshot()
            diagnosticSink?(
                "cmlx_lazy_prefill_barrier index=\(index) reason=\(reason) chunksSinceBarrier=\(chunksSinceBarrier) beforeFootprintMB=\(before.footprintMB) beforeAvailableMB=\(before.availableMB) afterFootprintMB=\(after.footprintMB) afterAvailableMB=\(after.availableMB) elapsedMs=\(String(format: "%.2f", elapsedMs))"
            )
            return true
        }
        guard tokenIDs.count > chunkSize else {
            if useSampledPath {
                try session.prefillSampledAsync(
                    tokenIDs: tokenIDs,
                    temperature: parameters.temperature,
                    topK: topK,
                    topP: topP,
                    minP: minP,
                    seed: rng.next()
                )
                return try session.nextSampledToken(
                    temperature: parameters.temperature,
                    topK: topK,
                    topP: topP,
                    minP: minP,
                    seed: rng.next()
                )
            }
            try session.prefillAsync(tokenIDs: tokenIDs)
            return try session.nextToken()
        }

        let chunkCount = Int(ceil(Double(tokenIDs.count) / Double(chunkSize)))
        diagnosticSink?(
            "cmlx_lazy_prefill_chunked total=\(tokenIDs.count) step=\(chunkSize) chunks=\(chunkCount) barrierAvailableMB=\(barrierAvailableMB) barrierMaxChunks=\(barrierMaxChunks)"
        )
        var offset = 0
        var chunkIndex = 0
        var chunksSinceBarrier = 0
        while offset < tokenIDs.count {
            let end = min(offset + chunkSize, tokenIDs.count)
            let chunk = Array(tokenIDs[offset..<end])
            let isLast = end == tokenIDs.count
            chunkIndex += 1
            logChunkProbe(
                phase: "before",
                index: chunkIndex,
                tokens: chunk.count,
                offset: offset,
                final: isLast,
                async: true
            )
            let chunkStartedAt = Date()
            if isLast {
                if useSampledPath {
                    try session.prefillSampledAsync(
                        tokenIDs: chunk,
                        temperature: parameters.temperature,
                        topK: topK,
                        topP: topP,
                        minP: minP,
                        seed: rng.next()
                    )
                    logChunkProbe(
                        phase: "after",
                        index: chunkIndex,
                        tokens: chunk.count,
                        offset: offset,
                        final: true,
                        async: true,
                        elapsedMs: Date().timeIntervalSince(chunkStartedAt) * 1000
                    )
                    diagnosticSink?(
                        "cmlx_lazy_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=true"
                    )
                    return try session.nextSampledToken(
                        temperature: parameters.temperature,
                        topK: topK,
                        topP: topP,
                        minP: minP,
                        seed: rng.next()
                    )
                }
                try session.prefillAsync(tokenIDs: chunk)
                logChunkProbe(
                    phase: "after",
                    index: chunkIndex,
                    tokens: chunk.count,
                    offset: offset,
                    final: true,
                    async: true,
                    elapsedMs: Date().timeIntervalSince(chunkStartedAt) * 1000
                )
                diagnosticSink?(
                    "cmlx_lazy_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=true"
                )
                return try session.nextToken()
            }

            try session.prefillAsync(tokenIDs: chunk)
            logChunkProbe(
                phase: "after",
                index: chunkIndex,
                tokens: chunk.count,
                offset: offset,
                final: false,
                async: true,
                elapsedMs: Date().timeIntervalSince(chunkStartedAt) * 1000
            )
            diagnosticSink?(
                "cmlx_lazy_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=false async=true"
            )
            chunksSinceBarrier += 1
            if try maybeApplyPrefillBarrier(
                index: chunkIndex,
                chunksSinceBarrier: chunksSinceBarrier
            ) {
                chunksSinceBarrier = 0
            }
            offset = end
        }

        throw QwenHybridModelReferenceError.emptyTokenIds
    }

    private static func canUpdateResidentDSRPoliciesInPlace(
        requested: [Int: QwenDSRKVCachePolicy],
        cached: [Int: QwenDSRKVCachePolicy]
    ) -> Bool {
        guard !requested.isEmpty, !cached.isEmpty, requested != cached else {
            return false
        }
        guard Set(requested.keys) == Set(cached.keys) else {
            return false
        }
        return requested.allSatisfy { layerIndex, requestedPolicy in
            guard let cachedPolicy = cached[layerIndex] else {
                return false
            }
            return requestedPolicy.maxSize == cachedPolicy.maxSize &&
                requestedPolicy.heavyBudget == cachedPolicy.heavyBudget &&
                requestedPolicy.recentBudget == cachedPolicy.recentBudget &&
                requestedPolicy.sinkSize == cachedPolicy.sinkSize
        }
    }

    private static func dsrPolicySummary(_ policies: [Int: QwenDSRKVCachePolicy]) -> String {
        let first = policies.sorted { $0.key < $1.key }.first?.value
        guard let first else {
            return "none"
        }
        return "layers=\(policies.count),max=\(first.maxSize),heavy=\(first.heavyBudget),recent=\(first.recentBudget),evict=\(first.evictionInterval)"
    }

    static func cmlxAttentionCacheQuantization(
        parameters: EdgeGenerateParameters,
        dsrPolicies: [Int: QwenDSRKVCachePolicy]
    ) -> NativeCmlxAttentionCacheQuantization? {
        guard parameters.quantizedKVStart == 0 else { return nil }
        let bits = parameters.kvBits ?? 4
        guard bits > 0 else { return nil }
        return NativeCmlxAttentionCacheQuantization(
            groupSize: max(1, parameters.kvGroupSize),
            bits: bits
        )
    }

    private func prepareNativeDecodeSession(
        messages: [ChatMessage],
        promptTokens: [Int],
        kvCapacity: Int,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        skippableCachedTokenSequences: [[Int]],
        tokenizer: Tokenizer,
        enableThinking: Bool,
        toolsAreEmpty: Bool,
        model: QwenHybridModelReference,
        runtime: EdgeMetalRuntime,
        executor: MetalKernelExecutor,
        architecture: QwenHybridArchitecture,
        useGreedyTokenPath: Bool
    ) throws -> (session: QwenGreedyDecodeSession, cachedTokensReused: Int) {
        guard let session = nativeDecodeSession else {
            diagnosticSink?("prompt_cache_reject reason=no_session requestedKvCapacity=\(kvCapacity)")
            return try makeNativeDecodeSession(
                promptTokens: promptTokens,
                kvCapacity: kvCapacity,
                dsrPolicies: dsrPolicies,
                model: model,
                runtime: runtime,
                executor: executor,
                architecture: architecture,
                useGreedyTokenPath: useGreedyTokenPath
            )
        }
        guard kvCapacity <= nativeDecodeSessionCapacity else {
            diagnosticSink?("prompt_cache_reject reason=capacity requested=\(kvCapacity) cached=\(nativeDecodeSessionCapacity)")
            return try makeNativeDecodeSession(
                promptTokens: promptTokens,
                kvCapacity: kvCapacity,
                dsrPolicies: dsrPolicies,
                model: model,
                runtime: runtime,
                executor: executor,
                architecture: architecture,
                useGreedyTokenPath: useGreedyTokenPath
            )
        }
        guard dsrPolicies == nativeDecodeSessionDSRPolicies else {
            diagnosticSink?("prompt_cache_reject reason=dsr_policy_changed requestedLayers=\(dsrPolicies.count) cachedLayers=\(nativeDecodeSessionDSRPolicies.count)")
            return try makeNativeDecodeSession(
                promptTokens: promptTokens,
                kvCapacity: kvCapacity,
                dsrPolicies: dsrPolicies,
                model: model,
                runtime: runtime,
                executor: executor,
                architecture: architecture,
                useGreedyTokenPath: useGreedyTokenPath
            )
        }

        if toolsAreEmpty {
            switch NativePromptSessionReuse.qwenIncrementalSuffix(
                previousPromptMessages: nativeDecodeSessionPromptMessages,
                lastAssistantText: nativeDecodeSessionLastAssistantText,
                currentMessages: messages,
                enableThinking: enableThinking,
                matchPath: "standard_text_suffix"
            ) {
            case let .match(suffixText):
                let suffix = tokenizer.encode(text: suffixText, addSpecialTokens: false)
                guard !suffix.isEmpty else {
                    throw QwenHybridModelReferenceError.emptyTokenIds
                }
                let reusedCachedTokens = nativeDecodeSessionTokenIds.count
                diagnosticSink?("prompt_cache_incremental_hit reused=\(reusedCachedTokens) suffix=\(suffix.count) kvCapacity=\(nativeDecodeSessionCapacity)")
                diagnosticSink?("prompt_suffix_advance_begin tokens=\(suffix.count)")
                if useGreedyTokenPath {
                    try session.advanceGreedy(tokenIds: suffix)
                } else {
                    try session.advance(tokenIds: suffix)
                }
                diagnosticSink?("prompt_suffix_advance_done")
                nativeDecodeSessionTokenIds.append(contentsOf: suffix)
                return (session, reusedCachedTokens)
            case let .reject(reason):
                diagnosticSink?("prompt_cache_incremental_reject reason=\(reason)")
            }
        } else {
            diagnosticSink?("prompt_cache_incremental_reject reason=tools_present")
        }

        if let reusablePrefix = NativePromptSessionReuse.reusablePrefixMatch(
            cachedTokenIds: nativeDecodeSessionTokenIds,
            promptTokenIds: promptTokens,
            skippableCachedTokenSequences: skippableCachedTokenSequences
        ) {
            let suffix = Array(promptTokens.dropFirst(reusablePrefix.promptTokenLength))
            diagnosticSink?("prompt_cache_hit reused=\(reusablePrefix.cachedTokenLength) suffix=\(suffix.count) kvCapacity=\(nativeDecodeSessionCapacity)")
            if !suffix.isEmpty {
                diagnosticSink?("prompt_suffix_advance_begin tokens=\(suffix.count)")
                if useGreedyTokenPath {
                    try session.advanceGreedy(tokenIds: suffix)
                } else {
                    try session.advance(tokenIds: suffix)
                }
                diagnosticSink?("prompt_suffix_advance_done")
            }
            let reusedCachedTokens = Array(nativeDecodeSessionTokenIds.prefix(reusablePrefix.cachedTokenLength))
            nativeDecodeSessionTokenIds = reusedCachedTokens + suffix
            return (session, reusablePrefix.cachedTokenLength)
        }
        diagnosticSink?("prompt_cache_reject reason=prefix_mismatch cachedTokens=\(nativeDecodeSessionTokenIds.count) promptTokens=\(promptTokens.count)")

        return try makeNativeDecodeSession(
            promptTokens: promptTokens,
            kvCapacity: kvCapacity,
            dsrPolicies: dsrPolicies,
            model: model,
            runtime: runtime,
            executor: executor,
            architecture: architecture,
            useGreedyTokenPath: useGreedyTokenPath
        )
    }

    private func makeNativeDecodeSession(
        promptTokens: [Int],
        kvCapacity: Int,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        model: QwenHybridModelReference,
        runtime: EdgeMetalRuntime,
        executor: MetalKernelExecutor,
        architecture: QwenHybridArchitecture,
        useGreedyTokenPath: Bool
    ) throws -> (session: QwenGreedyDecodeSession, cachedTokensReused: Int) {
        diagnosticSink?("session_init_begin kvCapacity=\(kvCapacity) dsrLayers=\(dsrPolicies.count) promptTokens=\(promptTokens.count)")
        let session = QwenGreedyDecodeSession(
            model: model,
            caches: try QwenHybridDecoderCaches(
                architecture: architecture,
                runtime: runtime,
                kvCapacity: kvCapacity,
                dsrPolicies: dsrPolicies
            ),
            executor: executor,
            diagnosticSink: diagnosticSink
        )
        diagnosticSink?("session_init_done")
        diagnosticSink?("prefill_begin tokens=\(promptTokens.count)")
        if useGreedyTokenPath {
            try session.prefillGreedy(promptTokenIds: promptTokens)
        } else {
            try session.prefill(promptTokenIds: promptTokens)
        }
        diagnosticSink?("prefill_done")
        nativeDecodeSession = session
        nativeDecodeSessionTokenIds = promptTokens
        nativeDecodeSessionCapacity = kvCapacity
        nativeDecodeSessionDSRPolicies = dsrPolicies
        return (session, 0)
    }

    static func qwenThinkingSentinelTokenSequences(tokenizer: Tokenizer) -> [[Int]] {
        [
            tokenizer.encode(text: "<think>\n\n</think>\n\n", addSpecialTokens: false),
            tokenizer.encode(text: "<think>\n", addSpecialTokens: false),
        ].filter { !$0.isEmpty }
    }

    private func clearNativeGreedyDecodeSession() {
        nativeDecodeSession = nil
        nativeDecodeSessionTokenIds = []
        nativeDecodeSessionCapacity = 0
        nativeDecodeSessionDSRPolicies = [:]
        nativeDecodeSessionPromptMessages = []
        nativeDecodeSessionLastAssistantText = ""
    }

    static func defaultEndTokenIds(
        tokenizer: Tokenizer,
        modelDirectory: URL? = nil
    ) -> Set<Int> {
        defaultEndTokenIds(
            eosTokenId: tokenizer.eosTokenId,
            tokenIdForToken: tokenizer.convertTokenToId,
            modelDirectory: modelDirectory
        )
    }

    static func defaultEndTokenIds(
        eosTokenId: Int?,
        tokenIdForToken: (String) -> Int?,
        modelDirectory: URL? = nil
    ) -> Set<Int> {
        var ids = Set<Int>()
        if let modelDirectory {
            ids.formUnion(generationConfigEndTokenIds(modelDirectory: modelDirectory))
        }
        if let eos = eosTokenId {
            ids.insert(eos)
        }
        if let imEnd = tokenIdForToken("<|im_end|>") {
            ids.insert(imEnd)
        }
        return ids
    }

    static func generationConfigEndTokenIds(modelDirectory: URL) -> Set<Int> {
        let url = modelDirectory.appendingPathComponent("generation_config.json")
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(GenerationConfig.self, from: data)
        else {
            return []
        }
        return Set(config.eosTokenID?.values ?? [])
    }

    private struct GenerationConfig: Decodable {
        var eosTokenID: TokenIDValue?

        private enum CodingKeys: String, CodingKey {
            case eosTokenID = "eos_token_id"
        }
    }

    private enum TokenIDValue: Decodable {
        case single(Int)
        case many([Int])

        var values: [Int] {
            switch self {
            case .single(let value):
                [value]
            case .many(let values):
                values
            }
        }

        init(from decoder: Swift.Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(Int.self) {
                self = .single(value)
                return
            }
            self = .many(try container.decode([Int].self))
        }
    }

    static func makeDSRPolicies(
        parameters: EdgeGenerateParameters,
        architecture: QwenHybridArchitecture
    ) throws -> [Int: QwenDSRKVCachePolicy] {
        guard parameters.useDSR,
              let maxSize = parameters.dsrMaxCritical
        else {
            return [:]
        }
        return try QwenDSRKVCachePolicy.layerAwarePolicies(
            for: architecture,
            maxSize: maxSize,
            heavyBudget: parameters.dsrHeavyBudget,
            recentBudget: parameters.dsrRecentBudget,
            scene: QwenDSRKVCacheScene(rawValue: parameters.dsrScene.rawValue) ?? .chat,
            evictionInterval: parameters.dsrEvictionInterval
        )
    }

    static func dsrEvictionIntervalEnvironmentOverride(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        for key in ["EDGE_DSR_EVICTION_INTERVAL", "EDGE_CMLX_DSR_EVICTION_INTERVAL"] {
            guard let rawValue = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !rawValue.isEmpty,
                  let value = Int(rawValue),
                  value > 0
            else {
                continue
            }
            return value
        }
        return nil
    }

    static func kvBitsEnvironmentOverride(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        for key in ["EDGE_KV_BITS", "EDGE_CMLX_KV_BITS"] {
            guard let rawValue = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !rawValue.isEmpty
            else {
                continue
            }
            switch rawValue.lowercased() {
            case "off", "none", "false", "disabled":
                return 0
            default:
                guard let value = Int(rawValue), value >= 0 else {
                    continue
                }
                return value
            }
        }
        return nil
    }

    static let defaultToolCallDetectionWindow = 50

    static func toolCallDetectionWindow(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        for key in ["EDGE_TOOL_CALL_DETECTION_WINDOW", "EDGE_CMLX_TOOL_CALL_DETECTION_WINDOW"] {
            guard let rawValue = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !rawValue.isEmpty,
                  let value = Int(rawValue),
                  value > 0
            else {
                continue
            }
            return min(value, 512)
        }
        return defaultToolCallDetectionWindow
    }

    static func decodedTextContainsToolCallStart(_ text: String) -> Bool {
        text.range(of: "<tool_call", options: [.caseInsensitive]) != nil
    }

    static func kvCapacity(
        promptTokenCount: Int,
        maxTokenCount: Int,
        parameters: EdgeGenerateParameters,
        architecture: QwenHybridArchitecture
    ) -> Int {
        let requested = max(1, promptTokenCount + maxTokenCount + 1)
        if parameters.useDSR, let dsrMaxCritical = parameters.dsrMaxCritical {
            return min(
                architecture.contextLength,
                max(requested, dsrMaxCritical + parameters.dsrEvictionInterval + 1)
            )
        }
        if let maxKVSize = parameters.maxKVSize {
            return min(architecture.contextLength, max(requested, maxKVSize))
        }
        return min(architecture.contextLength, requested)
    }
}
