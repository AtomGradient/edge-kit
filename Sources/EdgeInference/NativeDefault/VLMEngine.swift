// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import CoreImage
import EdgeEngine
import Foundation
import Tokenizers

public final class VLMEngine: ObservableObject {
    private static let eosSamplingLogitPenalty: Float = 20

    @Published public private(set) var state: EngineState = .idle
    @Published public private(set) var downloadProgress: Double = 0
    @Published public private(set) var lastPolicy: InferencePolicy.Resolved?
    @Published public private(set) var lastMetrics: InferenceMetrics?

    public private(set) var memoryPolicy: KVCacheMemoryPolicy?
    public private(set) var currentPlan: MemoryBudgetPlanner.Plan?
    public private(set) var archInfo: ModelArchInfo?
    public private(set) var visionOffloaded: Bool = false
    public let promptCache = PromptCacheManager()

    private var modelDirectory: URL?
    private var nativeContainer: QwenVLMNativeContainer?
    private var nativeTokenizer: Tokenizer?
    private var nativeEndTokenIds: Set<Int> = []
    private var turnCounter = 0
    private var nativeVLMCmlxSessionLoaded = false
    private var nativeVLMCmlxTokenIds: [Int] = []
    private var nativeVLMCmlxPromptMessages: [ChatMessage] = []
    private var nativeVLMCmlxLastAssistantText = ""
    private var nativeVLMCmlxContainsMediaContext = false
    private var nativeVLMCmlxDSRPolicies: [Int: QwenDSRKVCachePolicy] = [:]
    private var nativeVLMCmlxAttentionCacheQuantization: NativeCmlxAttentionCacheQuantization?
    private var nativeVLMCmlxFrogJumpLayerMask: UInt64 = 0
    private var nativeVLMCmlxAttentionCacheLimit: Int?
    public var diagnosticSink: ((String) -> Void)?

    private enum NativeVLMImageInput {
        case url(URL)
        case ciImage(CIImage)
    }

    struct NativeVLMTextDeltaPrefillPlan: Equatable {
        let cachedTokensReused: Int
        let suffixText: String
    }

    enum NativeVLMTextDeltaPrefillPlanResult: Equatable {
        case success(NativeVLMTextDeltaPrefillPlan)
        case failure(String)
    }

    enum NativeVLMTextDeltaPrefillPlanner {
        static func makePlan(
            cachedTokenCount: Int,
            previousPromptMessages: [ChatMessage],
            lastAssistantText: String,
            currentMessages: [ChatMessage],
            enableThinking: Bool
        ) -> NativeVLMTextDeltaPrefillPlanResult {
            guard cachedTokenCount > 0 else {
                return .failure("empty_cached_tokens")
            }
            switch NativePromptSessionReuse.qwenIncrementalSuffix(
                previousPromptMessages: previousPromptMessages,
                lastAssistantText: lastAssistantText,
                currentMessages: currentMessages,
                enableThinking: enableThinking,
                matchPath: "vlm_text_suffix"
            ) {
            case .match(let suffixText):
                return .success(NativeVLMTextDeltaPrefillPlan(
                    cachedTokensReused: cachedTokenCount,
                    suffixText: suffixText
                ))
            case .reject(let reason):
                return .failure(reason)
            }
        }
    }

    public init() {}

    public func loadLocal(directory: URL, onProgress: ((Double) -> Void)? = nil) async throws {
        guard state != .loading else { return }
        state = .loading
        downloadProgress = 0
        onProgress?(0)
        do {
            let index = try QwenVLMModelBundleIndex.load(from: directory)
            modelDirectory = directory
            archInfo = ModelArchInfo.load(from: directory)
            let benchmark = DeviceBenchmark.cachedOrCurrent()
            let profile = benchmark.profile
            let plan = MemoryBudgetPlanner.plan(
                profile: profile,
                modelSizeGB: LLMEngine.estimateModelSizeGB(directory: directory),
                measuredBandwidthGBs: benchmark.measuredBandwidthGBs
            )
            currentPlan = plan
            memoryPolicy = MemoryBudgetPlanner.toKVPolicy(plan)
            _ = NativeRuntimeBridge.applyMetalConfiguration(
                NativeRuntimeBridge.metalConfiguration(for: plan)
            )
            let runtime = try EdgeMetalRuntime(
                configuration: EdgeEngineMetalConfigurationStore.shared.currentConfiguration
            )
            let executor = try MetalKernelExecutor(runtime: runtime)
            let tokenizer = try await AutoTokenizer.from(
                modelFolder: directory,
                strict: false
            )
            nativeContainer = QwenVLMNativeContainer(
                index: index,
                runtime: runtime,
                executor: executor
            )
            nativeTokenizer = tokenizer
            nativeEndTokenIds = LLMEngine.defaultEndTokenIds(
                tokenizer: tokenizer,
                modelDirectory: directory
            )
            downloadProgress = 1
            onProgress?(1)
            state = .ready
        } catch {
            unload()
            throw EdgeRuntimeError.loadFailed(error.localizedDescription)
        }
    }

    public func load(config: ModelConfig, onProgress: ((Double) -> Void)? = nil) async throws {
        onProgress?(0)
        throw EdgeRuntimeError.unsupportedFeature(
            "Remote VLM download is not part of the native default build; use ODR or loadLocal(directory:)."
        )
    }

    public func generate(
        messages: [ChatMessage],
        images: [URL] = [],
        tools: [ToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters = .default
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        if !images.isEmpty {
            return generateNativeImage(
                messages: messages,
                inputs: images.map(NativeVLMImageInput.url),
                tools: tools,
                parameters: parameters
            )
        }
        return generateNativeText(
            messages: messages,
            tools: tools,
            parameters: parameters
        )
    }

    public func generate(
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [ToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters = .default
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        if !ciImages.isEmpty {
            return generateNativeImage(
                messages: messages,
                inputs: ciImages.map(NativeVLMImageInput.ciImage),
                tools: tools,
                parameters: parameters
            )
        }
        return generateNativeText(
            messages: messages,
            tools: tools,
            parameters: parameters
        )
    }

    public func generateStream(
        messages: [ChatMessage],
        images: [URL] = [],
        parameters: EdgeGenerateParameters = .default
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        generate(messages: messages, images: images, parameters: parameters)
    }

    public func tokenize(_ text: String) async throws -> [Int] {
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native VLM tokenizer is not initialized")
        }
        return tokenizer.encode(text: text)
    }

    public func captureHiddenStates(tokens: [Int], targetLayer: Int) async throws -> [Float] {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard !tokens.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM container is not initialized")
        }
        let architecture = container.index.languageIndex.architecture
        guard targetLayer >= 0, targetLayer < architecture.layerCount else {
            throw QwenHybridModelReferenceError.invalidCaptureLayer(targetLayer)
        }

        try prepareNativeVLMCmlxSession(
            container: container,
            dsrPolicies: [:],
            attentionQuantization: nil,
            frogJumpMask: 0,
            attentionCacheLimit: nil
        )

        let session = try container.resetCmlxDecoder()

        defer { markCmlxSessionCleanAfterCapture() }

        emitVLMDiagnostic("vlm_capture_hidden_begin tokens=\(tokens.count) layer=\(targetLayer)")
        let captured = try session.captureLastHidden(
            tokenIDs: tokens,
            targetLayer: targetLayer
        )
        emitVLMDiagnostic(
            "vlm_capture_hidden_done tokens=\(tokens.count) layer=\(targetLayer) hidden=\(captured.count)"
        )
        return captured
    }

    public func captureRouteMatrixEmbedding(
        text: String,
        encoder: RouteRouterEncoderSpec
    ) async throws -> [Float] {
        guard encoder.kind == "base_model_last_hidden" else {
            throw EdgeRuntimeError.loadFailed(
                "route matrix encoder kind unsupported: \(encoder.kind)"
            )
        }
        guard encoder.pooling == "mean_excluding_special" else {
            throw EdgeRuntimeError.loadFailed(
                "route matrix encoder pooling unsupported: \(encoder.pooling)"
            )
        }
        throw EdgeRuntimeError.unsupportedFeature(
            "Native route-matrix embedding capture is moving into edge-engine."
        )
    }

    public func ensureWeightsLoaded() async throws {
        guard state == .ready || state == .generating else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM container is not initialized")
        }
        if !container.isDecoderLoaded {
            try container.loadDecoderWeights()
        }
    }

    public func unload() {
        state = .idle
        modelDirectory = nil
        _ = nativeContainer?.unloadDecoderWeights()
        nativeContainer?.unloadCmlxDecoderWeights()
        nativeContainer?.unloadCmlxVisionWeights()
        nativeContainer = nil
        nativeTokenizer = nil
        nativeEndTokenIds = []
        resetNativeVLMCmlxLedger()
        turnCounter = 0
        archInfo = nil
        currentPlan = nil
        memoryPolicy = nil
        visionOffloaded = false
        lastPolicy = nil
        lastMetrics = nil
        promptCache.clear()
        downloadProgress = 0
    }

    private func emitVLMDiagnostic(_ message: String) {
        diagnosticSink?(message)
        NSLog("[VLM] %@", message)
    }

    private func resetNativeVLMCmlxLedger() {
        nativeVLMCmlxSessionLoaded = false
        nativeVLMCmlxTokenIds = []
        nativeVLMCmlxPromptMessages = []
        nativeVLMCmlxLastAssistantText = ""
        nativeVLMCmlxContainsMediaContext = false
        nativeVLMCmlxDSRPolicies = [:]
        nativeVLMCmlxAttentionCacheQuantization = nil
        nativeVLMCmlxFrogJumpLayerMask = 0
        nativeVLMCmlxAttentionCacheLimit = nil
    }

    /// Clear prompt-cache ledger and re-mark the Cmlx session as alive with
    /// clean base config after RPP capture (which resets KV but keeps session).
    private func markCmlxSessionCleanAfterCapture() {
        resetNativeVLMCmlxLedger()
        nativeVLMCmlxSessionLoaded = true
        nativeVLMCmlxDSRPolicies = [:]
        nativeVLMCmlxAttentionCacheQuantization = nil
        nativeVLMCmlxFrogJumpLayerMask = 0
        nativeVLMCmlxAttentionCacheLimit = nil
    }

    private func releaseNativeVLMCmlxSession(
        container: QwenVLMNativeContainer,
        reason: String
    ) {
        if container.isCmlxDecoderLoaded {
            emitVLMDiagnostic("vlm_cmlx_session_release reason=\(reason)")
            container.unloadCmlxDecoderWeights()
        }
        resetNativeVLMCmlxLedger()
    }

    private func nativeVLMCmlxSessionMatches(
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        attentionQuantization: NativeCmlxAttentionCacheQuantization?,
        frogJumpMask: UInt64,
        attentionCacheLimit: Int?
    ) -> Bool {
        nativeVLMCmlxSessionLoaded &&
            nativeVLMCmlxDSRPolicies == dsrPolicies &&
            nativeVLMCmlxAttentionCacheQuantization == attentionQuantization &&
            nativeVLMCmlxFrogJumpLayerMask == frogJumpMask &&
            nativeVLMCmlxAttentionCacheLimit == attentionCacheLimit
    }

    private func prepareNativeVLMCmlxSession(
        container: QwenVLMNativeContainer,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        attentionQuantization: NativeCmlxAttentionCacheQuantization?,
        frogJumpMask: UInt64,
        attentionCacheLimit: Int?
    ) throws {
        if container.isCmlxDecoderLoaded,
           !nativeVLMCmlxSessionMatches(
               dsrPolicies: dsrPolicies,
               attentionQuantization: attentionQuantization,
               frogJumpMask: frogJumpMask,
               attentionCacheLimit: attentionCacheLimit
           ) {
            releaseNativeVLMCmlxSession(
                container: container,
                reason: "config_changed"
            )
        }

        if container.isDecoderLoaded {
            _ = container.unloadDecoderWeights()
            emitVLMDiagnostic("vlm_swift_decoder_unload reason=cmlx_session_prepare")
        }

        if !container.isCmlxDecoderLoaded {
            emitVLMDiagnostic(
                "vlm_cmlx_session_init_begin dsrLayers=\(dsrPolicies.count) kv=\(Self.cmlxAttentionCacheQuantizationSummary(attentionQuantization)) frog=0x\(String(frogJumpMask, radix: 16)) limit=\(attentionCacheLimit.map(String.init) ?? "nil")"
            )
            try container.loadCmlxDecoderWeights(
                attentionCacheLimit: attentionCacheLimit,
                dsrPolicies: dsrPolicies,
                attentionCacheQuantizationGroupSize: attentionQuantization?.groupSize,
                attentionCacheQuantizationBits: attentionQuantization?.bits,
                frogJumpLayerMask: frogJumpMask
            )
            nativeVLMCmlxSessionLoaded = true
            nativeVLMCmlxDSRPolicies = dsrPolicies
            nativeVLMCmlxAttentionCacheQuantization = attentionQuantization
            nativeVLMCmlxFrogJumpLayerMask = frogJumpMask
            nativeVLMCmlxAttentionCacheLimit = attentionCacheLimit
            emitVLMDiagnostic("vlm_cmlx_session_init_done")
        } else {
            nativeVLMCmlxSessionLoaded = true
            emitVLMDiagnostic("vlm_cmlx_session_reuse")
        }
    }

    private func generateNativeText(
        messages: [ChatMessage],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try await self.runNativeTextGenerate(
                        messages: messages,
                        tools: tools,
                        requestedParameters: parameters,
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

    private func runNativeTextGenerate(
        messages: [ChatMessage],
        tools: [ToolSpec]?,
        requestedParameters: EdgeGenerateParameters,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard let container = nativeContainer,
              let tokenizer = nativeTokenizer
        else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }

        state = .generating
        defer {
            if state == .generating {
                state = .ready
            }
        }

        var parameters = requestedParameters
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: parameters.preserveThinking
        )
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: promptMessages.chatTemplateMessages(
                preserveThinking: parameters.preserveThinking
            ),
            tools: tools,
            additionalContext: LLMEngine.chatTemplateContext(parameters: parameters)
        )
        let architecture = container.index.languageIndex.architecture
        let arch = archInfo ?? ModelArchInfo.fallback(
            modelSizeGB: LLMEngine.estimateModelSizeGB(
                directory: modelDirectory ?? URL(fileURLWithPath: ".")
            )
        )
        let snapshot = InferencePolicy.DeviceSnapshot.capture()
        let resolvedPolicy: InferencePolicy.Resolved?
        let turn = turnCounter + 1
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
            staticPolicy: memoryPolicy
        )
        resolved.apply(to: &parameters)
        lastPolicy = resolved
        resolvedPolicy = resolved
        if resolved.shouldPause {
            throw EdgeRuntimeError.thermalPause
        }
        let dsrPolicies = try LLMEngine.makeDSRPolicies(
            parameters: parameters,
            architecture: architecture
        )
        let kvCapacity = LLMEngine.kvCapacity(
            promptTokenCount: promptTokens.count,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let attentionQuantization = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: parameters,
            dsrPolicies: dsrPolicies
        )
        let frogJumpMask = Self.frogJumpLayerMask(
            architecture: architecture,
            requestedEnabled: parameters.frogJumpEnabled,
            thinkingEnabled: parameters.enableThinking
        )
        let residentTokenCount = max(promptTokens.count, nativeVLMCmlxTokenIds.count)
        let residentKVCapacity = LLMEngine.kvCapacity(
            promptTokenCount: residentTokenCount,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let attentionCacheLimit = parameters.maxKVSize
            ?? parameters.dsrMaxCritical
            ?? residentKVCapacity

        let memoryBefore = DeviceProfile.captureMemorySnapshot().footprintMB
        let startedAt = Date()
        try Task.checkCancellation()
        if try await runNativeVLMCmlxTextGenerateIfPossible(
            messages: promptMessages,
            promptTokens: promptTokens,
            tools: tools,
            parameters: parameters,
            resolvedPolicy: resolvedPolicy,
            dsrPolicies: dsrPolicies,
            attentionQuantization: attentionQuantization,
            frogJumpMask: frogJumpMask,
            attentionCacheLimit: attentionCacheLimit,
            turn: turn,
            container: container,
            tokenizer: tokenizer,
            startedAt: startedAt,
            memoryBefore: memoryBefore,
            continuation: continuation
        ) {
            return
        }

        let decoderLoadStartedAt = Date()
        let deferredLoadMs: Double?
        if container.isDecoderLoaded {
            deferredLoadMs = nil
        } else {
            try container.loadDecoderWeights()
            deferredLoadMs = Date().timeIntervalSince(decoderLoadStartedAt) * 1000
        }

        let session = try container.makeGreedyDecodeSession(
            kvCapacity: kvCapacity,
            dsrPolicies: dsrPolicies
        )

        try Task.checkCancellation()
        let useGreedyTokenPath = parameters.temperature == 0
        if useGreedyTokenPath {
            try session.prefillGreedy(promptTokenIds: promptTokens)
        } else {
            try session.prefill(promptTokenIds: promptTokens)
        }
        promptCache.update(
            cache: [],
            totalTokenCount: promptTokens.count,
            tokenPrefix: promptTokens.map { Int32($0) }
        )

        var rng = EdgeSeededRandomNumberGenerator(seed: UInt64.random(in: 1...UInt64.max))
        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(parameters.maxTokens)
        var emittedText = ""
        var firstTokenAt: Date?

        while generatedTokenIds.count < parameters.maxTokens {
            try Task.checkCancellation()
            let tokenId: Int
            let sampling = LLMEngine.qwenSamplingConfiguration(
                parameters: parameters,
                promptSessionTokenIds: promptTokens,
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
            }
            generatedTokenIds.append(tokenId)
            let decodedText = tokenizer.decode(
                tokens: generatedTokenIds,
                skipSpecialTokens: true
            )
            let delta: String
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                delta = decodedText
            }
            emittedText = decodedText
            if !delta.isEmpty {
                continuation.yield(
                    GenerateChunk(
                        text: delta,
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
            }
            if useGreedyTokenPath {
                try session.advanceGreedy(with: tokenId)
            } else {
                try session.advance(with: tokenId)
            }
        }

        let endedAt = Date()
        let first = firstTokenAt ?? endedAt
        let decodeSeconds = max(endedAt.timeIntervalSince(first), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        turnCounter = turn
        lastMetrics = InferenceMetrics(
            ttftMs: first.timeIntervalSince(startedAt) * 1000,
            decodeTPS: Double(generatedTokenIds.count) / decodeSeconds,
            deferredLoadMs: deferredLoadMs,
            promptTokenCount: promptTokens.count,
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: resolvedPolicy?.reasoning ?? "policy unavailable",
            promptCacheHit: false,
            cachedTokensReused: 0,
            thermalState: ThermalManager().level.rawValue,
            turn: turn
        )
    }

    enum NativeVLMCmlxTextEligibility: Equatable {
        case eligible
        case ineligible(reason: String)
    }

    /// Pure eligibility check for the fast Metal CMLX text path. Temperature > 0
    /// (the default 0.7) is eligible — sampling runs on CMLX. This is the
    /// regression guard against re-introducing the old `temperature == 0` reject
    /// that forced VLM text-only turns onto the slow Swift decoder (~0.5 tok/s).
    static func nativeVLMCmlxTextEligibility(
        parameters: EdgeGenerateParameters,
        hasTools: Bool
    ) -> NativeVLMCmlxTextEligibility {
        if hasTools { return .ineligible(reason: "tools_present") }
        guard parameters.temperature >= 0, parameters.temperature.isFinite else {
            return .ineligible(reason: "invalid_temperature")
        }
        guard parameters.topP.isFinite else {
            return .ineligible(reason: "invalid_top_p")
        }
        guard parameters.minP >= 0, parameters.minP.isFinite else {
            return .ineligible(reason: "invalid_min_p")
        }
        if let topK = parameters.topK, topK <= 0 {
            return .ineligible(reason: "invalid_top_k")
        }
        guard parameters.repetitionPenalty > 0, parameters.repetitionPenalty.isFinite else {
            return .ineligible(reason: "invalid_repetition_penalty")
        }
        guard parameters.repetitionContextSize >= 0 else {
            return .ineligible(reason: "invalid_repetition_context_size")
        }
        guard parameters.presencePenalty.isFinite else {
            return .ineligible(reason: "invalid_presence_penalty")
        }
        guard parameters.presenceContextSize >= 0 else {
            return .ineligible(reason: "invalid_presence_context_size")
        }
        guard parameters.frequencyPenalty.isFinite else {
            return .ineligible(reason: "invalid_frequency_penalty")
        }
        guard parameters.frequencyContextSize >= 0 else {
            return .ineligible(reason: "invalid_frequency_context_size")
        }
        guard parameters.minimumGeneratedTokens >= 0 else {
            return .ineligible(reason: "invalid_minimum_generated_tokens")
        }
        guard parameters.eosPenaltyUntilToken >= 0 else {
            return .ineligible(reason: "invalid_eos_penalty_until_token")
        }
        return .eligible
    }

    private func runNativeVLMCmlxTextGenerateIfPossible(
        messages: [ChatMessage],
        promptTokens: [Int],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters,
        resolvedPolicy: InferencePolicy.Resolved?,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        attentionQuantization: NativeCmlxAttentionCacheQuantization?,
        frogJumpMask: UInt64,
        attentionCacheLimit: Int?,
        turn: Int,
        container: QwenVLMNativeContainer,
        tokenizer: Tokenizer,
        startedAt: Date,
        memoryBefore: Int,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws -> Bool {
        switch Self.nativeVLMCmlxTextEligibility(
            parameters: parameters,
            hasTools: !(tools?.isEmpty ?? true)
        ) {
        case .eligible:
            break
        case .ineligible(let reason):
            emitVLMDiagnostic("vlm_cmlx_text_reject reason=\(reason)")
            releaseNativeVLMCmlxSession(container: container, reason: reason)
            return false
        }
        let hadMatchingSession = container.isCmlxDecoderLoaded && nativeVLMCmlxSessionMatches(
            dsrPolicies: dsrPolicies,
            attentionQuantization: attentionQuantization,
            frogJumpMask: frogJumpMask,
            attentionCacheLimit: attentionCacheLimit
        )
        do {
            try prepareNativeVLMCmlxSession(
                container: container,
                dsrPolicies: dsrPolicies,
                attentionQuantization: attentionQuantization,
                frogJumpMask: frogJumpMask,
                attentionCacheLimit: attentionCacheLimit
            )
        } catch {
            emitVLMDiagnostic("vlm_cmlx_text_reject reason=session_init_failed error=\(error)")
            releaseNativeVLMCmlxSession(container: container, reason: "session_init_failed")
            return false
        }

        var prefillTokens = promptTokens
        var cachedTokensReused = 0
        var promptCacheHit = false
        if hadMatchingSession {
            if let reusablePrefix = NativePromptSessionReuse.reusablePrefixMatch(
                cachedTokenIds: nativeVLMCmlxTokenIds,
                promptTokenIds: promptTokens,
                skippableCachedTokenSequences: LLMEngine.qwenThinkingSentinelTokenSequences(
                    tokenizer: tokenizer
                )
            ) {
                let suffix = Array(promptTokens.dropFirst(reusablePrefix.promptTokenLength))
                if suffix.isEmpty {
                    emitVLMDiagnostic(
                        "vlm_cmlx_prompt_cache_token_reject reason=empty_suffix reused=\(reusablePrefix.cachedTokenLength)"
                    )
                } else {
                    cachedTokensReused = reusablePrefix.cachedTokenLength
                    promptCacheHit = true
                    prefillTokens = suffix
                    emitVLMDiagnostic(
                        "vlm_cmlx_prompt_cache_token_hit reused=\(cachedTokensReused) promptReused=\(reusablePrefix.promptTokenLength) suffix=\(suffix.count)"
                    )
                }
            }

            if !promptCacheHit, nativeVLMCmlxContainsMediaContext {
                let planResult = NativeVLMTextDeltaPrefillPlanner.makePlan(
                    cachedTokenCount: nativeVLMCmlxTokenIds.count,
                    previousPromptMessages: nativeVLMCmlxPromptMessages,
                    lastAssistantText: nativeVLMCmlxLastAssistantText,
                    currentMessages: messages,
                    enableThinking: parameters.enableThinking
                )
                switch planResult {
                case .success(let plan):
                    let suffix = tokenizer.encode(
                        text: plan.suffixText,
                        addSpecialTokens: false
                    )
                    if suffix.isEmpty {
                        emitVLMDiagnostic(
                            "vlm_cmlx_prompt_cache_incremental_reject reason=empty_suffix"
                        )
                    } else {
                        cachedTokensReused = plan.cachedTokensReused
                        promptCacheHit = true
                        prefillTokens = suffix
                        emitVLMDiagnostic(
                            "vlm_cmlx_prompt_cache_incremental_hit reused=\(cachedTokensReused) suffix=\(suffix.count)"
                        )
                    }
                case .failure(let reason):
                    emitVLMDiagnostic(
                        "vlm_cmlx_prompt_cache_incremental_reject reason=\(reason)"
                    )
                }
            }
        } else {
            emitVLMDiagnostic("vlm_cmlx_prompt_cache_incremental_reject reason=no_session")
        }

        // Sampling setup. VLM text turns now use the same fast Metal cmlx
        // decode path as LLM for temperature > 0 (the default is 0.7) instead
        // of falling back to the slow Swift greedy decoder. The vision tower is
        // never loaded on this path, so text-only memory stays at the LLM
        // profile.
        let useSampledPath = parameters.temperature > 0
        let cmlxTopK = parameters.topK
        let cmlxTopP = (parameters.topP > 0 && parameters.topP <= 1) ? parameters.topP : 1
        let cmlxMinP = parameters.minP
        var cmlxSamplingRNG = EdgeSeededRandomNumberGenerator(
            seed: UInt64.random(in: 1...UInt64.max)
        )
        try? container.clearCmlxRepetitionPenalty()
        try? container.clearCmlxEOSSamplingBias()
        let samplingPenaltiesAreActive = parameters.repetitionPenalty != 1.0 ||
            parameters.presencePenalty != 0.0 ||
            parameters.frequencyPenalty != 0.0
        let useSamplingPenalties = useSampledPath && samplingPenaltiesAreActive
        let eosSamplingBiasRequested = parameters.minimumGeneratedTokens > 0 ||
            parameters.eosPenaltyUntilToken > 0
        let useEOSSamplingBias = useSampledPath &&
            eosSamplingBiasRequested &&
            !nativeEndTokenIds.isEmpty
        func applyVLMCmlxSamplingPenalties(
            promptSessionTokenIds: [Int],
            generatedTokenIds: [Int] = []
        ) throws {
            try container.setCmlxSamplingPenalties(
                repetitionPenalty: parameters.repetitionPenalty,
                repetitionContextTokenIds: LLMEngine.cmlxRepetitionContextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.repetitionContextSize
                ),
                presencePenalty: parameters.presencePenalty,
                presenceContextTokenIds: LLMEngine.cmlxRepetitionContextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.presenceContextSize
                ),
                frequencyPenalty: parameters.frequencyPenalty,
                frequencyContextTokenIds: LLMEngine.cmlxRepetitionContextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.frequencyContextSize
                )
            )
        }
        func applyVLMCmlxEOSSamplingBias(generatedTokenCount: Int) throws {
            guard useEOSSamplingBias else { return }
            let suppress = generatedTokenCount < parameters.minimumGeneratedTokens
            let logitPenalty: Float = generatedTokenCount < parameters.eosPenaltyUntilToken
                ? Self.eosSamplingLogitPenalty
                : 0
            if suppress || logitPenalty > 0 {
                try container.setCmlxEOSSamplingBias(
                    tokenIds: Array(nativeEndTokenIds),
                    suppress: suppress,
                    logitPenalty: logitPenalty
                )
            } else {
                try container.clearCmlxEOSSamplingBias()
            }
        }
        defer {
            if useSamplingPenalties { try? container.clearCmlxRepetitionPenalty() }
            if useEOSSamplingBias { try? container.clearCmlxEOSSamplingBias() }
        }

        if !promptCacheHit {
            _ = try container.resetCmlxDecoder()
            nativeVLMCmlxTokenIds = []
            nativeVLMCmlxContainsMediaContext = false
        }
        emitVLMDiagnostic(
            "vlm_cmlx_text_prefill_begin tokens=\(prefillTokens.count) mode=\(promptCacheHit ? "incremental" : "full") cached=\(cachedTokensReused) decode=\(useSampledPath ? "sampled" : "greedy")"
        )
        let residentPromptTokensAfterPrefill = nativeVLMCmlxTokenIds + prefillTokens
        if useSamplingPenalties {
            try applyVLMCmlxSamplingPenalties(
                promptSessionTokenIds: residentPromptTokensAfterPrefill
            )
        }
        if useEOSSamplingBias {
            try applyVLMCmlxEOSSamplingBias(generatedTokenCount: 0)
        }
        try Task.checkCancellation()
        var nextTokenID: Int? = try runNativeVLMCmlxTokenPrefill(
            container: container,
            tokenIDs: prefillTokens,
            parameters: parameters,
            useSampledPath: useSampledPath,
            topK: cmlxTopK,
            topP: cmlxTopP,
            minP: cmlxMinP,
            rng: &cmlxSamplingRNG
        )
        if promptCacheHit {
            nativeVLMCmlxTokenIds.append(contentsOf: prefillTokens)
        } else {
            nativeVLMCmlxTokenIds = prefillTokens
        }
        let promptTokenCountAfterPrefill = nativeVLMCmlxTokenIds.count
        let firstTokenAt = Date()
        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(parameters.maxTokens)
        var emittedText = ""
        var stoppedTokenId: Int?

        while generatedTokenIds.count < parameters.maxTokens, let tokenID = nextTokenID {
            try Task.checkCancellation()
            if parameters.stopOnEndToken,
               nativeEndTokenIds.contains(tokenID),
               (!useEOSSamplingBias || generatedTokenIds.count >= parameters.minimumGeneratedTokens) {
                stoppedTokenId = tokenID
                break
            }
            generatedTokenIds.append(tokenID)
            let decodedText = tokenizer.decode(
                tokens: generatedTokenIds,
                skipSpecialTokens: true
            )
            let delta: String
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                delta = decodedText
            }
            emittedText = decodedText
            if !delta.isEmpty {
                continuation.yield(
                    GenerateChunk(
                        text: delta,
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
            }
            if generatedTokenIds.count < parameters.maxTokens {
                if useSampledPath {
                    if useSamplingPenalties {
                        try applyVLMCmlxSamplingPenalties(
                            promptSessionTokenIds: nativeVLMCmlxTokenIds,
                            generatedTokenIds: generatedTokenIds
                        )
                    }
                    if useEOSSamplingBias {
                        try applyVLMCmlxEOSSamplingBias(
                            generatedTokenCount: generatedTokenIds.count
                        )
                    }
                    nextTokenID = try container.nextSampledCmlxToken(
                        temperature: parameters.temperature,
                        topK: cmlxTopK,
                        topP: cmlxTopP,
                        minP: cmlxMinP,
                        seed: cmlxSamplingRNG.next()
                    )
                } else {
                    nextTokenID = try container.decodeCmlxStep(tokenID: tokenID)
                }
            } else {
                nextTokenID = nil
            }
        }

        let endedAt = Date()
        let decodeSeconds = max(endedAt.timeIntervalSince(firstTokenAt), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        nativeVLMCmlxTokenIds.append(contentsOf: generatedTokenIds)
        // Mirror LLMEngine: the sampled CMLX path commits the stop token into the
        // native session, so the Swift-side token ledger must include it to keep
        // next-turn prompt-cache prefix reuse in sync. The greedy path
        // (decodeCmlxStep) never feeds the unconsumed stop token, so it is excluded
        // there.
        if useSampledPath, let stoppedTokenId {
            nativeVLMCmlxTokenIds.append(stoppedTokenId)
        }
        nativeVLMCmlxPromptMessages = messages
        nativeVLMCmlxLastAssistantText = NativePromptSessionReuse.normalizeAssistantText(emittedText)
        turnCounter = turn
        promptCache.update(
            cache: [],
            totalTokenCount: nativeVLMCmlxTokenIds.count,
            tokenPrefix: promptTokens.map { Int32($0) }
        )
        lastMetrics = InferenceMetrics(
            ttftMs: firstTokenAt.timeIntervalSince(startedAt) * 1000,
            decodeTPS: Double(generatedTokenIds.count) / decodeSeconds,
            deferredLoadMs: nil,
            promptTokenCount: promptTokenCountAfterPrefill,
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: (resolvedPolicy?.reasoning ?? "policy unavailable")
                + " | nativeVLMTextCmlx=\(useSampledPath ? "sampled" : "greedy")",
            promptCacheHit: promptCacheHit,
            cachedTokensReused: cachedTokensReused,
            thermalState: ThermalManager().level.rawValue,
            turn: turn
        )
        return true
    }

    private func runNativeVLMCmlxTokenPrefill(
        container: QwenVLMNativeContainer,
        tokenIDs: [Int],
        parameters: EdgeGenerateParameters,
        useSampledPath: Bool,
        topK: Int?,
        topP: Float,
        minP: Float,
        rng: inout EdgeSeededRandomNumberGenerator
    ) throws -> Int {
        let chunkSize = max(1, parameters.prefillStepSize)
        guard tokenIDs.count > chunkSize else {
            if useSampledPath {
                try container.prefillSampledCmlxTokensAsync(
                    tokenIDs: tokenIDs,
                    temperature: parameters.temperature,
                    topK: topK,
                    topP: topP,
                    minP: minP,
                    seed: rng.next()
                )
                return try container.nextSampledCmlxToken(
                    temperature: parameters.temperature,
                    topK: topK,
                    topP: topP,
                    minP: minP,
                    seed: rng.next()
                )
            }
            return try container.prefillCmlxTokens(tokenIDs: tokenIDs)
        }

        let chunkCount = Int(ceil(Double(tokenIDs.count) / Double(chunkSize)))
        emitVLMDiagnostic(
            "vlm_cmlx_text_prefill_chunked total=\(tokenIDs.count) step=\(chunkSize) chunks=\(chunkCount) decode=\(useSampledPath ? "sampled" : "greedy")"
        )
        var offset = 0
        var chunkIndex = 0
        while offset < tokenIDs.count {
            let end = min(offset + chunkSize, tokenIDs.count)
            let chunk = Array(tokenIDs[offset..<end])
            let isLast = end == tokenIDs.count
            chunkIndex += 1
            if isLast {
                let nextToken: Int
                if useSampledPath {
                    try container.prefillSampledCmlxTokensAsync(
                        tokenIDs: chunk,
                        temperature: parameters.temperature,
                        topK: topK,
                        topP: topP,
                        minP: minP,
                        seed: rng.next()
                    )
                    nextToken = try container.nextSampledCmlxToken(
                        temperature: parameters.temperature,
                        topK: topK,
                        topP: topP,
                        minP: minP,
                        seed: rng.next()
                    )
                } else {
                    nextToken = try container.prefillCmlxTokens(tokenIDs: chunk)
                }
                emitVLMDiagnostic(
                    "vlm_cmlx_text_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=true"
                )
                return nextToken
            }

            try container.prefillCmlxTokensAsync(tokenIDs: chunk)
            emitVLMDiagnostic(
                "vlm_cmlx_text_prefill_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=false async=true"
            )
            offset = end
        }

        throw QwenHybridModelReferenceError.emptyTokenIds
    }

    private func generateNativeImage(
        messages: [ChatMessage],
        inputs: [NativeVLMImageInput],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try await self.runNativeImageGenerate(
                        messages: messages,
                        inputs: inputs,
                        tools: tools,
                        requestedParameters: parameters,
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

    private func runNativeImageGenerate(
        messages: [ChatMessage],
        inputs: [NativeVLMImageInput],
        tools: [ToolSpec]?,
        requestedParameters: EdgeGenerateParameters,
        continuation: AsyncThrowingStream<GenerateChunk, Error>.Continuation
    ) async throws {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard !inputs.isEmpty else {
            throw EdgeRuntimeError.loadFailed("Native VLM image input is empty")
        }
        guard let container = nativeContainer,
              let tokenizer = nativeTokenizer
        else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }

        state = .generating
        defer {
            if state == .generating {
                state = .ready
            }
        }

        let startedAt = Date()
        let memoryBefore = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] image generate start: %.0f MB", memoryBefore)

        var imageProcessorConfig = container.index.preflightResult.plan.imageProcessorConfiguration
        let deviceRAMGB = DeviceProfile.current.totalRAMGB
        if deviceRAMGB < 12 {
            let safeCap = 1280 * 28 * 28
            let currentMax = imageProcessorConfig.maxPixels ?? (16_384 * 28 * 28)
            if currentMax > safeCap {
                imageProcessorConfig.maxPixels = safeCap
                NSLog("[VLM-MEM] clamped maxPixels %d → %d (device %d GB RAM)", currentMax, safeCap, deviceRAMGB)
            }
        }
        let preprocessings = try inputs.map { input in
            switch input {
            case .url(let url):
                return try QwenImagePreprocessor.preprocessImage(
                    at: url,
                    configuration: imageProcessorConfig
                )
            case .ciImage(let image):
                return try QwenImagePreprocessor.preprocessCIImage(
                    image,
                    configuration: imageProcessorConfig
                )
            }
        }
        let afterPreprocess = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] after image preprocess: %.0f MB (+%.0f)", afterPreprocess, afterPreprocess - memoryBefore)

        let patchDim = try Self.patchDimension(preprocessings)
        let totalPatchCount = preprocessings.reduce(0) { partial, item in
            partial + item.pixelValuesShape[0]
        }
        NSLog("[VLM-MEM] patches=%d patchDim=%d grids=%@", totalPatchCount, patchDim, preprocessings.map(\.imageGridTHW).description)
        let pixelValues = preprocessings.reduce(into: [Float]()) { result, item in
            result.reserveCapacity(result.count + item.pixelValues.count)
            result.append(contentsOf: item.pixelValues)
        }
        let grids = preprocessings.map(\.imageGridTHW)
        let imageTokenCounts = try Self.imageTokenCounts(
            for: grids,
            plan: container.index.preflightResult.plan
        )
        NSLog("[VLM-MEM] imageTokenCounts=%@, pixelValues=%.1f MB", imageTokenCounts.description, Float(pixelValues.count * 4) / 1_048_576.0)

        try Task.checkCancellation()
        if container.isCmlxVisionLoaded {
            container.unloadCmlxVisionWeights()
        }
        let beforeVision = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] before vision encoder load: %.0f MB", beforeVision)
        try container.loadCmlxVisionWeights()
        let afterVisionLoad = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] after vision encoder load: %.0f MB (+%.0f)", afterVisionLoad, afterVisionLoad - beforeVision)
        defer {
            container.unloadCmlxVisionWeights()
            let afterUnload = DeviceProfile.captureMemorySnapshot().footprintMB
            NSLog("[VLM-MEM] after vision encoder unload: %.0f MB", afterUnload)
        }

        let visionEncoding = try container.visionEncode(
            pixelValues: pixelValues,
            pixelValuesShape: [totalPatchCount, patchDim],
            gridTHW: grids
        )
        let afterVisionEncode = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] after vision encode: %.0f MB (+%.0f from load)", afterVisionEncode, afterVisionEncode - afterVisionLoad)
        let totalImageTokenCount = imageTokenCounts.reduce(0, +)
        guard totalImageTokenCount > 0,
              visionEncoding.shape.first == totalImageTokenCount
        else {
            throw EdgeRuntimeError.loadFailed("Native VLM vision encoder returned no image tokens")
        }

        let imageTokenID = try Self.tokenID(
            "<|image_pad|>",
            tokenizer: tokenizer,
            fallback: 248_056
        )
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: requestedParameters.preserveThinking
        )
        let promptTokens = try Self.makeNativeVLMPromptTokens(
            messages: promptMessages,
            imageTokenCounts: imageTokenCounts,
            tokenizer: tokenizer,
            tools: tools,
            parameters: requestedParameters
        )
        let placeholderCount = promptTokens.reduce(0) { count, token in
            count + (token == imageTokenID ? 1 : 0)
        }
        guard placeholderCount == totalImageTokenCount else {
            throw EdgeRuntimeError.loadFailed(
                "Native VLM prompt image token count mismatch: prompt=\(placeholderCount), features=\(totalImageTokenCount)"
            )
        }

        var parameters = requestedParameters
        let architecture = container.index.languageIndex.architecture
        let arch = archInfo ?? ModelArchInfo.fallback(
            modelSizeGB: LLMEngine.estimateModelSizeGB(
                directory: modelDirectory ?? URL(fileURLWithPath: ".")
            )
        )
        let snapshot = InferencePolicy.DeviceSnapshot.capture()
        let turn = turnCounter + 1
        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: InferencePolicy.TurnContext(
                turn: turn,
                cachedTokenCount: 0,
                archInfo: arch,
                scene: parameters.dsrScene,
                requestedMaxTokens: parameters.maxTokens,
                planPrefillStepSize: currentPlan?.prefillStepSize,
                memoryIntent: currentPlan?.memoryIntent ?? memoryPolicy?.memoryIntent ?? .balanced
            ),
            staticPolicy: memoryPolicy
        )
        resolved.apply(to: &parameters)
        lastPolicy = resolved
        if resolved.shouldPause {
            throw EdgeRuntimeError.thermalPause
        }
        let dsrPolicies = try LLMEngine.makeDSRPolicies(
            parameters: parameters,
            architecture: architecture
        )
        let kvCapacity = LLMEngine.kvCapacity(
            promptTokenCount: promptTokens.count,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let attentionQuantization = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: parameters,
            dsrPolicies: dsrPolicies
        )
        let frogJumpMask = Self.frogJumpLayerMask(
            architecture: architecture,
            requestedEnabled: parameters.frogJumpEnabled,
            thinkingEnabled: parameters.enableThinking
        )

        let attentionCacheLimit = parameters.maxKVSize ?? parameters.dsrMaxCritical ?? kvCapacity
        try prepareNativeVLMCmlxSession(
            container: container,
            dsrPolicies: dsrPolicies,
            attentionQuantization: attentionQuantization,
            frogJumpMask: frogJumpMask,
            attentionCacheLimit: attentionCacheLimit
        )
        _ = try container.resetCmlxDecoder()
        nativeVLMCmlxTokenIds = []
        nativeVLMCmlxPromptMessages = []
        nativeVLMCmlxLastAssistantText = ""
        nativeVLMCmlxContainsMediaContext = false

        try Task.checkCancellation()
        var nextTokenID: Int? = try container.prefillImageFeatures(
            tokenIDs: promptTokens,
            imageFeatures: visionEncoding.values,
            imageFeatureShape: visionEncoding.shape,
            imageTokenID: imageTokenID
        )
        let firstTokenAt = Date()
        var generatedTokenIds: [Int] = []
        generatedTokenIds.reserveCapacity(parameters.maxTokens)
        var emittedText = ""

        while generatedTokenIds.count < parameters.maxTokens, let tokenID = nextTokenID {
            try Task.checkCancellation()
            if parameters.stopOnEndToken, nativeEndTokenIds.contains(tokenID) {
                break
            }
            generatedTokenIds.append(tokenID)
            let decodedText = tokenizer.decode(
                tokens: generatedTokenIds,
                skipSpecialTokens: true
            )
            let delta: String
            if decodedText.hasPrefix(emittedText) {
                let start = decodedText.index(decodedText.startIndex, offsetBy: emittedText.count)
                delta = String(decodedText[start...])
            } else {
                delta = decodedText
            }
            emittedText = decodedText
            if !delta.isEmpty {
                continuation.yield(
                    GenerateChunk(
                        text: delta,
                        generatedTokenCount: generatedTokenIds.count
                    )
                )
            }

            nextTokenID = try container.decodeCmlxStep(tokenID: tokenID)
        }

        let endedAt = Date()
        let decodeSeconds = max(endedAt.timeIntervalSince(firstTokenAt), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        turnCounter = turn
        nativeVLMCmlxTokenIds = promptTokens + generatedTokenIds
        nativeVLMCmlxPromptMessages = promptMessages
        nativeVLMCmlxLastAssistantText = NativePromptSessionReuse.normalizeAssistantText(emittedText)
        nativeVLMCmlxContainsMediaContext = true
        emitVLMDiagnostic(
            "vlm_cmlx_media_session_seeded prompt=\(promptTokens.count) generated=\(generatedTokenIds.count) total=\(nativeVLMCmlxTokenIds.count)"
        )
        promptCache.update(
            cache: [],
            totalTokenCount: nativeVLMCmlxTokenIds.count,
            tokenPrefix: nativeVLMCmlxTokenIds.map { Int32($0) }
        )
        lastMetrics = InferenceMetrics(
            ttftMs: firstTokenAt.timeIntervalSince(startedAt) * 1000,
            decodeTPS: Double(generatedTokenIds.count) / decodeSeconds,
            deferredLoadMs: nil,
            promptTokenCount: promptTokens.count,
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: resolved.reasoning + " | nativeVLMImageCmlx=on",
            promptCacheHit: false,
            cachedTokensReused: 0,
            thermalState: ThermalManager().level.rawValue,
            turn: turn
        )
    }

    private static func patchDimension(
        _ preprocessings: [QwenImagePreprocessingResult]
    ) throws -> Int {
        guard let first = preprocessings.first,
              first.pixelValuesShape.count == 2,
              first.pixelValuesShape[0] > 0,
              first.pixelValuesShape[1] > 0
        else {
            throw EdgeRuntimeError.loadFailed("Native VLM image preprocessing returned an invalid shape")
        }
        let patchDim = first.pixelValuesShape[1]
        for item in preprocessings {
            guard item.pixelValuesShape.count == 2,
                  item.pixelValuesShape[0] > 0,
                  item.pixelValuesShape[1] == patchDim,
                  item.pixelValues.count == item.pixelValuesShape[0] * patchDim
            else {
                throw EdgeRuntimeError.loadFailed("Native VLM image preprocessing shape mismatch")
            }
        }
        return patchDim
    }

    private static func imageTokenCounts(
        for grids: [QwenImageGridTHW],
        plan: QwenVLMRuntimePlan
    ) throws -> [Int] {
        let mergeSize = plan.visionConfiguration.spatialMergeSize
            ?? plan.imageProcessorConfiguration.mergeSize
            ?? 2
        let mergeArea = mergeSize * mergeSize
        guard mergeArea > 0 else {
            throw EdgeRuntimeError.loadFailed("Native VLM merge size must be positive")
        }
        return try grids.map { grid in
            guard grid.product % mergeArea == 0 else {
                throw EdgeRuntimeError.loadFailed("Native VLM image grid is not merge-aligned")
            }
            return grid.product / mergeArea
        }
    }

    private static func tokenID(
        _ token: String,
        tokenizer: Tokenizer,
        fallback: Int
    ) throws -> Int {
        guard let id = tokenizer.convertTokenToId(token) else {
            throw EdgeRuntimeError.loadFailed("Native VLM tokenizer is missing \(token)")
        }
        if id != fallback {
            return id
        }
        return fallback
    }

    private static func makeNativeVLMPromptTokens(
        messages: [ChatMessage],
        imageTokenCounts: [Int],
        tokenizer: Tokenizer,
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters
    ) throws -> [Int] {
        guard !imageTokenCounts.isEmpty,
              imageTokenCounts.allSatisfy({ $0 > 0 })
        else {
            throw EdgeRuntimeError.loadFailed("Native VLM image token count must be positive")
        }
        let visionBlock = imageTokenCounts.map { imageTokenCount in
            let imagePadding = String(repeating: "<|image_pad|>", count: imageTokenCount)
            return "<|vision_start|>\(imagePadding)<|vision_end|>"
        }.joined()
        var promptMessages = messages
        if let lastUserIndex = promptMessages.lastIndex(where: { $0.role == .user }) {
            let current = promptMessages[lastUserIndex]
            promptMessages[lastUserIndex] = ChatMessage(
                role: .user,
                content: visionBlock + current.content
            )
        } else {
            promptMessages.append(.user(visionBlock))
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: promptMessages.chatTemplateMessages(
                preserveThinking: parameters.preserveThinking
            ),
            tools: tools,
            additionalContext: LLMEngine.chatTemplateContext(parameters: parameters)
        )
        guard !tokens.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        return tokens
    }

    private static func frogJumpLayerMask(
        architecture: QwenHybridArchitecture,
        requestedEnabled: Bool,
        thinkingEnabled: Bool
    ) -> UInt64 {
        let plan = QwenFrogJumpPlan.compute(
            architecture: architecture,
            requestedEnabled: requestedEnabled,
            thinkingEnabled: thinkingEnabled
        )
        guard plan.enabled,
              plan.skipLayers.contains(12),
              plan.skipLayers.contains(13)
        else {
            return 0
        }
        return [12, 13].reduce(UInt64.zero) { mask, layer in
            mask | (UInt64(1) << UInt64(layer))
        }
    }

    private static func cmlxAttentionCacheQuantizationSummary(
        _ quantization: NativeCmlxAttentionCacheQuantization?
    ) -> String {
        guard let quantization else { return "none" }
        return "int\(quantization.bits)@\(quantization.groupSize)"
    }

    private func _generateNativePending(_ message: String) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: EdgeRuntimeError.unsupportedFeature(message))
        }
    }
}
