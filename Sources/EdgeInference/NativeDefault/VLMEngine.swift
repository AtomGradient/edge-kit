// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import CoreImage
import EdgeEngine
import Foundation
import Tokenizers

public enum VLMImagePolicy: Sendable, Equatable {
    /// Preserves more visual detail for OCR, documents, small objects, and charts.
    case quality
    /// Balanced latency and detail for everyday photos and scene understanding.
    case balanced
    /// Prioritizes low TTFT for simple scene previews and follow-up chat.
    case fast
    /// Custom image-token budget with optional post-vision feature pruning.
    case custom(maxImageTokens: Int, pruneTokens: Int?)

    public static let `default`: VLMImagePolicy = .balanced
}

extension VLMImagePolicy: CustomStringConvertible {
    public var description: String {
        switch self {
        case .quality:
            return "quality"
        case .balanced:
            return "balanced"
        case .fast:
            return "fast"
        case .custom(let maxImageTokens, let pruneTokens):
            return "custom(maxImageTokens:\(maxImageTokens),pruneTokens:\(pruneTokens.map(String.init) ?? "nil"))"
        }
    }
}

private actor VLMNativeOperationSerializer {
    private var isRunning = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func run<T>(_ operation: () async throws -> T) async throws -> T {
        await enter()
        do {
            try Task.checkCancellation()
            let result = try await operation()
            leave()
            return result
        } catch {
            leave()
            throw error
        }
    }

    private func enter() async {
        if !isRunning {
            isRunning = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func leave() {
        if waiters.isEmpty {
            isRunning = false
            return
        }
        waiters.removeFirst().resume()
    }
}

public final class VLMEngine: ObservableObject {
    public static let neuralImprintArtifactFileName = NeuralImprintRuntimeSupport.artifactFileName
    public static let legacyPersonaKVArtifactFileName = NeuralImprintRuntimeSupport.legacyArtifactFileName
    public static let neuralImprintMetadataFileName = NeuralImprintRuntimeSupport.metadataFileName
    public static let legacyPersonaKVMetadataFileName = NeuralImprintRuntimeSupport.legacyMetadataFileName
    public static let neuralImprintPrefixSplitSentinel = NeuralImprintRuntimeSupport.prefixSplitSentinel

    public typealias NeuralImprintCacheStatus = NeuralImprintRuntimeCacheStatus

    @Published public private(set) var state: EngineState = .idle
    @Published public private(set) var downloadProgress: Double = 0
    @Published public private(set) var lastPolicy: InferencePolicy.Resolved?
    @Published public private(set) var lastMetrics: InferenceMetrics?

    public private(set) var memoryPolicy: KVCacheMemoryPolicy?
    public private(set) var currentPlan: MemoryBudgetPlanner.Plan?
    public private(set) var archInfo: ModelArchInfo?
    public private(set) var visionOffloaded: Bool = false
    public private(set) var defaultImagePolicy: VLMImagePolicy = .default
    public private(set) var activeNeuralImprintCache: NeuralImprintCacheStatus?
    public let promptCache = PromptCacheManager()

    private var modelDirectory: URL?
    private var nativeContainer: QwenVLMNativeContainer?
    private let nativeOperationSerializer = VLMNativeOperationSerializer()
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
    private var nativeCmlxLimitState = NativeCmlxCommandBufferLimitState()
    private var nativeVLMPreparedImageCache: [NativeVLMPreparedImageKey: NativeVLMPreparedImageFeatures] = [:]
    private var nativeVLMPreparedImageCacheOrder: [NativeVLMPreparedImageKey] = []
    private var nativeVLMPreparedImagePreloadTasks: [NativeVLMPreparedImageKey: Task<NativeVLMPreparedImageFeatures, Error>] = [:]
    public var diagnosticSink: ((String) -> Void)?

    private enum NativeVLMImageInput {
        case url(URL)
        case ciImage(CIImage)
    }

    private struct NativeVLMPreparedImageKey: Hashable, CustomStringConvertible {
        let rawValue: String

        var description: String { rawValue }
    }

    private struct NativeVLMPreprocessedImages {
        let cacheKey: NativeVLMPreparedImageKey?
        let pixelValues: [Float]
        let patchDim: Int
        let totalPatchCount: Int
        let grids: [QwenImageGridTHW]
        let originalImageTokenCounts: [Int]
        let imageTokenCounts: [Int]
        let imageFeaturePrunePlan: NativeVLMImageFeaturePrunePlan?
    }

    private struct NativeVLMPreparedImageFeatures {
        let cacheKey: NativeVLMPreparedImageKey?
        let values: [Float]
        let shape: [Int]
        let imageTokenCounts: [Int]
        let source: String
    }

    struct NativeVLMImagePolicySettings: Equatable, Sendable {
        let policy: VLMImagePolicy
        let maxImageTokens: Int?
        let pruneTokens: Int?
        let maxImageTokensOverriddenByEnvironment: Bool
        let pruneTokensOverriddenByEnvironment: Bool

        var diagnosticSummary: String {
            [
                "policy=\(policy)",
                "maxImageTokens=\(maxImageTokens.map(String.init) ?? "nil")",
                "pruneTokens=\(pruneTokens.map(String.init) ?? "nil")",
                "envMax=\(maxImageTokensOverriddenByEnvironment)",
                "envPrune=\(pruneTokensOverriddenByEnvironment)",
            ].joined(separator: " ")
        }
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

    struct NativeVLMImageAppendPlan: Equatable {
        let cachedTokensReused: Int
        let suffixTokenIds: [Int]
    }

    enum NativeVLMImageAppendPlanResult: Equatable {
        case success(NativeVLMImageAppendPlan)
        case failure(String)
    }

    enum NativeVLMImageAppendPlanner {
        static func makePlan(
            cachedTokenIds: [Int],
            currentPromptTokenIds: [Int],
            skippableCachedTokenSequences: [[Int]],
            previousPromptMessages: [ChatMessage],
            lastAssistantText: String,
            currentMediaPromptMessages: [ChatMessage],
            imageTokenID: Int,
            totalImageTokenCount: Int,
            enableThinking: Bool,
            encodeSuffix: (String) -> [Int]
        ) -> NativeVLMImageAppendPlanResult {
            guard !cachedTokenIds.isEmpty else {
                return .failure("empty_cached_tokens")
            }
            guard totalImageTokenCount > 0 else {
                return .failure("empty_image_tokens")
            }
            if let reusablePrefix = NativePromptSessionReuse.reusablePrefixMatch(
                cachedTokenIds: cachedTokenIds,
                promptTokenIds: currentPromptTokenIds,
                skippableCachedTokenSequences: skippableCachedTokenSequences
            ) {
                let suffixTokenIds = Array(currentPromptTokenIds.dropFirst(reusablePrefix.promptTokenLength))
                return validate(
                    suffixTokenIds: suffixTokenIds,
                    cachedTokensReused: reusablePrefix.cachedTokenLength,
                    imageTokenID: imageTokenID,
                    totalImageTokenCount: totalImageTokenCount,
                    reasonPrefix: "token_prefix"
                )
            }
            switch NativePromptSessionReuse.qwenIncrementalSuffix(
                previousPromptMessages: previousPromptMessages,
                lastAssistantText: lastAssistantText,
                currentMessages: currentMediaPromptMessages,
                enableThinking: enableThinking,
                matchPath: "vlm_image_append_suffix"
            ) {
            case .match(let suffixText):
                let suffixTokenIds = encodeSuffix(suffixText)
                return validate(
                    suffixTokenIds: suffixTokenIds,
                    cachedTokensReused: cachedTokenIds.count,
                    imageTokenID: imageTokenID,
                    totalImageTokenCount: totalImageTokenCount,
                    reasonPrefix: "text_suffix"
                )
            case .reject(let reason):
                return .failure(reason)
            }
        }

        private static func validate(
            suffixTokenIds: [Int],
            cachedTokensReused: Int,
            imageTokenID: Int,
            totalImageTokenCount: Int,
            reasonPrefix: String
        ) -> NativeVLMImageAppendPlanResult {
            guard !suffixTokenIds.isEmpty else {
                return .failure("\(reasonPrefix)_empty_suffix_tokens")
            }
            let imagePlaceholderCount = suffixTokenIds.reduce(0) { count, token in
                count + (token == imageTokenID ? 1 : 0)
            }
            guard imagePlaceholderCount == totalImageTokenCount else {
                return .failure(
                    "\(reasonPrefix)_image_token_count_mismatch suffix=\(imagePlaceholderCount) expected=\(totalImageTokenCount)"
                )
            }
            return .success(NativeVLMImageAppendPlan(
                cachedTokensReused: cachedTokensReused,
                suffixTokenIds: suffixTokenIds
            ))
        }
    }

    struct NativeVLMImageFeaturePrunePlan: Equatable {
        let originalImageTokenCounts: [Int]
        let effectiveImageTokenCounts: [Int]
        let selectedRowIndices: [Int]

        var isPruned: Bool {
            originalImageTokenCounts != effectiveImageTokenCounts
        }
    }

    enum NativeVLMImageFeaturePruner {
        static func makePlan(
            imageTokenCounts: [Int],
            maxTokensPerImage: Int?
        ) -> NativeVLMImageFeaturePrunePlan? {
            guard let maxTokensPerImage,
                  maxTokensPerImage > 0,
                  !imageTokenCounts.isEmpty,
                  imageTokenCounts.allSatisfy({ $0 > 0 })
            else {
                return nil
            }

            var selectedRowIndices: [Int] = []
            selectedRowIndices.reserveCapacity(imageTokenCounts.reduce(0) { total, count in
                total + min(count, maxTokensPerImage)
            })
            var effectiveImageTokenCounts: [Int] = []
            effectiveImageTokenCounts.reserveCapacity(imageTokenCounts.count)
            var rowOffset = 0

            for imageTokenCount in imageTokenCounts {
                let effectiveCount = min(imageTokenCount, maxTokensPerImage)
                effectiveImageTokenCounts.append(effectiveCount)
                for localIndex in uniformRowIndices(
                    sourceCount: imageTokenCount,
                    targetCount: effectiveCount
                ) {
                    selectedRowIndices.append(rowOffset + localIndex)
                }
                rowOffset += imageTokenCount
            }

            return NativeVLMImageFeaturePrunePlan(
                originalImageTokenCounts: imageTokenCounts,
                effectiveImageTokenCounts: effectiveImageTokenCounts,
                selectedRowIndices: selectedRowIndices
            )
        }

        static func apply(
            plan: NativeVLMImageFeaturePrunePlan?,
            values: [Float],
            shape: [Int]
        ) throws -> EdgeMLXQwen35VisionEncoding {
            guard let plan, plan.isPruned else {
                return EdgeMLXQwen35VisionEncoding(values: values, shape: shape)
            }
            guard shape.count == 2,
                  shape[0] == plan.originalImageTokenCounts.reduce(0, +),
                  shape[1] > 0,
                  values.count == shape[0] * shape[1]
            else {
                throw EdgeRuntimeError.loadFailed("Native VLM feature pruning shape mismatch")
            }

            let hiddenSize = shape[1]
            var prunedValues: [Float] = []
            prunedValues.reserveCapacity(plan.selectedRowIndices.count * hiddenSize)
            for rowIndex in plan.selectedRowIndices {
                guard rowIndex >= 0, rowIndex < shape[0] else {
                    throw EdgeRuntimeError.loadFailed("Native VLM feature pruning index out of range")
                }
                let start = rowIndex * hiddenSize
                prunedValues.append(contentsOf: values[start..<(start + hiddenSize)])
            }
            return EdgeMLXQwen35VisionEncoding(
                values: prunedValues,
                shape: [plan.selectedRowIndices.count, hiddenSize]
            )
        }

        static func uniformRowIndices(sourceCount: Int, targetCount: Int) -> [Int] {
            guard sourceCount > 0, targetCount > 0 else {
                return []
            }
            guard targetCount < sourceCount else {
                return Array(0..<sourceCount)
            }
            guard targetCount > 1 else {
                return [sourceCount / 2]
            }

            var indices: [Int] = []
            indices.reserveCapacity(targetCount)
            var previous = -1
            for targetIndex in 0..<targetCount {
                var sourceIndex = Int(
                    (Double(targetIndex) * Double(sourceCount - 1) / Double(targetCount - 1)).rounded()
                )
                sourceIndex = max(0, min(sourceCount - 1, sourceIndex))
                if sourceIndex <= previous {
                    sourceIndex = min(sourceCount - 1, previous + 1)
                }
                indices.append(sourceIndex)
                previous = sourceIndex
            }
            return indices
        }
    }

    public init(imagePolicy: VLMImagePolicy = .default) {
        defaultImagePolicy = imagePolicy
    }

    public func loadLocal(
        directory: URL,
        imagePolicy: VLMImagePolicy = .default,
        onProgress: ((Double) -> Void)? = nil
    ) async throws {
        try await loadLocal(
            directory: directory,
            options: nil,
            imagePolicy: imagePolicy,
            onProgress: onProgress
        )
    }

    public func loadLocal(
        directory: URL,
        options: NativeRuntimeLoadOptions?,
        imagePolicy: VLMImagePolicy = .default,
        onProgress: ((Double) -> Void)? = nil
    ) async throws {
        guard state != .loading else { return }
        state = .loading
        downloadProgress = 0
        onProgress?(0)
        defaultImagePolicy = imagePolicy
        activeNeuralImprintCache = nil
        do {
            let index = try QwenVLMModelBundleIndex.load(from: directory)
            modelDirectory = directory
            archInfo = ModelArchInfo.load(from: directory)
            let benchmark = DeviceBenchmark.cachedOrCurrent()
            let profile = benchmark.profile
            let modelSizeGB = LLMEngine.estimateModelSizeGB(directory: directory)
            let plan = MemoryBudgetPlanner.plan(
                profile: profile,
                modelSizeGB: modelSizeGB,
                measuredBandwidthGBs: benchmark.measuredBandwidthGBs,
                intent: options?.memoryIntent ?? .balanced
            ).applying(options)
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
            if let greedyOutputHeadArgmax = options?.greedyOutputHeadArgmaxEnabled {
                metalConfiguration.useGreedyOutputHeadArgmax = greedyOutputHeadArgmax
            }
            if let maxInFlight = options?.maxInFlightCommandBuffers {
                metalConfiguration.maxInFlightCommandBuffers = maxInFlight
            }
            _ = NativeRuntimeBridge.applyMetalConfiguration(metalConfiguration)
            nativeCmlxLimitState.reset()
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
            clearNativeVLMPreparedImageCache()
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
        parameters: EdgeGenerateParameters = .default,
        imagePolicy: VLMImagePolicy? = nil
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        if !images.isEmpty {
            return generateNativeImage(
                messages: messages,
                inputs: images.map(NativeVLMImageInput.url),
                tools: tools,
                parameters: parameters,
                imagePolicy: imagePolicy ?? defaultImagePolicy
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
        parameters: EdgeGenerateParameters = .default,
        imagePolicy: VLMImagePolicy? = nil
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        if !ciImages.isEmpty {
            return generateNativeImage(
                messages: messages,
                inputs: ciImages.map(NativeVLMImageInput.ciImage),
                tools: tools,
                parameters: parameters,
                imagePolicy: imagePolicy ?? defaultImagePolicy
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
        parameters: EdgeGenerateParameters = .default,
        imagePolicy: VLMImagePolicy? = nil
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        generate(
            messages: messages,
            images: images,
            parameters: parameters,
            imagePolicy: imagePolicy
        )
    }

    @discardableResult
    public func preloadImageFeatures(
        image: URL,
        imagePolicy: VLMImagePolicy? = nil
    ) async throws -> Int {
        try await nativeOperationSerializer.run {
            try await preloadImageFeaturesLocked(
                image: image,
                imagePolicy: imagePolicy
            )
        }
    }

    private func preloadImageFeaturesLocked(
        image: URL,
        imagePolicy: VLMImagePolicy?
    ) async throws -> Int {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("VLM image preload requires a ready model")
        }
        guard let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }

        let inputs: [NativeVLMImageInput] = [.url(image)]
        let resolvedImagePolicy = imagePolicy ?? defaultImagePolicy
        let imagePolicySettings = Self.nativeVLMImagePolicySettings(
            for: resolvedImagePolicy
        )
        let imageProcessorConfig = nativeVLMImageProcessorConfiguration(
            container: container,
            settings: imagePolicySettings,
            emitDiagnostics: false
        )
        guard let cacheKey = nativeVLMPreparedImageCacheKey(
            inputs: inputs,
            container: container,
            imageProcessorConfig: imageProcessorConfig,
            settings: imagePolicySettings
        ) else {
            throw EdgeRuntimeError.loadFailed("VLM image preload requires file-backed image input")
        }

        if let cached = nativeVLMPreparedImageCache[cacheKey] {
            emitVLMDiagnostic(
                "vlm_image_feature_preload_cache_hit key=\(cacheKey.rawValue.hashValue) imageTokens=\(cached.imageTokenCounts)"
            )
            return cached.imageTokenCounts.reduce(0, +)
        }
        if let task = nativeVLMPreparedImagePreloadTasks[cacheKey] {
            emitVLMDiagnostic(
                "vlm_image_feature_preload_join key=\(cacheKey.rawValue.hashValue)"
            )
            let prepared = try await task.value
            nativeVLMPreparedImagePreloadTasks[cacheKey] = nil
            storeNativeVLMPreparedImageFeatures(prepared)
            return prepared.imageTokenCounts.reduce(0, +)
        }

        emitVLMDiagnostic(
            "vlm_image_feature_preload_begin key=\(cacheKey.rawValue.hashValue)"
        )
        let task = Task.detached(priority: .userInitiated) { [self] in
            try await self.prepareNativeVLMImageFeaturesForPreload(
                inputs: inputs,
                cacheKey: cacheKey,
                imageProcessorConfig: imageProcessorConfig,
                settings: imagePolicySettings
            )
        }
        nativeVLMPreparedImagePreloadTasks[cacheKey] = task
        do {
            let prepared = try await task.value
            nativeVLMPreparedImagePreloadTasks[cacheKey] = nil
            storeNativeVLMPreparedImageFeatures(prepared)
            emitVLMDiagnostic(
                "vlm_image_feature_preload_done key=\(cacheKey.rawValue.hashValue) imageTokens=\(prepared.imageTokenCounts) shape=\(prepared.shape)"
            )
            return prepared.imageTokenCounts.reduce(0, +)
        } catch {
            nativeVLMPreparedImagePreloadTasks[cacheKey] = nil
            emitVLMDiagnostic(
                "vlm_image_feature_preload_failed key=\(cacheKey.rawValue.hashValue) error=\(error)"
            )
            throw error
        }
    }

    public func tokenize(_ text: String) async throws -> [Int] {
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native VLM tokenizer is not initialized")
        }
        return tokenizer.encode(text: text)
    }

    public func renderNeuralImprintPrefix(
        profileBody: String,
        tools: [ToolSpec] = [],
        parameters requestedParameters: EdgeGenerateParameters = .default
    ) async throws -> NeuralImprintPrefixRender {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native VLM tokenizer is not initialized")
        }
        return try NeuralImprintRuntimeSupport.renderPrefix(
            profileBody: profileBody,
            tools: tools,
            parameters: requestedParameters,
            tokenizer: tokenizer,
            additionalContext: LLMEngine.chatTemplateContext(parameters:)
        )
    }

    /// Renders the exact text prompt token IDs used by text generation.
    ///
    /// Evaluation code should persist only a digest/count because token IDs are
    /// reversible.
    public func renderPromptTokenIDs(
        messages: [ChatMessage],
        tools: [ToolSpec]? = nil,
        parameters: EdgeGenerateParameters = .default
    ) async throws -> [Int] {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard let tokenizer = nativeTokenizer else {
            throw EdgeRuntimeError.loadFailed("Native VLM tokenizer is not initialized")
        }
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: parameters.preserveThinking
        )
        return try NeuralImprintRuntimeSupport.renderPromptTokenIDs(
            promptMessages: promptMessages,
            tools: tools,
            parameters: parameters,
            tokenizer: tokenizer,
            additionalContext: LLMEngine.chatTemplateContext(parameters:)
        )
    }

    @discardableResult
    public func restoreNeuralImprintCache(from directory: URL) throws -> NeuralImprintCacheStatus {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard let modelDirectory, let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }

        let status = try NeuralImprintRuntimeSupport.loadCacheStatus(
            directory: directory,
            modelDirectory: modelDirectory,
            architecture: container.index.languageIndex.architecture
        )
        activeNeuralImprintCache = status
        releaseNativeVLMCmlxSession(
            container: container,
            reason: "neural_imprint_restore_configured"
        )
        if container.isDecoderLoaded {
            _ = container.unloadDecoderWeights()
            emitVLMDiagnostic("vlm_swift_decoder_unload reason=neural_imprint_restore_configured")
        }
        promptCache.clear()
        clearNativeVLMPreparedImageCache()
        emitVLMDiagnostic(
            "vlm_neural_imprint_restore_configured prefix=\(status.prefixTokenCount) artifactSHA256=\(status.artifactSHA256)"
        )
        return status
    }

    public func unloadNeuralImprintCache() {
        activeNeuralImprintCache = nil
        if let container = nativeContainer {
            releaseNativeVLMCmlxSession(
                container: container,
                reason: "neural_imprint_unloaded"
            )
        } else {
            resetNativeVLMCmlxLedger()
        }
        promptCache.clear()
        clearNativeVLMPreparedImageCache()
    }

    @discardableResult
    public func captureNeuralImprintArtifact(
        request: NeuralImprintArtifactCaptureRequest
    ) async throws -> NeuralImprintCacheStatus {
        guard state == .ready else {
            throw EdgeRuntimeError.loadFailed("No VLM model loaded")
        }
        guard !request.prefixTokenIDs.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        guard let modelDirectory, let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }

        let architecture = container.index.languageIndex.architecture
        let prefixTokenCount = request.prefixTokenIDs.count
        let modelID = request.modelID ?? modelDirectory.lastPathComponent
        emitVLMDiagnostic("vlm_neural_imprint_capture_model_identity_begin")
        let modelIdentityHashes = try NeuralImprintRuntimeSupport.computeModelIdentityHashes(
            modelDirectory: modelDirectory
        )
        emitVLMDiagnostic("vlm_neural_imprint_capture_model_identity_done")

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
        emitVLMDiagnostic(
            "vlm_neural_imprint_capture_session_init_begin prefix=\(prefixTokenCount) prefillStep=\(capturePrefillStep) syncPrefill=\(captureUsesSyncPrefill) availableMB=\(captureMemorySnapshot.availableMB) footprintMB=\(captureMemorySnapshot.footprintMB) jetsamLimitMB=\(captureMemorySnapshot.jetsamLimitMB)"
        )
        if container.isCmlxDecoderLoaded {
            releaseNativeVLMCmlxSession(
                container: container,
                reason: "neural_imprint_capture_requires_clean_session"
            )
        }
        if container.isDecoderLoaded {
            _ = container.unloadDecoderWeights()
            emitVLMDiagnostic("vlm_swift_decoder_unload reason=neural_imprint_capture")
        }
        try applyNativeVLMCmlxCommandBufferLimits(contextLengthHint: capturePrefillStep)
        let session = try QwenCmlxLazyDecodeSession(
            bundleIndex: container.index.languageIndex,
            runtime: container.runtime
        )
        emitVLMDiagnostic(
            "vlm_neural_imprint_capture_session_init_done floats=\(session.registeredFloatTensorCount) quantized=\(session.registeredQuantizedTensorCount)"
        )
        return try NeuralImprintRuntimeSupport.captureArtifact(
            request: request,
            modelDirectory: modelDirectory,
            architecture: architecture,
            modelID: modelID,
            modelIdentityHashes: modelIdentityHashes,
            session: session,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill,
            diagnosticPrefix: "vlm_",
            diagnosticSink: { [weak self] marker in self?.emitVLMDiagnostic(marker) }
        )
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
        activeNeuralImprintCache = nil
        clearNativeVLMPreparedImageCache()
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

    static func neuralImprintCompatibleParameters(
        _ parameters: EdgeGenerateParameters
    ) -> EdgeGenerateParameters {
        NeuralImprintRuntimeSupport.neuralImprintCompatibleParameters(parameters)
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

    nonisolated static func nativeVLMImagePolicySettings(
        for policy: VLMImagePolicy,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> NativeVLMImagePolicySettings {
        let defaultMaxImageTokens: Int? = {
            switch policy {
            case .quality:
                return 768
            case .balanced:
                return 256
            case .fast:
                return 130
            case .custom(let maxImageTokens, _):
                return max(maxImageTokens, 1)
            }
        }()
        let defaultPruneTokens: Int? = {
            switch policy {
            case .quality, .balanced:
                return nil
            case .fast:
                return 64
            case .custom(_, let pruneTokens):
                guard let pruneTokens, pruneTokens > 0 else { return nil }
                return pruneTokens
            }
        }()
        let envBudget = NativeEnvironment.int(
            "EDGE_VLM_IMAGE_TOKEN_BUDGET",
            environment: environment
        )
        let envPrune = NativeEnvironment.int(
            "EDGE_VLM_IMAGE_FEATURE_PRUNE_TOKENS",
            environment: environment
        )
        let maxImageTokens = envBudget.map { $0 > 0 ? $0 : nil } ?? defaultMaxImageTokens
        let pruneTokens = envPrune.map { $0 > 0 ? $0 : nil } ?? defaultPruneTokens
        return NativeVLMImagePolicySettings(
            policy: policy,
            maxImageTokens: maxImageTokens,
            pruneTokens: pruneTokens,
            maxImageTokensOverriddenByEnvironment: envBudget != nil,
            pruneTokensOverriddenByEnvironment: envPrune != nil
        )
    }

    private func nativeVLMImageProcessorConfiguration(
        container: QwenVLMNativeContainer,
        settings: NativeVLMImagePolicySettings,
        emitDiagnostics: Bool
    ) -> QwenImageProcessorConfiguration {
        var imageProcessorConfig = container.index.preflightResult.plan.imageProcessorConfiguration
        if let imageTokenBudget = settings.maxImageTokens,
           imageTokenBudget > 0 {
            let plan = container.index.preflightResult.plan
            let patchSize = imageProcessorConfig.patchSize
                ?? plan.visionConfiguration.patchSize
                ?? 16
            let mergeSize = plan.visionConfiguration.spatialMergeSize
                ?? imageProcessorConfig.mergeSize
                ?? 2
            let maxPixels = imageTokenBudget * patchSize * patchSize * mergeSize * mergeSize
            let currentMax = imageProcessorConfig.maxPixels ?? (16_384 * 28 * 28)
            if maxPixels < currentMax {
                imageProcessorConfig.maxPixels = maxPixels
            }
            if emitDiagnostics {
                emitVLMDiagnostic(
                    "vlm_image_token_budget_apply \(settings.diagnosticSummary) budget=\(imageTokenBudget) patch=\(patchSize) merge=\(mergeSize) maxPixels=\(maxPixels) appliedMax=\(imageProcessorConfig.maxPixels ?? currentMax)"
                )
            }
        } else if emitDiagnostics {
            emitVLMDiagnostic("vlm_image_policy \(settings.diagnosticSummary)")
        }
        let deviceRAMGB = DeviceProfile.current.totalRAMGB
        if deviceRAMGB < 12 {
            let safeCap = 1280 * 28 * 28
            let currentMax = imageProcessorConfig.maxPixels ?? (16_384 * 28 * 28)
            if currentMax > safeCap {
                imageProcessorConfig.maxPixels = safeCap
                if emitDiagnostics {
                    NSLog("[VLM-MEM] clamped maxPixels %d → %d (device %d GB RAM)", currentMax, safeCap, deviceRAMGB)
                }
            }
        }
        return imageProcessorConfig
    }

    private func nativeVLMPreparedImageCacheKey(
        inputs: [NativeVLMImageInput],
        container: QwenVLMNativeContainer,
        imageProcessorConfig: QwenImageProcessorConfiguration,
        settings: NativeVLMImagePolicySettings
    ) -> NativeVLMPreparedImageKey? {
        guard !inputs.isEmpty else {
            return nil
        }
        var descriptors: [String] = []
        descriptors.reserveCapacity(inputs.count)
        for input in inputs {
            switch input {
            case .url(let url):
                let standardizedURL = url.standardizedFileURL
                let attributes = try? FileManager.default.attributesOfItem(
                    atPath: standardizedURL.path
                )
                let size = (attributes?[.size] as? NSNumber)?.int64Value ?? -1
                let modifiedAt = (attributes?[.modificationDate] as? Date)?
                    .timeIntervalSince1970 ?? -1
                let modifiedMs = Int64((modifiedAt * 1000).rounded())
                descriptors.append(
                    "url=\(standardizedURL.path)|size=\(size)|modifiedMs=\(modifiedMs)"
                )
            case .ciImage:
                return nil
            }
        }

        let plan = container.index.preflightResult.plan
        let patchSize = imageProcessorConfig.patchSize
            ?? plan.visionConfiguration.patchSize
            ?? 16
        let mergeSize = plan.visionConfiguration.spatialMergeSize
            ?? imageProcessorConfig.mergeSize
            ?? 2
        let pruneTokens = settings.pruneTokens ?? -1
        let modelKey = modelDirectory?.standardizedFileURL.path ?? "unknown"
        let rawValue = [
            "model=\(modelKey)",
            "maxPixels=\(imageProcessorConfig.maxPixels ?? -1)",
            "minPixels=\(imageProcessorConfig.minPixels ?? -1)",
            "patch=\(patchSize)",
            "merge=\(mergeSize)",
            "prune=\(pruneTokens)",
            "inputs=\(descriptors.joined(separator: ","))"
        ].joined(separator: ";")
        return NativeVLMPreparedImageKey(rawValue: rawValue)
    }

    private func storeNativeVLMPreparedImageFeatures(
        _ prepared: NativeVLMPreparedImageFeatures
    ) {
        guard let cacheKey = prepared.cacheKey else {
            return
        }
        nativeVLMPreparedImageCache[cacheKey] = prepared
        nativeVLMPreparedImageCacheOrder.removeAll { $0 == cacheKey }
        nativeVLMPreparedImageCacheOrder.append(cacheKey)
        while nativeVLMPreparedImageCacheOrder.count > 4 {
            let evicted = nativeVLMPreparedImageCacheOrder.removeFirst()
            nativeVLMPreparedImageCache.removeValue(forKey: evicted)
            emitVLMDiagnostic(
                "vlm_image_feature_cache_evict key=\(evicted.rawValue.hashValue)"
            )
        }
    }

    private func clearNativeVLMPreparedImageCache() {
        for task in nativeVLMPreparedImagePreloadTasks.values {
            task.cancel()
        }
        nativeVLMPreparedImagePreloadTasks.removeAll()
        nativeVLMPreparedImageCache.removeAll()
        nativeVLMPreparedImageCacheOrder.removeAll()
    }

    private func cachedOrPreloadedNativeVLMImageFeatures(
        cacheKey: NativeVLMPreparedImageKey?
    ) async throws -> NativeVLMPreparedImageFeatures? {
        guard let cacheKey else {
            return nil
        }
        if let cached = nativeVLMPreparedImageCache[cacheKey] {
            nativeVLMPreparedImageCacheOrder.removeAll { $0 == cacheKey }
            nativeVLMPreparedImageCacheOrder.append(cacheKey)
            emitVLMDiagnostic(
                "vlm_image_feature_cache_hit key=\(cacheKey.rawValue.hashValue) imageTokens=\(cached.imageTokenCounts) shape=\(cached.shape)"
            )
            return cached
        }
        guard let task = nativeVLMPreparedImagePreloadTasks[cacheKey] else {
            emitVLMDiagnostic(
                "vlm_image_feature_cache_miss key=\(cacheKey.rawValue.hashValue)"
            )
            return nil
        }
        emitVLMDiagnostic(
            "vlm_image_feature_preload_await key=\(cacheKey.rawValue.hashValue)"
        )
        do {
            let prepared = try await task.value
            nativeVLMPreparedImagePreloadTasks[cacheKey] = nil
            storeNativeVLMPreparedImageFeatures(prepared)
            emitVLMDiagnostic(
                "vlm_image_feature_preload_consume key=\(cacheKey.rawValue.hashValue) imageTokens=\(prepared.imageTokenCounts) shape=\(prepared.shape)"
            )
            return prepared
        } catch {
            nativeVLMPreparedImagePreloadTasks[cacheKey] = nil
            emitVLMDiagnostic(
                "vlm_image_feature_preload_consume_failed key=\(cacheKey.rawValue.hashValue) error=\(error)"
            )
            throw error
        }
    }

    private func preprocessNativeVLMImages(
        inputs: [NativeVLMImageInput],
        container: QwenVLMNativeContainer,
        imageProcessorConfig: QwenImageProcessorConfiguration,
        settings: NativeVLMImagePolicySettings,
        cacheKey: NativeVLMPreparedImageKey?,
        memoryBefore: Int
    ) throws -> NativeVLMPreprocessedImages {
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
        emitVLMDiagnostic(
            "vlm_mem after_image_preprocess mb=\(afterPreprocess) delta=\(afterPreprocess - memoryBefore)"
        )

        let resolvedPatchSize = imageProcessorConfig.patchSize
            ?? container.index.preflightResult.plan.visionConfiguration.patchSize
            ?? 16
        let targetSizes = preprocessings.map { item in
            "\(item.imageGridTHW.height * resolvedPatchSize)x\(item.imageGridTHW.width * resolvedPatchSize)"
        }
        emitVLMDiagnostic(
            "vlm_mem image_target_sizes sizes=\(targetSizes) patch=\(resolvedPatchSize)"
        )

        let patchDim = try Self.patchDimension(preprocessings)
        let totalPatchCount = preprocessings.reduce(0) { partial, item in
            partial + item.pixelValuesShape[0]
        }
        NSLog("[VLM-MEM] patches=%d patchDim=%d grids=%@", totalPatchCount, patchDim, preprocessings.map(\.imageGridTHW).description)
        emitVLMDiagnostic(
            "vlm_mem image_patches patches=\(totalPatchCount) patchDim=\(patchDim) grids=\(preprocessings.map(\.imageGridTHW).description)"
        )
        let pixelValues = preprocessings.reduce(into: [Float]()) { result, item in
            result.reserveCapacity(result.count + item.pixelValues.count)
            result.append(contentsOf: item.pixelValues)
        }
        let grids = preprocessings.map(\.imageGridTHW)
        let originalImageTokenCounts = try Self.imageTokenCounts(
            for: grids,
            plan: container.index.preflightResult.plan
        )
        let imageFeaturePrunePlan = NativeVLMImageFeaturePruner.makePlan(
            imageTokenCounts: originalImageTokenCounts,
            maxTokensPerImage: settings.pruneTokens
        )
        let imageTokenCounts = imageFeaturePrunePlan?.effectiveImageTokenCounts ?? originalImageTokenCounts
        let pixelValuesMB = Float(pixelValues.count * 4) / 1_048_576.0
        NSLog("[VLM-MEM] imageTokenCounts=%@, originalImageTokenCounts=%@, pixelValues=%.1f MB", imageTokenCounts.description, originalImageTokenCounts.description, pixelValuesMB)
        emitVLMDiagnostic(
            "vlm_mem image_tokens counts=\(imageTokenCounts) originalCounts=\(originalImageTokenCounts) pixelValuesMB=\(String(format: "%.1f", pixelValuesMB))"
        )
        if let imageFeaturePrunePlan, imageFeaturePrunePlan.isPruned {
            emitVLMDiagnostic(
                "vlm_image_feature_prune_plan strategy=uniform_row_major original=\(imageFeaturePrunePlan.originalImageTokenCounts) effective=\(imageFeaturePrunePlan.effectiveImageTokenCounts) selected=\(imageFeaturePrunePlan.selectedRowIndices.count)"
            )
        }
        return NativeVLMPreprocessedImages(
            cacheKey: cacheKey,
            pixelValues: pixelValues,
            patchDim: patchDim,
            totalPatchCount: totalPatchCount,
            grids: grids,
            originalImageTokenCounts: originalImageTokenCounts,
            imageTokenCounts: imageTokenCounts,
            imageFeaturePrunePlan: imageFeaturePrunePlan
        )
    }

    private func encodeNativeVLMImageFeatures(
        preprocessed: NativeVLMPreprocessedImages,
        container: QwenVLMNativeContainer,
        memoryBefore: Int,
        source: String
    ) throws -> NativeVLMPreparedImageFeatures {
        if container.isCmlxVisionLoaded {
            container.unloadCmlxVisionWeights()
        }
        let beforeVision = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] before vision encoder load: %.0f MB", beforeVision)
        emitVLMDiagnostic(
            "vlm_mem before_vision_encoder_load mb=\(beforeVision) deltaFromStart=\(beforeVision - memoryBefore)"
        )
        try container.loadCmlxVisionWeights(
            diagnosticSink: { [weak self] marker in
                self?.emitVLMDiagnostic(marker)
            }
        )
        let afterVisionLoad = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] after vision encoder load: %.0f MB (+%.0f)", afterVisionLoad, afterVisionLoad - beforeVision)
        emitVLMDiagnostic(
            "vlm_mem after_vision_encoder_load mb=\(afterVisionLoad) delta=\(afterVisionLoad - beforeVision)"
        )

        let visionEncoding: EdgeMLXQwen35VisionEncoding
        do {
            defer {
                container.unloadCmlxVisionWeights()
                let afterUnload = DeviceProfile.captureMemorySnapshot().footprintMB
                NSLog("[VLM-MEM] after vision encoder unload: %.0f MB", afterUnload)
                emitVLMDiagnostic(
                    "vlm_mem after_vision_encoder_unload mb=\(afterUnload) deltaFromStart=\(afterUnload - memoryBefore)"
                )
            }
            visionEncoding = try container.visionEncode(
                pixelValues: preprocessed.pixelValues,
                pixelValuesShape: [preprocessed.totalPatchCount, preprocessed.patchDim],
                gridTHW: preprocessed.grids
            )
            let afterVisionEncode = DeviceProfile.captureMemorySnapshot().footprintMB
            NSLog("[VLM-MEM] after vision encode: %.0f MB (+%.0f from load)", afterVisionEncode, afterVisionEncode - afterVisionLoad)
            emitVLMDiagnostic(
                "vlm_mem after_vision_encode mb=\(afterVisionEncode) deltaFromVisionLoad=\(afterVisionEncode - afterVisionLoad)"
            )
        }

        let originalTotalImageTokenCount = preprocessed.originalImageTokenCounts.reduce(0, +)
        guard visionEncoding.shape.first == originalTotalImageTokenCount else {
            throw EdgeRuntimeError.loadFailed("Native VLM vision encoder returned no image tokens")
        }
        let effectiveVisionEncoding = try NativeVLMImageFeaturePruner.apply(
            plan: preprocessed.imageFeaturePrunePlan,
            values: visionEncoding.values,
            shape: visionEncoding.shape
        )
        let totalImageTokenCount = preprocessed.imageTokenCounts.reduce(0, +)
        guard effectiveVisionEncoding.shape.first == totalImageTokenCount else {
            throw EdgeRuntimeError.loadFailed("Native VLM pruned image token count mismatch")
        }
        if let imageFeaturePrunePlan = preprocessed.imageFeaturePrunePlan,
           imageFeaturePrunePlan.isPruned {
            emitVLMDiagnostic(
                "vlm_image_feature_prune_applied originalShape=\(visionEncoding.shape) effectiveShape=\(effectiveVisionEncoding.shape)"
            )
        }
        return NativeVLMPreparedImageFeatures(
            cacheKey: preprocessed.cacheKey,
            values: effectiveVisionEncoding.values,
            shape: effectiveVisionEncoding.shape,
            imageTokenCounts: preprocessed.imageTokenCounts,
            source: source
        )
    }

    private func prepareNativeVLMImageFeaturesForPreload(
        inputs: [NativeVLMImageInput],
        cacheKey: NativeVLMPreparedImageKey,
        imageProcessorConfig: QwenImageProcessorConfiguration,
        settings: NativeVLMImagePolicySettings
    ) async throws -> NativeVLMPreparedImageFeatures {
        guard let container = nativeContainer else {
            throw EdgeRuntimeError.loadFailed("Native VLM runtime is not initialized")
        }
        try Task.checkCancellation()
        let memoryBefore = DeviceProfile.captureMemorySnapshot().footprintMB
        emitVLMDiagnostic("vlm_mem image_preload_start mb=\(memoryBefore)")
        let preprocessed = try preprocessNativeVLMImages(
            inputs: inputs,
            container: container,
            imageProcessorConfig: imageProcessorConfig,
            settings: settings,
            cacheKey: cacheKey,
            memoryBefore: memoryBefore
        )

        var didUnloadDecoderWeightsPreservingState = false
        if container.isCmlxDecoderLoaded {
            emitVLMDiagnostic("vlm_image_feature_preload_decoder_unload_begin")
            try container.unloadCmlxDecoderWeightsPreservingState()
            didUnloadDecoderWeightsPreservingState = true
            emitVLMDiagnostic("vlm_image_feature_preload_decoder_unload_done")
        }

        do {
            let prepared = try encodeNativeVLMImageFeatures(
                preprocessed: preprocessed,
                container: container,
                memoryBefore: memoryBefore,
                source: "preload"
            )
            if didUnloadDecoderWeightsPreservingState {
                emitVLMDiagnostic("vlm_image_feature_preload_decoder_reload_begin")
                try container.reloadCmlxDecoderWeightsPreservingState()
                emitVLMDiagnostic("vlm_image_feature_preload_decoder_reload_done")
            }
            return prepared
        } catch {
            if didUnloadDecoderWeightsPreservingState {
                emitVLMDiagnostic("vlm_image_feature_preload_decoder_reload_after_error_begin")
                try? container.reloadCmlxDecoderWeightsPreservingState()
                emitVLMDiagnostic("vlm_image_feature_preload_decoder_reload_after_error_done")
            }
            throw error
        }
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
        nativeCmlxLimitState.reset()
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
                "vlm_cmlx_session_init_begin dsrLayers=\(dsrPolicies.count) kv=\(NativeCmlxAttentionCacheQuantization.summary(attentionQuantization)) frog=0x\(String(frogJumpMask, radix: 16)) limit=\(attentionCacheLimit.map(String.init) ?? "nil")"
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

    private func applyNativeVLMCmlxCommandBufferLimits(contextLengthHint: Int) throws {
        try NativeCmlxCommandBufferLimitApplier.apply(
            contextLengthHint: contextLengthHint,
            state: &nativeCmlxLimitState,
            commandBufferDiagnosticName: "vlm_cmlx_command_buffer_limits",
            memoryLimitDiagnosticName: "vlm_cmlx_memory_limit",
            includeRequestedContext: true,
            emitDiagnostic: { self.emitVLMDiagnostic($0) }
        )
    }

    private func generateNativeText(
        messages: [ChatMessage],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try await self.nativeOperationSerializer.run {
                        try await self.runNativeTextGenerate(
                            messages: messages,
                            tools: tools,
                            requestedParameters: parameters,
                            continuation: continuation
                        )
                    }
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
        let neuralImprintStatus = activeNeuralImprintCache
        if let neuralImprintStatus,
           parameters.enableThinking != neuralImprintStatus.enableThinking {
            throw EdgeRuntimeError.unsupportedFeature(
                "VLM Neural Imprint enableThinking mismatch: artifact=\(neuralImprintStatus.enableThinking) request=\(parameters.enableThinking)"
            )
        }
        let promptTools = neuralImprintStatus == nil ? tools : nil
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: parameters.preserveThinking
        )
        let promptTokens = try NeuralImprintRuntimeSupport.renderPromptTokenIDs(
            promptMessages: promptMessages,
            tools: promptTools,
            parameters: parameters,
            tokenizer: tokenizer,
            additionalContext: LLMEngine.chatTemplateContext(parameters:)
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
            staticPolicy: memoryPolicy,
            dsrMaxCriticalOverride: requestedParameters.dsrMaxCritical
        )
        resolved.apply(to: &parameters)
        lastPolicy = resolved
        resolvedPolicy = resolved
        if resolved.shouldPause {
            throw EdgeRuntimeError.thermalPause
        }
        var policyReasonSuffix = ""
        if neuralImprintStatus != nil {
            parameters = Self.neuralImprintCompatibleParameters(parameters)
            policyReasonSuffix += " | neuralImprint full-cache restore disables DSR/KV-quant/FrogJump"
        }
        let dsrPolicies = try LLMEngine.makeDSRPolicies(
            parameters: parameters,
            architecture: architecture
        )
        let neuralImprintPrefixTokenCount = neuralImprintStatus?.prefixTokenCount ?? 0
        let kvCapacity = LLMEngine.kvCapacity(
            promptTokenCount: neuralImprintPrefixTokenCount + promptTokens.count,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let attentionQuantization = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: parameters,
            dsrPolicies: dsrPolicies
        )
        let frogJumpMask = QwenFrogJumpPlan.compute(
            architecture: architecture,
            requestedEnabled: parameters.frogJumpEnabled,
            thinkingEnabled: parameters.enableThinking
        ).layerMask
        let residentTokenCount = max(
            neuralImprintPrefixTokenCount + promptTokens.count,
            neuralImprintPrefixTokenCount + nativeVLMCmlxTokenIds.count
        )
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
            tools: promptTools,
            parameters: parameters,
            resolvedPolicy: resolvedPolicy,
            policyReasonSuffix: policyReasonSuffix,
            dsrPolicies: dsrPolicies,
            attentionQuantization: attentionQuantization,
            frogJumpMask: frogJumpMask,
            attentionCacheLimit: attentionCacheLimit,
            neuralImprintStatus: neuralImprintStatus,
            turn: turn,
            container: container,
            tokenizer: tokenizer,
            startedAt: startedAt,
            memoryBefore: memoryBefore,
            continuation: continuation
        ) {
            return
        }
        if neuralImprintStatus != nil {
            throw EdgeRuntimeError.unsupportedFeature(
                "VLM Neural Imprint restore requires CMLX text decode"
            )
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
            let sampling = NativeCmlxSampling.qwenSamplingConfiguration(
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
            promptTokenIDsSHA256: LLMEngine.neuralImprintTokenIDsSHA256(promptTokens),
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
        policyReasonSuffix: String,
        dsrPolicies: [Int: QwenDSRKVCachePolicy],
        attentionQuantization: NativeCmlxAttentionCacheQuantization?,
        frogJumpMask: UInt64,
        attentionCacheLimit: Int?,
        neuralImprintStatus: NeuralImprintCacheStatus?,
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
        let hadMatchingSession = neuralImprintStatus == nil &&
            container.isCmlxDecoderLoaded &&
            nativeVLMCmlxSessionMatches(
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

        if hadMatchingSession,
           NativeEnvironment.bool("EDGE_VLM_PRESERVE_STATE_UNLOAD_WEIGHTS_TEXT_SMOKE_EXPERIMENT") {
            let beforeSummary = (try? container.cmlxDecoderMemorySummary()) ?? "unavailable"
            emitVLMDiagnostic("vlm_scheme_d_text_smoke_unload_begin \(beforeSummary)")
            do {
                try container.unloadCmlxDecoderWeightsPreservingState()
                let afterUnloadSummary = (try? container.cmlxDecoderMemorySummary()) ?? "unavailable"
                emitVLMDiagnostic("vlm_scheme_d_text_smoke_unload_done \(afterUnloadSummary)")
                try container.reloadCmlxDecoderWeightsPreservingState()
                let afterReloadSummary = (try? container.cmlxDecoderMemorySummary()) ?? "unavailable"
                emitVLMDiagnostic("vlm_scheme_d_text_smoke_reload_done \(afterReloadSummary)")
            } catch {
                emitVLMDiagnostic(
                    "vlm_cmlx_text_reject reason=scheme_d_text_smoke_failed error=\(error)"
                )
                releaseNativeVLMCmlxSession(
                    container: container,
                    reason: "scheme_d_text_smoke_failed"
                )
                return false
            }
        }

        var prefillTokens = promptTokens
        var cachedTokensReused = 0
        var promptCacheHit = false
        if let neuralImprintStatus {
            try container.restoreCmlxNeuralImprintCache(
                artifactURL: neuralImprintStatus.artifactURL,
                prefixTokenCount: neuralImprintStatus.prefixTokenCount
            )
            nativeVLMCmlxTokenIds = []
            nativeVLMCmlxContainsMediaContext = false
            emitVLMDiagnostic(
                "vlm_cmlx_neural_imprint_restore prefix=\(neuralImprintStatus.prefixTokenCount) artifactSHA256=\(neuralImprintStatus.artifactSHA256)"
            )
        } else if hadMatchingSession {
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
        let samplingPenaltyApplier = NativeCmlxSampling.PenaltyApplier(
            parameters: parameters,
            endTokenIds: nativeEndTokenIds,
            setSamplingPenalties: { repetitionPenalty, repetitionTokenIds, presencePenalty, presenceTokenIds, frequencyPenalty, frequencyTokenIds in
                try container.setCmlxSamplingPenalties(
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextTokenIds: repetitionTokenIds,
                    presencePenalty: presencePenalty,
                    presenceContextTokenIds: presenceTokenIds,
                    frequencyPenalty: frequencyPenalty,
                    frequencyContextTokenIds: frequencyTokenIds
                )
            },
            setEOSSamplingBias: { tokenIds, suppress, logitPenalty in
                try container.setCmlxEOSSamplingBias(
                    tokenIds: tokenIds,
                    suppress: suppress,
                    logitPenalty: logitPenalty
                )
            },
            clearEOSSamplingBias: {
                try container.clearCmlxEOSSamplingBias()
            }
        )
        let useSamplingPenalties = useSampledPath &&
            samplingPenaltyApplier.samplingPenaltiesAreActive
        let useEOSSamplingBias = useSampledPath &&
            samplingPenaltyApplier.eosSamplingBiasRequested &&
            !nativeEndTokenIds.isEmpty
        defer {
            if useSamplingPenalties { try? container.clearCmlxRepetitionPenalty() }
            if useEOSSamplingBias { try? container.clearCmlxEOSSamplingBias() }
        }

        if neuralImprintStatus == nil && !promptCacheHit {
            _ = try container.resetCmlxDecoder()
            nativeVLMCmlxTokenIds = []
            nativeVLMCmlxContainsMediaContext = false
        }
        let neuralImprintPrefixTokenCount = neuralImprintStatus?.prefixTokenCount ?? 0
        let prefillMode = neuralImprintStatus != nil
            ? "neural_imprint"
            : (promptCacheHit ? "incremental" : "full")
        emitVLMDiagnostic(
            "vlm_cmlx_text_prefill_begin tokens=\(prefillTokens.count) mode=\(prefillMode) cached=\(cachedTokensReused) neuralImprintPrefix=\(neuralImprintPrefixTokenCount) decode=\(useSampledPath ? "sampled" : "greedy")"
        )
        let residentPromptTokensAfterPrefill = nativeVLMCmlxTokenIds + prefillTokens
        if useSamplingPenalties {
            try samplingPenaltyApplier.applySamplingPenalties(
                promptSessionTokenIds: residentPromptTokensAfterPrefill
            )
        }
        if useEOSSamplingBias {
            try samplingPenaltyApplier.applyEOSSamplingBias(generatedTokenCount: 0)
        }
        try Task.checkCancellation()
        try applyNativeVLMCmlxCommandBufferLimits(
            contextLengthHint: neuralImprintPrefixTokenCount + nativeVLMCmlxTokenIds.count + prefillTokens.count
        )
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
                        try samplingPenaltyApplier.applySamplingPenalties(
                            promptSessionTokenIds: nativeVLMCmlxTokenIds,
                            generatedTokenIds: generatedTokenIds
                        )
                    }
                    if useEOSSamplingBias {
                        try samplingPenaltyApplier.applyEOSSamplingBias(
                            generatedTokenCount: generatedTokenIds.count
                        )
                    }
                    try applyNativeVLMCmlxCommandBufferLimits(
                        contextLengthHint: neuralImprintPrefixTokenCount +
                            nativeVLMCmlxTokenIds.count +
                            generatedTokenIds.count
                    )
                    nextTokenID = try container.nextSampledCmlxToken(
                        temperature: parameters.temperature,
                        topK: cmlxTopK,
                        topP: cmlxTopP,
                        minP: cmlxMinP,
                        seed: cmlxSamplingRNG.next()
                    )
                } else {
                    try applyNativeVLMCmlxCommandBufferLimits(
                        contextLengthHint: neuralImprintPrefixTokenCount +
                            nativeVLMCmlxTokenIds.count +
                            generatedTokenIds.count
                    )
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
            promptTokenIDsSHA256: LLMEngine.neuralImprintTokenIDsSHA256(promptTokens),
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: (resolvedPolicy?.reasoning ?? "policy unavailable")
                + policyReasonSuffix
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

    private func runNativeVLMImageFeaturePrefill(
        container: QwenVLMNativeContainer,
        tokenIDs: [Int],
        imageFeatures: [Float],
        imageFeatureShape: [Int],
        imageTokenID: Int,
        parameters: EdgeGenerateParameters
    ) throws -> Int {
        let chunkSize = max(1, parameters.prefillStepSize)
        guard tokenIDs.count > chunkSize,
              let firstImageTokenIndex = tokenIDs.firstIndex(of: imageTokenID),
              firstImageTokenIndex > chunkSize
        else {
            emitVLMDiagnostic(
                "vlm_cmlx_media_prefill_direct_begin tokens=\(tokenIDs.count) imageFeatures=\(imageFeatureShape)"
            )
            let nextTokenID = try container.prefillImageFeatures(
                tokenIDs: tokenIDs,
                imageFeatures: imageFeatures,
                imageFeatureShape: imageFeatureShape,
                imageTokenID: imageTokenID
            )
            emitVLMDiagnostic(
                "vlm_cmlx_media_prefill_direct_done tokens=\(tokenIDs.count)"
            )
            return nextTokenID
        }

        let prefixTokens = Array(tokenIDs[..<firstImageTokenIndex])
        let mediaAndSuffixTokens = Array(tokenIDs[firstImageTokenIndex...])
        let prefixChunkCount = Int(ceil(Double(prefixTokens.count) / Double(chunkSize)))
        emitVLMDiagnostic(
            "vlm_cmlx_media_prefill_chunked total=\(tokenIDs.count) prefix=\(prefixTokens.count) mediaSuffix=\(mediaAndSuffixTokens.count) step=\(chunkSize) prefixChunks=\(prefixChunkCount)"
        )

        var offset = 0
        var chunkIndex = 0
        while offset < prefixTokens.count {
            let end = min(offset + chunkSize, prefixTokens.count)
            let chunk = Array(prefixTokens[offset..<end])
            chunkIndex += 1
            try container.prefillCmlxTokensAsync(tokenIDs: chunk)
            emitVLMDiagnostic(
                "vlm_cmlx_media_prefill_prefix_chunk_done index=\(chunkIndex) tokens=\(chunk.count) final=false async=true"
            )
            offset = end
        }

        emitVLMDiagnostic(
            "vlm_cmlx_media_prefill_media_chunk_begin tokens=\(mediaAndSuffixTokens.count) imageFeatures=\(imageFeatureShape)"
        )
        let nextTokenID = try container.prefillImageFeatures(
            tokenIDs: mediaAndSuffixTokens,
            imageFeatures: imageFeatures,
            imageFeatureShape: imageFeatureShape,
            imageTokenID: imageTokenID
        )
        emitVLMDiagnostic(
            "vlm_cmlx_media_prefill_media_chunk_done tokens=\(mediaAndSuffixTokens.count) final=true"
        )
        return nextTokenID
    }

    private func generateNativeImage(
        messages: [ChatMessage],
        inputs: [NativeVLMImageInput],
        tools: [ToolSpec]?,
        parameters: EdgeGenerateParameters,
        imagePolicy: VLMImagePolicy
    ) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    try await self.nativeOperationSerializer.run {
                        try await self.runNativeImageGenerate(
                            messages: messages,
                            inputs: inputs,
                            tools: tools,
                            imagePolicy: imagePolicy,
                            requestedParameters: parameters,
                            continuation: continuation
                        )
                    }
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
        imagePolicy: VLMImagePolicy,
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

        var parameters = requestedParameters
        let neuralImprintStatus = activeNeuralImprintCache
        if let neuralImprintStatus,
           parameters.enableThinking != neuralImprintStatus.enableThinking {
            throw EdgeRuntimeError.unsupportedFeature(
                "VLM Neural Imprint enableThinking mismatch: artifact=\(neuralImprintStatus.enableThinking) request=\(parameters.enableThinking)"
            )
        }
        let promptTools = neuralImprintStatus == nil ? tools : nil

        let startedAt = Date()
        let memoryBefore = DeviceProfile.captureMemorySnapshot().footprintMB
        NSLog("[VLM-MEM] image generate start: %.0f MB", memoryBefore)
        emitVLMDiagnostic("vlm_mem image_generate_start mb=\(memoryBefore)")

        let imagePolicySettings = Self.nativeVLMImagePolicySettings(
            for: imagePolicy
        )
        emitVLMDiagnostic("vlm_image_policy \(imagePolicySettings.diagnosticSummary)")
        let imageProcessorConfig = nativeVLMImageProcessorConfiguration(
            container: container,
            settings: imagePolicySettings,
            emitDiagnostics: true
        )
        let preparedImageCacheKey = nativeVLMPreparedImageCacheKey(
            inputs: inputs,
            container: container,
            imageProcessorConfig: imageProcessorConfig,
            settings: imagePolicySettings
        )
        var preparedImageFeatures = try await cachedOrPreloadedNativeVLMImageFeatures(
            cacheKey: preparedImageCacheKey
        )
        let preprocessedImages: NativeVLMPreprocessedImages?
        if let preparedImageFeatures {
            preprocessedImages = nil
            emitVLMDiagnostic(
                "vlm_image_feature_prepared_reuse source=\(preparedImageFeatures.source) imageTokens=\(preparedImageFeatures.imageTokenCounts) shape=\(preparedImageFeatures.shape)"
            )
        } else {
            preprocessedImages = try preprocessNativeVLMImages(
                inputs: inputs,
                container: container,
                imageProcessorConfig: imageProcessorConfig,
                settings: imagePolicySettings,
                cacheKey: preparedImageCacheKey,
                memoryBefore: memoryBefore
            )
        }
        let preserveStateUnloadWeightsExperiment = NativeEnvironment.bool(
            "EDGE_VLM_PRESERVE_STATE_UNLOAD_WEIGHTS_EXPERIMENT"
        )
        let imageTokenCounts = preparedImageFeatures?.imageTokenCounts
            ?? preprocessedImages?.imageTokenCounts
            ?? []
        let totalImageTokenCount = imageTokenCounts.reduce(0, +)
        let imageTokenID = try Self.tokenID(
            "<|image_pad|>",
            tokenizer: tokenizer,
            fallback: 248_056
        )
        let promptMessages = messages.promptCacheMessages(
            preserveThinking: parameters.preserveThinking
        )
        let mediaPromptMessages = try Self.makeNativeVLMImagePromptMessages(
            messages: promptMessages,
            imageTokenCounts: imageTokenCounts
        )
        let promptTokens = try Self.makeNativeVLMPromptTokens(
            messages: promptMessages,
            imageTokenCounts: imageTokenCounts,
            tokenizer: tokenizer,
            tools: promptTools,
            parameters: parameters
        )
        let placeholderCount = promptTokens.reduce(0) { count, token in
            count + (token == imageTokenID ? 1 : 0)
        }
        guard totalImageTokenCount > 0,
              placeholderCount == totalImageTokenCount
        else {
            throw EdgeRuntimeError.loadFailed(
                "Native VLM prompt image token count mismatch: prompt=\(placeholderCount), features=\(totalImageTokenCount)"
            )
        }

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
                cachedTokenCount: preserveStateUnloadWeightsExperiment ? nativeVLMCmlxTokenIds.count : 0,
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
        if resolved.shouldPause {
            throw EdgeRuntimeError.thermalPause
        }
        var policyReasonSuffix = ""
        if neuralImprintStatus != nil {
            parameters = Self.neuralImprintCompatibleParameters(parameters)
            policyReasonSuffix += " | neuralImprint full-cache restore disables DSR/KV-quant/FrogJump"
        }
        let dsrPolicies = try LLMEngine.makeDSRPolicies(
            parameters: parameters,
            architecture: architecture
        )
        let neuralImprintPrefixTokenCount = neuralImprintStatus?.prefixTokenCount ?? 0
        let kvCapacity = LLMEngine.kvCapacity(
            promptTokenCount: neuralImprintPrefixTokenCount + promptTokens.count,
            maxTokenCount: parameters.maxTokens,
            parameters: parameters,
            architecture: architecture
        )
        let attentionQuantization = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: parameters,
            dsrPolicies: dsrPolicies
        )
        let frogJumpMask = QwenFrogJumpPlan.compute(
            architecture: architecture,
            requestedEnabled: parameters.frogJumpEnabled,
            thinkingEnabled: parameters.enableThinking
        ).layerMask
        let attentionCacheLimit = parameters.maxKVSize ?? parameters.dsrMaxCritical ?? kvCapacity

        var didUnloadDecoderWeightsPreservingState = false
        var appendPlan: NativeVLMImageAppendPlan?
        func emitCmlxDecoderMemorySummary(_ label: String) {
            if let summary = try? container.cmlxDecoderMemorySummary() {
                emitVLMDiagnostic("vlm_cmlx_memory_summary label=\(label) \(summary)")
            } else {
                emitVLMDiagnostic("vlm_cmlx_memory_summary label=\(label) unavailable")
            }
        }

        try Task.checkCancellation()
        if container.isCmlxDecoderLoaded {
            if neuralImprintStatus != nil {
                releaseNativeVLMCmlxSession(
                    container: container,
                    reason: "neural_imprint_media_generate_before_vision_load"
                )
                promptCache.clear()
            } else {
                let shouldTryAppendPlan = preserveStateUnloadWeightsExperiment || preparedImageFeatures != nil
                if shouldTryAppendPlan {
                    let canReuseSession = nativeVLMCmlxSessionMatches(
                        dsrPolicies: dsrPolicies,
                        attentionQuantization: attentionQuantization,
                        frogJumpMask: frogJumpMask,
                        attentionCacheLimit: attentionCacheLimit
                    )
                    let planResult: NativeVLMImageAppendPlanResult
                    if !(promptTools?.isEmpty ?? true) {
                        planResult = .failure("tools_present")
                    } else if canReuseSession {
                        planResult = NativeVLMImageAppendPlanner.makePlan(
                            cachedTokenIds: nativeVLMCmlxTokenIds,
                            currentPromptTokenIds: promptTokens,
                            skippableCachedTokenSequences: LLMEngine.qwenThinkingSentinelTokenSequences(
                                tokenizer: tokenizer
                            ),
                            previousPromptMessages: nativeVLMCmlxPromptMessages,
                            lastAssistantText: nativeVLMCmlxLastAssistantText,
                            currentMediaPromptMessages: mediaPromptMessages,
                            imageTokenID: imageTokenID,
                            totalImageTokenCount: totalImageTokenCount,
                            enableThinking: parameters.enableThinking,
                            encodeSuffix: { text in
                                tokenizer.encode(text: text, addSpecialTokens: false)
                            }
                        )
                    } else {
                        planResult = .failure("session_config_mismatch")
                    }

                    switch planResult {
                    case .success(let plan):
                        appendPlan = plan
                        emitVLMDiagnostic(
                            "vlm_scheme_d_append_plan_hit cached=\(plan.cachedTokensReused) suffix=\(plan.suffixTokenIds.count) fullPrompt=\(promptTokens.count)"
                        )
                        if preparedImageFeatures == nil {
                            emitVLMDiagnostic(
                                "vlm_scheme_d_unload_begin reason=media_append_before_vision_load"
                            )
                            emitCmlxDecoderMemorySummary("scheme_d_before_unload")
                            do {
                                try container.unloadCmlxDecoderWeightsPreservingState()
                                didUnloadDecoderWeightsPreservingState = true
                                emitCmlxDecoderMemorySummary("scheme_d_after_unload")
                                emitVLMDiagnostic("vlm_scheme_d_unload_done")
                            } catch {
                                appendPlan = nil
                                emitVLMDiagnostic(
                                    "vlm_scheme_d_fallback reason=unload_failed error=\(error)"
                                )
                                releaseNativeVLMCmlxSession(
                                    container: container,
                                    reason: "scheme_d_unload_failed"
                                )
                                promptCache.clear()
                            }
                        } else {
                            emitVLMDiagnostic(
                                "vlm_scheme_d_unload_skip reason=prepared_image_features"
                            )
                        }
                    case .failure(let reason):
                        emitVLMDiagnostic(
                            "vlm_scheme_d_fallback reason=\(reason)"
                        )
                        if preparedImageFeatures != nil {
                            emitVLMDiagnostic(
                                "vlm_prepared_hit_append_fallback reason=\(reason)"
                            )
                            emitVLMDiagnostic(
                                "vlm_cmlx_session_preserve reason=prepared_image_features_append_plan_failed"
                            )
                        } else {
                            releaseNativeVLMCmlxSession(
                                container: container,
                                reason: "scheme_d_append_plan_failed"
                            )
                            promptCache.clear()
                        }
                    }
                } else if NativeEnvironment.bool("EDGE_VLM_PRESERVE_DECODER_FOR_VISION_EXPERIMENT") {
                    emitVLMDiagnostic(
                        "vlm_cmlx_session_preserve reason=media_generate_before_vision_load_experiment"
                    )
                } else if preparedImageFeatures != nil {
                    emitVLMDiagnostic(
                        "vlm_cmlx_session_preserve reason=prepared_image_features"
                    )
                } else {
                    releaseNativeVLMCmlxSession(
                        container: container,
                        reason: "media_generate_before_vision_load"
                    )
                    promptCache.clear()
                }
            }
        }
        if preparedImageFeatures == nil {
            guard let preprocessedImages else {
                throw EdgeRuntimeError.loadFailed("Native VLM image preprocessing result missing")
            }
            let encoded = try encodeNativeVLMImageFeatures(
                preprocessed: preprocessedImages,
                container: container,
                memoryBefore: memoryBefore,
                source: "generate"
            )
            preparedImageFeatures = encoded
            storeNativeVLMPreparedImageFeatures(encoded)
            if encoded.cacheKey != nil {
                emitVLMDiagnostic(
                    "vlm_image_feature_cache_store imageTokens=\(encoded.imageTokenCounts) shape=\(encoded.shape)"
                )
            }
        } else {
            emitVLMDiagnostic(
                "vlm_image_feature_encode_skip reason=prepared_image_features"
            )
        }

        if didUnloadDecoderWeightsPreservingState {
            emitVLMDiagnostic("vlm_scheme_d_reload_begin")
            emitCmlxDecoderMemorySummary("scheme_d_before_reload")
            do {
                try container.reloadCmlxDecoderWeightsPreservingState()
                emitCmlxDecoderMemorySummary("scheme_d_after_reload")
                emitVLMDiagnostic("vlm_scheme_d_reload_done")
            } catch {
                emitVLMDiagnostic(
                    "vlm_scheme_d_fallback reason=reload_failed error=\(error)"
                )
                releaseNativeVLMCmlxSession(
                    container: container,
                    reason: "scheme_d_reload_failed"
                )
                promptCache.clear()
                didUnloadDecoderWeightsPreservingState = false
                appendPlan = nil
            }
        }

        try applyNativeVLMCmlxCommandBufferLimits(
            contextLengthHint: neuralImprintPrefixTokenCount +
                (appendPlan?.cachedTokensReused ?? 0) +
                (appendPlan?.suffixTokenIds.count ?? promptTokens.count) +
                parameters.maxTokens
        )
        emitVLMDiagnostic(
            "vlm_cmlx_media_policy prompt=\(promptTokens.count) neuralImprintPrefix=\(neuralImprintPrefixTokenCount) maxTokens=\(parameters.maxTokens) kvCapacity=\(kvCapacity) limit=\(attentionCacheLimit) useDSR=\(parameters.useDSR) dsrMax=\(parameters.dsrMaxCritical.map(String.init) ?? "nil") prefill=\(parameters.prefillStepSize) policy=\(resolved.reasoning + policyReasonSuffix)"
        )
        try prepareNativeVLMCmlxSession(
            container: container,
            dsrPolicies: dsrPolicies,
            attentionQuantization: attentionQuantization,
            frogJumpMask: frogJumpMask,
            attentionCacheLimit: attentionCacheLimit
        )
        let prefillTokenIds: [Int]
        let promptCacheHit: Bool
        let cachedTokensReused: Int
        if let neuralImprintStatus {
            try container.restoreCmlxNeuralImprintCache(
                artifactURL: neuralImprintStatus.artifactURL,
                prefixTokenCount: neuralImprintStatus.prefixTokenCount
            )
            nativeVLMCmlxTokenIds = []
            nativeVLMCmlxPromptMessages = []
            nativeVLMCmlxLastAssistantText = ""
            nativeVLMCmlxContainsMediaContext = false
            prefillTokenIds = promptTokens
            promptCacheHit = false
            cachedTokensReused = 0
            emitVLMDiagnostic(
                "vlm_cmlx_neural_imprint_restore prefix=\(neuralImprintStatus.prefixTokenCount) artifactSHA256=\(neuralImprintStatus.artifactSHA256)"
            )
        } else if let appendPlan {
            prefillTokenIds = appendPlan.suffixTokenIds
            promptCacheHit = true
            cachedTokensReused = appendPlan.cachedTokensReused
            emitVLMDiagnostic(
                "vlm_scheme_d_append_prefill_begin cached=\(cachedTokensReused) suffix=\(prefillTokenIds.count) fullPrompt=\(promptTokens.count)"
            )
        } else {
            _ = try container.resetCmlxDecoder()
            nativeVLMCmlxTokenIds = []
            nativeVLMCmlxPromptMessages = []
            nativeVLMCmlxLastAssistantText = ""
            nativeVLMCmlxContainsMediaContext = false
            prefillTokenIds = promptTokens
            promptCacheHit = false
            cachedTokensReused = 0
        }

        try Task.checkCancellation()
        guard let preparedImageFeatures else {
            throw EdgeRuntimeError.loadFailed("Native VLM prepared image features missing")
        }
        var nextTokenID: Int? = try runNativeVLMImageFeaturePrefill(
            container: container,
            tokenIDs: prefillTokenIds,
            imageFeatures: preparedImageFeatures.values,
            imageFeatureShape: preparedImageFeatures.shape,
            imageTokenID: imageTokenID,
            parameters: parameters
        )
        emitVLMDiagnostic(
            "vlm_cmlx_media_first_token_ready tokenAvailable=\(nextTokenID != nil)"
        )
        let residentPromptTokenCountAfterPrefill = (appendPlan != nil)
            ? nativeVLMCmlxTokenIds.count + prefillTokenIds.count
            : prefillTokenIds.count
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

            try applyNativeVLMCmlxCommandBufferLimits(
                contextLengthHint: neuralImprintPrefixTokenCount +
                    residentPromptTokenCountAfterPrefill +
                    generatedTokenIds.count
            )
            nextTokenID = try container.decodeCmlxStep(tokenID: tokenID)
        }

        let endedAt = Date()
        let decodeSeconds = max(endedAt.timeIntervalSince(firstTokenAt), 0.001)
        let memoryAfter = DeviceProfile.captureMemorySnapshot().footprintMB
        turnCounter = turn
        if appendPlan != nil {
            nativeVLMCmlxTokenIds.append(contentsOf: prefillTokenIds)
            nativeVLMCmlxTokenIds.append(contentsOf: generatedTokenIds)
        } else {
            nativeVLMCmlxTokenIds = promptTokens + generatedTokenIds
        }
        nativeVLMCmlxPromptMessages = promptMessages
        nativeVLMCmlxLastAssistantText = NativePromptSessionReuse.normalizeAssistantText(emittedText)
        nativeVLMCmlxContainsMediaContext = true
        emitVLMDiagnostic(
            "vlm_cmlx_media_session_seeded prompt=\(promptTokens.count) prefill=\(prefillTokenIds.count) generated=\(generatedTokenIds.count) total=\(nativeVLMCmlxTokenIds.count) append=\(appendPlan != nil) neuralImprintPrefix=\(neuralImprintPrefixTokenCount)"
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
            promptTokenIDsSHA256: LLMEngine.neuralImprintTokenIDsSHA256(promptTokens),
            generationTokenCount: generatedTokenIds.count,
            memoryBeforeMB: memoryBefore,
            memoryAfterMB: memoryAfter,
            policyReasoning: resolved.reasoning + policyReasonSuffix + " | nativeVLMImageCmlx=on",
            promptCacheHit: promptCacheHit,
            cachedTokensReused: cachedTokensReused,
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
        let promptMessages = try makeNativeVLMImagePromptMessages(
            messages: messages,
            imageTokenCounts: imageTokenCounts
        )
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

    private static func makeNativeVLMImagePromptMessages(
        messages: [ChatMessage],
        imageTokenCounts: [Int]
    ) throws -> [ChatMessage] {
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
        return promptMessages
    }

    private func _generateNativePending(_ message: String) -> AsyncThrowingStream<GenerateChunk, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: EdgeRuntimeError.unsupportedFeature(message))
        }
    }
}
