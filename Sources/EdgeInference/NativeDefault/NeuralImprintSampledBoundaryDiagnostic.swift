// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

#if DEBUG
import CryptoKit
import EdgeEngine
import Foundation
import Tokenizers

public struct NeuralImprintSampledBoundaryComparison: Codable, Sendable, Equatable {
    public let originalAlignedTokenIDsEqual: Bool
    public let originalNITokenIDsEqual: Bool
    public let alignedNITokenIDsEqual: Bool
    public let originalAlignedFirstTokenDifference: Int?
    public let originalNIFirstTokenDifference: Int?
    public let alignedNIFirstTokenDifference: Int?
}

public struct NeuralImprintSampledBoundaryFirstTokenProbe: Codable, Sendable, Equatable {
    public let selectedTokenID: Int
    public let logitMargin: Double?
    public let sampleDiagnostics: String
}

public struct NeuralImprintSampledBoundaryResult: Codable, Sendable, Equatable {
    public let schemaVersion: String
    public let prefixTokenCount: Int
    public let suffixTokenCount: Int
    public let visibleInputTokenCount: Int
    public let visibleTokensEqualPrefixPlusSuffix: Bool
    public let prefillStep: Int
    public let captureUsesSyncPrefill: Bool
    public let samplingSeed: String
    public let temperature: Float
    public let topK: Int?
    public let topP: Float
    public let minP: Float
    public let repetitionPenalty: Float
    public let repetitionContextSize: Int
    public let presencePenalty: Float
    public let presenceContextSize: Int
    public let frequencyPenalty: Float
    public let frequencyContextSize: Int
    public let stopOnEndToken: Bool
    public let minimumGeneratedTokens: Int
    public let eosPenaltyUntilToken: Int
    public let maxTokens: Int
    public let selectedMode: String?
    public let pathOrder: [String]
    public let sampleDiagnosticsAvailable: Bool
    public let paths: [NeuralImprintGreedyPathReceipt]
    public let comparison: NeuralImprintSampledBoundaryComparison?
    public let firstTokenProbesByMode: [String: NeuralImprintSampledBoundaryFirstTokenProbe]
}

enum NeuralImprintSampledBoundarySupport {
    static let schemaVersion = "edge-kit.neural_imprint_sampled_boundary.v1"
    static let originalMode = "visible_profile_original_boundary"
    static let alignedMode = "visible_profile_aligned_boundary"
    static let neuralImprintMode = "neural_imprint_normalized"

    private struct GeneratedPath {
        let receipt: NeuralImprintGreedyPathReceipt
        let tokenIDs: [Int]
        let sampleDiagnostics: [String]
        let decodedText: String
    }

    static func run(
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime,
        tokenizer: Tokenizer,
        endTokenIDs: Set<Int>,
        prefixTokenIDs: [Int],
        suffixTokenIDs: [Int],
        visibleTokenIDs: [Int],
        parameters: EdgeGenerateParameters,
        samplingSeed: UInt64,
        prefillStep: Int,
        selectedMode: String? = nil,
        outputTextSink: (([String: String]) -> Void)? = nil
    ) throws -> NeuralImprintSampledBoundaryResult {
        let session = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )

        let allModes = [originalMode, alignedMode, neuralImprintMode]
        let requestedModes = selectedMode.map { [$0] } ?? allModes
        guard requestedModes.allSatisfy(allModes.contains) else {
            throw EdgeRuntimeError.loadFailed("Unknown sampled boundary diagnostic mode")
        }
        var paths: [GeneratedPath] = []
        paths.reserveCapacity(requestedModes.count)
        for mode in requestedModes {
            try session.reset()
            switch mode {
            case originalMode:
                paths.append(
                    try generate(
                        mode: mode,
                        session: session,
                        tokenizer: tokenizer,
                        inputTokenIDs: visibleTokenIDs,
                        penaltyContextTokenIDs: visibleTokenIDs,
                        parameters: parameters,
                        samplingSeed: samplingSeed,
                        endTokenIDs: endTokenIDs,
                        prefillStep: prefillStep
                    )
                )
            case alignedMode:
                paths.append(
                    try generate(
                        mode: mode,
                        session: session,
                        tokenizer: tokenizer,
                        inputTokenIDs: visibleTokenIDs,
                        penaltyContextTokenIDs: visibleTokenIDs,
                        parameters: parameters,
                        samplingSeed: samplingSeed,
                        endTokenIDs: endTokenIDs,
                        prefillStep: prefillStep,
                        prefillSplitIndices: [prefixTokenIDs.count]
                    )
                )
            case neuralImprintMode:
                try prefillCapturedPrefix(
                    session: session,
                    tokenIDs: prefixTokenIDs,
                    chunkSize: prefillStep
                )
                paths.append(
                    try generate(
                        mode: mode,
                        session: session,
                        tokenizer: tokenizer,
                        inputTokenIDs: suffixTokenIDs,
                        // Production NI sampling penalties currently see the suffix tokens,
                        // not the serialized prefix token IDs. Preserve that behavior here.
                        penaltyContextTokenIDs: suffixTokenIDs,
                        parameters: parameters,
                        samplingSeed: samplingSeed,
                        endTokenIDs: endTokenIDs,
                        prefillStep: prefillStep
                    )
                )
            default:
                preconditionFailure("validated sampled boundary diagnostic mode")
            }
        }
        outputTextSink?(
            Dictionary(uniqueKeysWithValues: paths.map { ($0.receipt.mode, $0.decodedText) })
        )

        return NeuralImprintSampledBoundaryResult(
            schemaVersion: schemaVersion,
            prefixTokenCount: prefixTokenIDs.count,
            suffixTokenCount: suffixTokenIDs.count,
            visibleInputTokenCount: visibleTokenIDs.count,
            visibleTokensEqualPrefixPlusSuffix:
                visibleTokenIDs == prefixTokenIDs + suffixTokenIDs,
            prefillStep: prefillStep,
            captureUsesSyncPrefill: false,
            samplingSeed: String(samplingSeed),
            temperature: parameters.temperature,
            topK: parameters.topK,
            topP: normalizedTopP(parameters.topP),
            minP: parameters.minP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionContextSize: parameters.repetitionContextSize,
            presencePenalty: parameters.presencePenalty,
            presenceContextSize: parameters.presenceContextSize,
            frequencyPenalty: parameters.frequencyPenalty,
            frequencyContextSize: parameters.frequencyContextSize,
            stopOnEndToken: parameters.stopOnEndToken,
            minimumGeneratedTokens: parameters.minimumGeneratedTokens,
            eosPenaltyUntilToken: parameters.eosPenaltyUntilToken,
            maxTokens: parameters.maxTokens,
            selectedMode: selectedMode,
            pathOrder: paths.map(\.receipt.mode),
            sampleDiagnosticsAvailable: paths.allSatisfy {
                !$0.sampleDiagnostics.isEmpty && $0.sampleDiagnostics.allSatisfy { !$0.isEmpty }
            },
            paths: paths.map(\.receipt),
            comparison: comparison(for: paths),
            firstTokenProbesByMode: Dictionary(
                uniqueKeysWithValues: paths.compactMap { path in
                    guard let tokenID = path.tokenIDs.first,
                          let diagnostics = path.sampleDiagnostics.first
                    else { return nil }
                    return (
                        path.receipt.mode,
                        NeuralImprintSampledBoundaryFirstTokenProbe(
                            selectedTokenID: tokenID,
                            logitMargin:
                                NeuralImprintGreedyEquivalenceSupport.diagnosticValue(
                                    named: "margin",
                                    in: diagnostics
                                ),
                            sampleDiagnostics: diagnostics
                        )
                    )
                }
            )
        )
    }

    private static func comparison(
        for paths: [GeneratedPath]
    ) -> NeuralImprintSampledBoundaryComparison? {
        let byMode = Dictionary(uniqueKeysWithValues: paths.map { ($0.receipt.mode, $0) })
        guard let original = byMode[originalMode],
              let aligned = byMode[alignedMode],
              let neuralImprint = byMode[neuralImprintMode]
        else { return nil }
        return compare(
            original: original.tokenIDs,
            aligned: aligned.tokenIDs,
            neuralImprint: neuralImprint.tokenIDs
        )
    }

    static func compare(
        original: [Int],
        aligned: [Int],
        neuralImprint: [Int]
    ) -> NeuralImprintSampledBoundaryComparison {
        NeuralImprintSampledBoundaryComparison(
            originalAlignedTokenIDsEqual: original == aligned,
            originalNITokenIDsEqual: original == neuralImprint,
            alignedNITokenIDsEqual: aligned == neuralImprint,
            originalAlignedFirstTokenDifference: firstDifference(original, aligned),
            originalNIFirstTokenDifference: firstDifference(original, neuralImprint),
            alignedNIFirstTokenDifference: firstDifference(aligned, neuralImprint)
        )
    }

    private static func firstDifference(_ lhs: [Int], _ rhs: [Int]) -> Int? {
        let sharedCount = min(lhs.count, rhs.count)
        if let index = (0..<sharedCount).first(where: { lhs[$0] != rhs[$0] }) {
            return index
        }
        return lhs.count == rhs.count ? nil : sharedCount
    }

    private static func generate(
        mode: String,
        session: QwenCmlxLazyDecodeSession,
        tokenizer: Tokenizer,
        inputTokenIDs: [Int],
        penaltyContextTokenIDs: [Int],
        parameters: EdgeGenerateParameters,
        samplingSeed: UInt64,
        endTokenIDs: Set<Int>,
        prefillStep: Int,
        prefillSplitIndices: [Int] = []
    ) throws -> GeneratedPath {
        try? session.clearRepetitionPenalty()
        try? session.clearEOSSamplingBias()
        let penaltyApplier = NativeCmlxSampling.PenaltyApplier(
            parameters: parameters,
            endTokenIds: endTokenIDs,
            setSamplingPenalties: {
                repetitionPenalty,
                repetitionTokenIDs,
                presencePenalty,
                presenceTokenIDs,
                frequencyPenalty,
                frequencyTokenIDs in
                try session.setSamplingPenalties(
                    repetitionPenalty: repetitionPenalty,
                    repetitionContextTokenIds: repetitionTokenIDs,
                    presencePenalty: presencePenalty,
                    presenceContextTokenIds: presenceTokenIDs,
                    frequencyPenalty: frequencyPenalty,
                    frequencyContextTokenIds: frequencyTokenIDs
                )
            },
            setEOSSamplingBias: { tokenIDs, suppress, logitPenalty in
                try session.setEOSSamplingBias(
                    tokenIds: tokenIDs,
                    suppress: suppress,
                    logitPenalty: logitPenalty
                )
            },
            clearEOSSamplingBias: {
                try session.clearEOSSamplingBias()
            }
        )
        if penaltyApplier.samplingPenaltiesAreActive {
            try penaltyApplier.applySamplingPenalties(
                promptSessionTokenIds: penaltyContextTokenIDs
            )
        }
        if penaltyApplier.eosSamplingBiasRequested, !endTokenIDs.isEmpty {
            try penaltyApplier.applyEOSSamplingBias(generatedTokenCount: 0)
        }
        defer {
            try? session.clearRepetitionPenalty()
            try? session.clearEOSSamplingBias()
        }

        var rng = EdgeSeededRandomNumberGenerator(seed: samplingSeed)
        var nextTokenID = try prefillForSampledDecode(
            session: session,
            tokenIDs: inputTokenIDs,
            parameters: parameters,
            rng: &rng,
            chunkSize: prefillStep,
            splitIndices: prefillSplitIndices
        )
        var generatedTokenIDs: [Int] = []
        var sampleDiagnostics: [String] = []
        generatedTokenIDs.reserveCapacity(parameters.maxTokens)
        sampleDiagnostics.reserveCapacity(parameters.maxTokens)

        while generatedTokenIDs.count < parameters.maxTokens,
              (!parameters.stopOnEndToken || !endTokenIDs.contains(nextTokenID)) {
            generatedTokenIDs.append(nextTokenID)
            sampleDiagnostics.append((try session.lastSampleDiagnostics()) ?? "")
            guard generatedTokenIDs.count < parameters.maxTokens else { break }
            if penaltyApplier.samplingPenaltiesAreActive {
                try penaltyApplier.applySamplingPenalties(
                    promptSessionTokenIds: penaltyContextTokenIDs,
                    generatedTokenIds: generatedTokenIDs
                )
            }
            if penaltyApplier.eosSamplingBiasRequested, !endTokenIDs.isEmpty {
                try penaltyApplier.applyEOSSamplingBias(
                    generatedTokenCount: generatedTokenIDs.count
                )
            }
            nextTokenID = try session.nextSampledToken(
                temperature: parameters.temperature,
                topK: parameters.topK,
                topP: normalizedTopP(parameters.topP),
                minP: parameters.minP,
                seed: rng.next()
            )
        }

        let decodedText = tokenizer.decode(
            tokens: generatedTokenIDs,
            skipSpecialTokens: true
        )
        return GeneratedPath(
            receipt: NeuralImprintGreedyPathReceipt(
                mode: mode,
                inputTokenCount: inputTokenIDs.count,
                generatedTokenCount: generatedTokenIDs.count,
                tokenIDsSHA256: sha256JSON(generatedTokenIDs),
                textSHA256: sha256Text(decodedText)
            ),
            tokenIDs: generatedTokenIDs,
            sampleDiagnostics: sampleDiagnostics,
            decodedText: decodedText
        )
    }

    private static func prefillForSampledDecode(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        parameters: EdgeGenerateParameters,
        rng: inout EdgeSeededRandomNumberGenerator,
        chunkSize: Int,
        splitIndices: [Int]
    ) throws -> Int {
        let ranges = NeuralImprintGreedyEquivalenceSupport.prefillChunkRanges(
            tokenCount: tokenIDs.count,
            chunkSize: chunkSize,
            splitIndices: splitIndices
        )
        guard !ranges.isEmpty else {
            throw QwenHybridModelReferenceError.emptyTokenIds
        }
        for (index, range) in ranges.enumerated() {
            let chunk = Array(tokenIDs[range])
            if index == ranges.count - 1 {
                try session.prefillSampledAsync(
                    tokenIDs: chunk,
                    temperature: parameters.temperature,
                    topK: parameters.topK,
                    topP: normalizedTopP(parameters.topP),
                    minP: parameters.minP,
                    seed: rng.next()
                )
            } else {
                try session.prefillAsync(tokenIDs: chunk)
            }
        }
        return try session.nextSampledToken(
            temperature: parameters.temperature,
            topK: parameters.topK,
            topP: normalizedTopP(parameters.topP),
            minP: parameters.minP,
            seed: rng.next()
        )
    }

    private static func prefillCapturedPrefix(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        chunkSize: Int
    ) throws {
        let ranges = NeuralImprintGreedyEquivalenceSupport.prefillChunkRanges(
            tokenCount: tokenIDs.count,
            chunkSize: chunkSize,
            splitIndices: []
        )
        for range in ranges {
            try session.prefillAsync(tokenIDs: Array(tokenIDs[range]))
        }
    }

    private static func normalizedTopP(_ topP: Float) -> Float {
        topP > 0 && topP <= 1 ? topP : 1
    }

    private static func sha256JSON(_ tokenIDs: [Int]) -> String {
        let data = (try? JSONEncoder().encode(tokenIDs)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Text(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
#endif
