// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import EdgeEngine
import Foundation
import Tokenizers

#if DEBUG
public struct NeuralImprintGreedyPathReceipt: Codable, Sendable, Equatable {
    public let mode: String
    public let inputTokenCount: Int
    public let generatedTokenCount: Int
    public let tokenIDsSHA256: String
    public let textSHA256: String

    public init(
        mode: String,
        inputTokenCount: Int,
        generatedTokenCount: Int,
        tokenIDsSHA256: String,
        textSHA256: String
    ) {
        self.mode = mode
        self.inputTokenCount = inputTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.tokenIDsSHA256 = tokenIDsSHA256
        self.textSHA256 = textSHA256
    }
}

public struct NeuralImprintGreedyDivergenceProbe: Codable, Sendable, Equatable {
    public let tokenIndex: Int
    public let selectedTokenIDsByMode: [String: Int]
    public let logitMarginsByMode: [String: Double]
    public let sampleDiagnosticsByMode: [String: String]
}

public struct NeuralImprintGreedyPathComparison: Codable, Sendable, Equatable {
    public let visibleLiveTokenIDsEqual: Bool
    public let liveRestoredTokenIDsEqual: Bool
    public let visibleRestoredTokenIDsEqual: Bool
    public let allThreeTokenIDsEqual: Bool
    public let visibleLiveFirstTokenDifference: Int?
    public let liveRestoredFirstTokenDifference: Int?
    public let visibleRestoredFirstTokenDifference: Int?
}

public struct NeuralImprintForcedTokenPathReceipt: Codable, Sendable, Equatable {
    public let mode: String
    public let stateSource: String
    public let interventionTokenSource: String
    public let interventionIndex: Int
    public let interventionTokenID: Int
    public let generatedTokenCount: Int
    public let postInterventionTokenCount: Int
    public let tokenIDsSHA256: String
    public let postInterventionTokenIDsSHA256: String
    public let textSHA256: String
}

public struct NeuralImprintForcedTokenPairComparison: Codable, Sendable, Equatable {
    public let comparisonKind: String
    public let leftMode: String
    public let rightMode: String
    public let postInterventionTokenIDsEqual: Bool
    public let postInterventionFirstTokenDifference: Int?
    public let postInterventionEditDistance: Int
    public let postInterventionNormalizedEditDistance: Double
}

public struct NeuralImprintForcedTokenCrossoverResult: Codable, Sendable, Equatable {
    public let schemaVersion: String
    public let interventionIndex: Int
    public let visibleTokenID: Int
    public let liveCacheTokenID: Int
    public let naturalReplayMatchesBaselineByState: [String: Bool]
    public let paths: [NeuralImprintForcedTokenPathReceipt]
    public let pairComparisons: [NeuralImprintForcedTokenPairComparison]
}

public struct NeuralImprintGreedyEquivalenceResult: Codable, Sendable, Equatable {
    public let schemaVersion: String
    public let prefixTokenCount: Int
    public let suffixTokenCount: Int
    public let visibleInputTokenCount: Int
    public let visibleTokensEqualPrefixPlusSuffix: Bool
    public let visiblePrefillStep: Int
    public let capturePrefillStep: Int
    public let captureUsesSyncPrefill: Bool
    public let visiblePrefillSplitAtPrefixBoundary: Bool
    public let sampleDiagnosticsAvailable: Bool
    public let paths: [NeuralImprintGreedyPathReceipt]
    public let comparison: NeuralImprintGreedyPathComparison
    public let divergenceProbes: [NeuralImprintGreedyDivergenceProbe]
    public let forcedTokenCrossover: NeuralImprintForcedTokenCrossoverResult?
}

enum NeuralImprintGreedyEquivalenceSupport {
    static let schemaVersion = "edge-kit.neural_imprint_greedy_equivalence.v4"
    static let crossoverSchemaVersion =
        "edge-kit.neural_imprint_forced_token_crossover.v1"

    private struct GeneratedPath {
        let receipt: NeuralImprintGreedyPathReceipt
        let tokenIDs: [Int]
        let sampleDiagnostics: [String]
        let decodedText: String
    }

    private struct ForcedGeneratedPath {
        let receipt: NeuralImprintForcedTokenPathReceipt
        let tokenIDs: [Int]
    }

    static func run(
        bundleIndex: QwenModelBundleIndex,
        runtime: EdgeMetalRuntime,
        tokenizer: Tokenizer,
        endTokenIDs: Set<Int>,
        prefixTokenIDs: [Int],
        suffixTokenIDs: [Int],
        visibleTokenIDs: [Int],
        artifactURL: URL,
        maxTokens: Int,
        visiblePrefillStep: Int,
        capturePrefillStep: Int,
        captureUsesSyncPrefill: Bool,
        includeForcedTokenCrossover: Bool = true,
        alignVisiblePrefixBoundary: Bool = false,
        outputTextSink: (([String: String]) -> Void)? = nil
    ) throws -> NeuralImprintGreedyEquivalenceResult {
        let session = try QwenCmlxLazyDecodeSession(
            bundleIndex: bundleIndex,
            runtime: runtime
        )

        try session.reset()
        let visible = try generate(
            mode: "visible_profile_chat",
            session: session,
            tokenizer: tokenizer,
            inputTokenIDs: visibleTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            prefillStep: visiblePrefillStep,
            prefillSplitIndices:
                alignVisiblePrefixBoundary ? [prefixTokenIDs.count] : []
        )

        try session.reset()
        try prefillCapturedPrefix(
            session: session,
            tokenIDs: prefixTokenIDs,
            chunkSize: capturePrefillStep,
            synchronous: captureUsesSyncPrefill
        )
        let live = try generate(
            mode: "same_session_live_cache_chat",
            session: session,
            tokenizer: tokenizer,
            inputTokenIDs: suffixTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            prefillStep: visiblePrefillStep
        )

        try session.reset()
        try prefillCapturedPrefix(
            session: session,
            tokenIDs: prefixTokenIDs,
            chunkSize: capturePrefillStep,
            synchronous: captureUsesSyncPrefill
        )
        try session.session.saveNeuralImprintCache(
            artifactURL: artifactURL,
            metadata: [
                "artifact_type": "neural_imprint_greedy_equivalence_probe",
                "prefix_token_count": String(prefixTokenIDs.count),
                "schema_version": schemaVersion,
            ]
        )
        try session.reset()
        try session.restoreNeuralImprintCache(
            artifactURL: artifactURL,
            prefixTokenCount: prefixTokenIDs.count
        )
        let restored = try generate(
            mode: "serialized_restored_cache_chat",
            session: session,
            tokenizer: tokenizer,
            inputTokenIDs: suffixTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            prefillStep: visiblePrefillStep
        )

        outputTextSink?([
            visible.receipt.mode: visible.decodedText,
            live.receipt.mode: live.decodedText,
            restored.receipt.mode: restored.decodedText,
        ])

        let comparison = compare(
            visible: visible.tokenIDs,
            live: live.tokenIDs,
            restored: restored.tokenIDs
        )
        let probeIndices = Set([
            0,
            comparison.visibleLiveFirstTokenDifference,
            comparison.liveRestoredFirstTokenDifference,
            comparison.visibleRestoredFirstTokenDifference,
        ].compactMap { $0 }).sorted()
        let crossoverResult = try (
            includeForcedTokenCrossover
                ? comparison.visibleLiveFirstTokenDifference
                : nil
        ).flatMap {
            try forcedTokenCrossover(
                interventionIndex: $0,
                session: session,
                tokenizer: tokenizer,
                prefixTokenIDs: prefixTokenIDs,
                suffixTokenIDs: suffixTokenIDs,
                visibleTokenIDs: visibleTokenIDs,
                visibleNaturalTokenIDs: visible.tokenIDs,
                liveNaturalTokenIDs: live.tokenIDs,
                maxTokens: maxTokens,
                endTokenIDs: endTokenIDs,
                visiblePrefillStep: visiblePrefillStep,
                capturePrefillStep: capturePrefillStep,
                captureUsesSyncPrefill: captureUsesSyncPrefill
            )
        }
        return NeuralImprintGreedyEquivalenceResult(
            schemaVersion: schemaVersion,
            prefixTokenCount: prefixTokenIDs.count,
            suffixTokenCount: suffixTokenIDs.count,
            visibleInputTokenCount: visibleTokenIDs.count,
            visibleTokensEqualPrefixPlusSuffix:
                visibleTokenIDs == prefixTokenIDs + suffixTokenIDs,
            visiblePrefillStep: visiblePrefillStep,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill,
            visiblePrefillSplitAtPrefixBoundary: alignVisiblePrefixBoundary,
            sampleDiagnosticsAvailable: [visible, live, restored].allSatisfy {
                !$0.sampleDiagnostics.isEmpty && $0.sampleDiagnostics.allSatisfy { !$0.isEmpty }
            },
            paths: [visible.receipt, live.receipt, restored.receipt],
            comparison: comparison,
            divergenceProbes: probeIndices.compactMap { index in
                divergenceProbe(
                    index: index,
                    paths: [visible, live, restored]
                )
            },
            forcedTokenCrossover: crossoverResult
        )
    }

    private static func forcedTokenCrossover(
        interventionIndex: Int,
        session: QwenCmlxLazyDecodeSession,
        tokenizer: Tokenizer,
        prefixTokenIDs: [Int],
        suffixTokenIDs: [Int],
        visibleTokenIDs: [Int],
        visibleNaturalTokenIDs: [Int],
        liveNaturalTokenIDs: [Int],
        maxTokens: Int,
        endTokenIDs: Set<Int>,
        visiblePrefillStep: Int,
        capturePrefillStep: Int,
        captureUsesSyncPrefill: Bool
    ) throws -> NeuralImprintForcedTokenCrossoverResult? {
        guard interventionIndex > 0,
              visibleNaturalTokenIDs.indices.contains(interventionIndex),
              liveNaturalTokenIDs.indices.contains(interventionIndex)
        else {
            return nil
        }
        let visibleTokenID = visibleNaturalTokenIDs[interventionIndex]
        let liveTokenID = liveNaturalTokenIDs[interventionIndex]
        guard visibleTokenID != liveTokenID else { return nil }

        let visibleVisible = try generateForcedCrossoverPath(
            mode: "visible_state_visible_token",
            stateSource: "visible_profile",
            interventionTokenSource: "visible_profile",
            interventionTokenID: visibleTokenID,
            interventionIndex: interventionIndex,
            session: session,
            tokenizer: tokenizer,
            prefixTokenIDs: prefixTokenIDs,
            suffixTokenIDs: suffixTokenIDs,
            visibleTokenIDs: visibleTokenIDs,
            expectedNaturalTokenIDs: visibleNaturalTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            visiblePrefillStep: visiblePrefillStep,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill
        )
        let visibleLive = try generateForcedCrossoverPath(
            mode: "visible_state_live_cache_token",
            stateSource: "visible_profile",
            interventionTokenSource: "live_cache",
            interventionTokenID: liveTokenID,
            interventionIndex: interventionIndex,
            session: session,
            tokenizer: tokenizer,
            prefixTokenIDs: prefixTokenIDs,
            suffixTokenIDs: suffixTokenIDs,
            visibleTokenIDs: visibleTokenIDs,
            expectedNaturalTokenIDs: visibleNaturalTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            visiblePrefillStep: visiblePrefillStep,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill
        )
        let liveLive = try generateForcedCrossoverPath(
            mode: "live_cache_state_live_cache_token",
            stateSource: "live_cache",
            interventionTokenSource: "live_cache",
            interventionTokenID: liveTokenID,
            interventionIndex: interventionIndex,
            session: session,
            tokenizer: tokenizer,
            prefixTokenIDs: prefixTokenIDs,
            suffixTokenIDs: suffixTokenIDs,
            visibleTokenIDs: visibleTokenIDs,
            expectedNaturalTokenIDs: liveNaturalTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            visiblePrefillStep: visiblePrefillStep,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill
        )
        let liveVisible = try generateForcedCrossoverPath(
            mode: "live_cache_state_visible_token",
            stateSource: "live_cache",
            interventionTokenSource: "visible_profile",
            interventionTokenID: visibleTokenID,
            interventionIndex: interventionIndex,
            session: session,
            tokenizer: tokenizer,
            prefixTokenIDs: prefixTokenIDs,
            suffixTokenIDs: suffixTokenIDs,
            visibleTokenIDs: visibleTokenIDs,
            expectedNaturalTokenIDs: liveNaturalTokenIDs,
            maxTokens: maxTokens,
            endTokenIDs: endTokenIDs,
            visiblePrefillStep: visiblePrefillStep,
            capturePrefillStep: capturePrefillStep,
            captureUsesSyncPrefill: captureUsesSyncPrefill
        )

        let paths = [visibleVisible, visibleLive, liveLive, liveVisible]
        let pairComparisons = [
            compareCrossoverTokens(
                comparisonKind: "natural_replay",
                leftMode: "visible_profile_chat",
                left: visibleNaturalTokenIDs,
                rightMode: visibleVisible.receipt.mode,
                right: visibleVisible.tokenIDs,
                interventionIndex: interventionIndex
            ),
            compareCrossoverTokens(
                comparisonKind: "natural_replay",
                leftMode: "same_session_live_cache_chat",
                left: liveNaturalTokenIDs,
                rightMode: liveLive.receipt.mode,
                right: liveLive.tokenIDs,
                interventionIndex: interventionIndex
            ),
            compareCrossoverTokens(
                comparisonKind: "same_state",
                leftMode: visibleVisible.receipt.mode,
                left: visibleVisible.tokenIDs,
                rightMode: visibleLive.receipt.mode,
                right: visibleLive.tokenIDs,
                interventionIndex: interventionIndex
            ),
            compareCrossoverTokens(
                comparisonKind: "same_state",
                leftMode: liveLive.receipt.mode,
                left: liveLive.tokenIDs,
                rightMode: liveVisible.receipt.mode,
                right: liveVisible.tokenIDs,
                interventionIndex: interventionIndex
            ),
            compareCrossoverTokens(
                comparisonKind: "same_token",
                leftMode: visibleVisible.receipt.mode,
                left: visibleVisible.tokenIDs,
                rightMode: liveVisible.receipt.mode,
                right: liveVisible.tokenIDs,
                interventionIndex: interventionIndex
            ),
            compareCrossoverTokens(
                comparisonKind: "same_token",
                leftMode: visibleLive.receipt.mode,
                left: visibleLive.tokenIDs,
                rightMode: liveLive.receipt.mode,
                right: liveLive.tokenIDs,
                interventionIndex: interventionIndex
            ),
        ]
        return NeuralImprintForcedTokenCrossoverResult(
            schemaVersion: crossoverSchemaVersion,
            interventionIndex: interventionIndex,
            visibleTokenID: visibleTokenID,
            liveCacheTokenID: liveTokenID,
            naturalReplayMatchesBaselineByState: [
                "visible_profile": visibleVisible.tokenIDs == visibleNaturalTokenIDs,
                "live_cache": liveLive.tokenIDs == liveNaturalTokenIDs,
            ],
            paths: paths.map(\.receipt),
            pairComparisons: pairComparisons
        )
    }

    static func compare(
        visible: [Int],
        live: [Int],
        restored: [Int]
    ) -> NeuralImprintGreedyPathComparison {
        let visibleLiveDifference = firstDifference(visible, live)
        let liveRestoredDifference = firstDifference(live, restored)
        let visibleRestoredDifference = firstDifference(visible, restored)
        return NeuralImprintGreedyPathComparison(
            visibleLiveTokenIDsEqual: visibleLiveDifference == nil,
            liveRestoredTokenIDsEqual: liveRestoredDifference == nil,
            visibleRestoredTokenIDsEqual: visibleRestoredDifference == nil,
            allThreeTokenIDsEqual:
                visibleLiveDifference == nil && liveRestoredDifference == nil,
            visibleLiveFirstTokenDifference: visibleLiveDifference,
            liveRestoredFirstTokenDifference: liveRestoredDifference,
            visibleRestoredFirstTokenDifference: visibleRestoredDifference
        )
    }

    static func firstDifference(_ left: [Int], _ right: [Int]) -> Int? {
        for index in 0..<min(left.count, right.count) where left[index] != right[index] {
            return index
        }
        return left.count == right.count ? nil : min(left.count, right.count)
    }

    static func compareCrossoverTokens(
        comparisonKind: String,
        leftMode: String,
        left: [Int],
        rightMode: String,
        right: [Int],
        interventionIndex: Int
    ) -> NeuralImprintForcedTokenPairComparison {
        let firstPostInterventionIndex = max(0, interventionIndex + 1)
        let leftPostIntervention = Array(
            left.dropFirst(min(firstPostInterventionIndex, left.count))
        )
        let rightPostIntervention = Array(
            right.dropFirst(min(firstPostInterventionIndex, right.count))
        )
        let distance = editDistance(leftPostIntervention, rightPostIntervention)
        let denominator = max(
            1,
            max(leftPostIntervention.count, rightPostIntervention.count)
        )
        return NeuralImprintForcedTokenPairComparison(
            comparisonKind: comparisonKind,
            leftMode: leftMode,
            rightMode: rightMode,
            postInterventionTokenIDsEqual: leftPostIntervention == rightPostIntervention,
            postInterventionFirstTokenDifference: firstDifference(
                leftPostIntervention,
                rightPostIntervention
            ),
            postInterventionEditDistance: distance,
            postInterventionNormalizedEditDistance:
                Double(distance) / Double(denominator)
        )
    }

    static func editDistance(_ left: [Int], _ right: [Int]) -> Int {
        guard !left.isEmpty else { return right.count }
        guard !right.isEmpty else { return left.count }
        var previous = Array(0...right.count)
        var current = Array(repeating: 0, count: right.count + 1)
        for (leftIndex, leftToken) in left.enumerated() {
            current[0] = leftIndex + 1
            for (rightIndex, rightToken) in right.enumerated() {
                current[rightIndex + 1] = min(
                    previous[rightIndex + 1] + 1,
                    current[rightIndex] + 1,
                    previous[rightIndex] + (leftToken == rightToken ? 0 : 1)
                )
            }
            swap(&previous, &current)
        }
        return previous[right.count]
    }

    private static func generateForcedCrossoverPath(
        mode: String,
        stateSource: String,
        interventionTokenSource: String,
        interventionTokenID: Int,
        interventionIndex: Int,
        session: QwenCmlxLazyDecodeSession,
        tokenizer: Tokenizer,
        prefixTokenIDs: [Int],
        suffixTokenIDs: [Int],
        visibleTokenIDs: [Int],
        expectedNaturalTokenIDs: [Int],
        maxTokens: Int,
        endTokenIDs: Set<Int>,
        visiblePrefillStep: Int,
        capturePrefillStep: Int,
        captureUsesSyncPrefill: Bool
    ) throws -> ForcedGeneratedPath {
        try session.reset()
        let inputTokenIDs: [Int]
        switch stateSource {
        case "visible_profile":
            inputTokenIDs = visibleTokenIDs
        case "live_cache":
            try prefillCapturedPrefix(
                session: session,
                tokenIDs: prefixTokenIDs,
                chunkSize: capturePrefillStep,
                synchronous: captureUsesSyncPrefill
            )
            inputTokenIDs = suffixTokenIDs
        default:
            throw EdgeRuntimeError.loadFailed(
                "Unknown forced-token crossover state source: \(stateSource)"
            )
        }

        var naturalTokenID = try prefillForGreedyDecode(
            session: session,
            tokenIDs: inputTokenIDs,
            chunkSize: visiblePrefillStep
        )
        var generatedTokenIDs: [Int] = []
        generatedTokenIDs.reserveCapacity(maxTokens)
        for index in 0..<interventionIndex {
            guard expectedNaturalTokenIDs.indices.contains(index),
                  naturalTokenID == expectedNaturalTokenIDs[index]
            else {
                throw EdgeRuntimeError.loadFailed(
                    "Forced-token crossover natural-prefix replay mismatch at index \(index)"
                )
            }
            generatedTokenIDs.append(naturalTokenID)
            if index + 1 < interventionIndex {
                naturalTokenID = try session.nextToken()
            }
        }

        generatedTokenIDs.append(interventionTokenID)
        if generatedTokenIDs.count < maxTokens,
           !endTokenIDs.contains(interventionTokenID) {
            // The state has consumed token interventionIndex - 1 and holds the
            // natural token at interventionIndex as pending. Advancing with the
            // selected intervention token replaces that pending choice while
            // preserving the normal async decode schedule for the continuation.
            try session.prefillAsync(tokenIDs: [interventionTokenID])
            while generatedTokenIDs.count < maxTokens {
                let tokenID = try session.nextToken()
                guard !endTokenIDs.contains(tokenID) else { break }
                generatedTokenIDs.append(tokenID)
            }
        }

        let postInterventionTokenIDs = Array(
            generatedTokenIDs.dropFirst(
                min(interventionIndex + 1, generatedTokenIDs.count)
            )
        )
        let decodedText = tokenizer.decode(
            tokens: generatedTokenIDs,
            skipSpecialTokens: true
        )
        return ForcedGeneratedPath(
            receipt: NeuralImprintForcedTokenPathReceipt(
                mode: mode,
                stateSource: stateSource,
                interventionTokenSource: interventionTokenSource,
                interventionIndex: interventionIndex,
                interventionTokenID: interventionTokenID,
                generatedTokenCount: generatedTokenIDs.count,
                postInterventionTokenCount: postInterventionTokenIDs.count,
                tokenIDsSHA256: sha256JSON(generatedTokenIDs),
                postInterventionTokenIDsSHA256: sha256JSON(
                    postInterventionTokenIDs
                ),
                textSHA256: sha256Text(decodedText)
            ),
            tokenIDs: generatedTokenIDs
        )
    }

    private static func divergenceProbe(
        index: Int,
        paths: [GeneratedPath]
    ) -> NeuralImprintGreedyDivergenceProbe? {
        let available = paths.filter {
            $0.tokenIDs.indices.contains(index)
                && $0.sampleDiagnostics.indices.contains(index)
        }
        guard available.count == paths.count else { return nil }
        return NeuralImprintGreedyDivergenceProbe(
            tokenIndex: index,
            selectedTokenIDsByMode: Dictionary(
                uniqueKeysWithValues: available.map {
                    ($0.receipt.mode, $0.tokenIDs[index])
                }
            ),
            logitMarginsByMode: Dictionary(
                uniqueKeysWithValues: available.compactMap { path in
                    diagnosticValue(
                        named: "margin",
                        in: path.sampleDiagnostics[index]
                    ).map { (path.receipt.mode, $0) }
                }
            ),
            sampleDiagnosticsByMode: Dictionary(
                uniqueKeysWithValues: available.map {
                    ($0.receipt.mode, $0.sampleDiagnostics[index])
                }
            )
        )
    }

    private static func generate(
        mode: String,
        session: QwenCmlxLazyDecodeSession,
        tokenizer: Tokenizer,
        inputTokenIDs: [Int],
        maxTokens: Int,
        endTokenIDs: Set<Int>,
        prefillStep: Int,
        prefillSplitIndices: [Int] = []
    ) throws -> GeneratedPath {
        var nextTokenID = try prefillForGreedyDecode(
            session: session,
            tokenIDs: inputTokenIDs,
            chunkSize: prefillStep,
            splitIndices: prefillSplitIndices
        )
        var generatedTokenIDs: [Int] = []
        var sampleDiagnostics: [String] = []
        generatedTokenIDs.reserveCapacity(maxTokens)
        sampleDiagnostics.reserveCapacity(maxTokens)

        while generatedTokenIDs.count < maxTokens,
              !endTokenIDs.contains(nextTokenID) {
            generatedTokenIDs.append(nextTokenID)
            sampleDiagnostics.append((try session.lastSampleDiagnostics()) ?? "")
            nextTokenID = try session.nextToken()
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

    static func diagnosticValue(named name: String, in diagnostic: String) -> Double? {
        let prefix = "\(name)="
        guard let valueStart = diagnostic.range(of: prefix)?.upperBound else {
            return nil
        }
        let value = diagnostic[valueStart...].prefix { character in
            character == "-" || character == "+" || character == "."
                || character.isNumber
        }
        guard !value.isEmpty else { return nil }
        return Double(String(value))
    }

    private static func prefillCapturedPrefix(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        chunkSize: Int,
        synchronous: Bool
    ) throws {
        let step = max(1, chunkSize)
        var offset = 0
        while offset < tokenIDs.count {
            let end = min(offset + step, tokenIDs.count)
            let chunk = Array(tokenIDs[offset..<end])
            if synchronous {
                _ = try session.prefill(tokenIDs: chunk)
            } else {
                try session.prefillAsync(tokenIDs: chunk)
            }
            offset = end
        }
    }

    private static func prefillForGreedyDecode(
        session: QwenCmlxLazyDecodeSession,
        tokenIDs: [Int],
        chunkSize: Int,
        splitIndices: [Int] = []
    ) throws -> Int {
        for range in prefillChunkRanges(
            tokenCount: tokenIDs.count,
            chunkSize: chunkSize,
            splitIndices: splitIndices
        ) {
            try session.prefillAsync(tokenIDs: Array(tokenIDs[range]))
        }
        return try session.nextToken()
    }

    static func prefillChunkRanges(
        tokenCount: Int,
        chunkSize: Int,
        splitIndices: [Int]
    ) -> [Range<Int>] {
        guard tokenCount > 0 else { return [] }
        let step = max(1, chunkSize)
        let boundaries = Array(
            Set([0, tokenCount] + splitIndices.filter { $0 > 0 && $0 < tokenCount })
        ).sorted()
        return zip(boundaries, boundaries.dropFirst()).flatMap { pair in
            let (start, segmentEnd) = pair
            return stride(from: start, to: segmentEnd, by: step).map { offset in
                offset..<min(offset + step, segmentEnd)
            }
        }
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
