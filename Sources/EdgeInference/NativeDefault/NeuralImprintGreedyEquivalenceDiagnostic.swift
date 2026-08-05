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

public struct NeuralImprintGreedyEquivalenceResult: Codable, Sendable, Equatable {
    public let schemaVersion: String
    public let prefixTokenCount: Int
    public let suffixTokenCount: Int
    public let visibleInputTokenCount: Int
    public let visibleTokensEqualPrefixPlusSuffix: Bool
    public let visiblePrefillStep: Int
    public let capturePrefillStep: Int
    public let captureUsesSyncPrefill: Bool
    public let sampleDiagnosticsAvailable: Bool
    public let paths: [NeuralImprintGreedyPathReceipt]
    public let comparison: NeuralImprintGreedyPathComparison
    public let divergenceProbes: [NeuralImprintGreedyDivergenceProbe]
}

enum NeuralImprintGreedyEquivalenceSupport {
    static let schemaVersion = "edge-kit.neural_imprint_greedy_equivalence.v1"

    private struct GeneratedPath {
        let receipt: NeuralImprintGreedyPathReceipt
        let tokenIDs: [Int]
        let sampleDiagnostics: [String]
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
        captureUsesSyncPrefill: Bool
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
            prefillStep: visiblePrefillStep
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
            }
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
        prefillStep: Int
    ) throws -> GeneratedPath {
        var nextTokenID = try prefillForGreedyDecode(
            session: session,
            tokenIDs: inputTokenIDs,
            chunkSize: prefillStep
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
            sampleDiagnostics: sampleDiagnostics
        )
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
        chunkSize: Int
    ) throws -> Int {
        let step = max(1, chunkSize)
        var offset = 0
        while offset < tokenIDs.count {
            let end = min(offset + step, tokenIDs.count)
            try session.prefillAsync(tokenIDs: Array(tokenIDs[offset..<end]))
            offset = end
        }
        return try session.nextToken()
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
