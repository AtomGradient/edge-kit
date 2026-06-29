// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation

enum NativeEnvironment {
    static func bool(
        _ name: String,
        defaultValue: Bool = false,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        boolOverride([name], environment: environment) ?? defaultValue
    }

    static func bool(
        _ names: [String],
        defaultValue: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        boolOverride(names, environment: environment) ?? defaultValue
    }

    static func boolOverride(
        _ names: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool? {
        for name in names {
            guard let raw = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty
            else {
                continue
            }
            switch raw.lowercased() {
            case "1", "true", "yes", "on":
                return true
            case "0", "false", "no", "off":
                return false
            default:
                return nil
            }
        }
        return nil
    }

    static func int(
        _ name: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        int([name], environment: environment)
    }

    static func int(
        _ names: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int? {
        for name in names {
            guard let raw = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty,
                  let value = Int(raw)
            else {
                continue
            }
            return value
        }
        return nil
    }

    static func int(
        _ names: [String],
        defaultValue: Int,
        range: ClosedRange<Int>,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard let value = int(names, environment: environment) else {
            return defaultValue
        }
        return range.contains(value) ? value : defaultValue
    }
}

extension NativeCmlxAttentionCacheQuantization {
    static func summary(_ quantization: NativeCmlxAttentionCacheQuantization?) -> String {
        guard let quantization else { return "none" }
        return "int\(quantization.bits)@\(quantization.groupSize)"
    }
}

extension QwenFrogJumpPlan {
    var layerMask: UInt64 {
        layerMaskLayers.reduce(UInt64.zero) { mask, layer in
            mask | (UInt64(1) << UInt64(layer))
        }
    }

    var layerSummary: String {
        let layers = layerMaskLayers
        return layers.isEmpty ? "off" : layers.map(String.init).joined(separator: ",")
    }

    private var layerMaskLayers: [Int] {
        guard enabled,
              skipLayers.contains(12),
              skipLayers.contains(13)
        else {
            return []
        }
        return [12, 13]
    }
}

enum NativeCmlxSampling {
    static let eosSamplingLogitPenalty: Float = 20

    static func contextTokenIds(
        promptSessionTokenIds: [Int],
        generatedTokenIds: [Int] = [],
        contextSize: Int? = nil
    ) -> [Int] {
        var context = promptSessionTokenIds
        context.reserveCapacity(promptSessionTokenIds.count + generatedTokenIds.count)
        context.append(contentsOf: generatedTokenIds)
        guard let contextSize else { return context }
        guard contextSize > 0 else { return [] }
        return Array(context.suffix(contextSize))
    }

    static func qwenSamplingConfiguration(
        parameters: EdgeGenerateParameters,
        promptSessionTokenIds: [Int],
        generatedTokenIds: [Int] = [],
        endTokenIds: Set<Int> = []
    ) -> QwenSamplingConfiguration {
        QwenSamplingConfiguration(
            temperature: parameters.temperature,
            topK: parameters.topK,
            topP: parameters.topP > 0 && parameters.topP < 1 ? parameters.topP : nil,
            minP: parameters.minP,
            repetitionPenalty: parameters.repetitionPenalty,
            repetitionTokenIds: contextTokenIds(
                promptSessionTokenIds: promptSessionTokenIds,
                generatedTokenIds: generatedTokenIds,
                contextSize: parameters.repetitionContextSize
            ),
            presencePenalty: parameters.presencePenalty,
            presenceTokenIds: contextTokenIds(
                promptSessionTokenIds: promptSessionTokenIds,
                generatedTokenIds: generatedTokenIds,
                contextSize: parameters.presenceContextSize
            ),
            frequencyPenalty: parameters.frequencyPenalty,
            frequencyTokenIds: contextTokenIds(
                promptSessionTokenIds: promptSessionTokenIds,
                generatedTokenIds: generatedTokenIds,
                contextSize: parameters.frequencyContextSize
            ),
            endTokenIds: Array(endTokenIds),
            generatedTokenCount: generatedTokenIds.count,
            minimumGeneratedTokens: parameters.minimumGeneratedTokens,
            eosPenaltyUntilToken: parameters.eosPenaltyUntilToken
        )
    }

    struct PenaltyApplier {
        let parameters: EdgeGenerateParameters
        let endTokenIds: Set<Int>
        let setSamplingPenalties: (
            _ repetitionPenalty: Float,
            _ repetitionTokenIds: [Int],
            _ presencePenalty: Float,
            _ presenceTokenIds: [Int],
            _ frequencyPenalty: Float,
            _ frequencyTokenIds: [Int]
        ) throws -> Void
        let setEOSSamplingBias: (
            _ tokenIds: [Int],
            _ suppress: Bool,
            _ logitPenalty: Float
        ) throws -> Void
        let clearEOSSamplingBias: () throws -> Void

        var samplingPenaltiesAreActive: Bool {
            parameters.repetitionPenalty != 1.0 ||
                parameters.presencePenalty != 0.0 ||
                parameters.frequencyPenalty != 0.0
        }

        var eosSamplingBiasRequested: Bool {
            parameters.minimumGeneratedTokens > 0 ||
                parameters.eosPenaltyUntilToken > 0
        }

        func applySamplingPenalties(
            promptSessionTokenIds: [Int],
            generatedTokenIds: [Int] = []
        ) throws {
            try setSamplingPenalties(
                parameters.repetitionPenalty,
                contextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.repetitionContextSize
                ),
                parameters.presencePenalty,
                contextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.presenceContextSize
                ),
                parameters.frequencyPenalty,
                contextTokenIds(
                    promptSessionTokenIds: promptSessionTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    contextSize: parameters.frequencyContextSize
                )
            )
        }

        func applyEOSSamplingBias(generatedTokenCount: Int) throws {
            let suppress = generatedTokenCount < parameters.minimumGeneratedTokens
            let logitPenalty = generatedTokenCount < parameters.eosPenaltyUntilToken
                ? NativeCmlxSampling.eosSamplingLogitPenalty
                : 0
            if suppress || logitPenalty > 0 {
                try setEOSSamplingBias(Array(endTokenIds), suppress, logitPenalty)
            } else {
                try clearEOSSamplingBias()
            }
        }
    }
}
