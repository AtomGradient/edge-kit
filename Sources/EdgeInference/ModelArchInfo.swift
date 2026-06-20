// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Model architecture metadata parsed from `config.json` for KV cache budgeting.
///
/// KV cache size can differ significantly across architectures, so runtime policy
/// should use model-specific parameters instead of coarse heuristics.
public struct ModelArchInfo: Sendable {

    /// Number of transformer blocks.
    public let numLayers: Int

    /// Number of KV attention heads.
    public let numKVHeads: Int

    /// Per-head dimension.
    public let headDim: Int

    /// Model type identifier from the config.
    public let modelType: String

    /// Number of GDN (GatedDeltaNet) layers in hybrid architectures.
    /// GDN layers use fixed-size SSM state and do not grow KV cache per token.
    public let numGDNLayers: Int

    /// Number of KV-sharing layers that reuse other layers' KV state.
    public let numKVSharedLayers: Int

    /// Number of full-attention layers that store KV cache.
    public var numFALayers: Int { numLayers - numGDNLayers - numKVSharedLayers }

    /// KV cache bytes per token across all full-attention layers in FP16.
    /// = 2 (K+V) * numFALayers * numKVHeads * headDim * 2 (FP16 bytes)
    public var kvBytesPerTokenFP16: Int {
        2 * numFALayers * numKVHeads * headDim * 2
    }

    /// KV cache bytes per token across all full-attention layers with INT4 quantization.
    /// INT4 storage is roughly one quarter of FP16 plus scale overhead.
    public var kvBytesPerTokenINT4: Int {
        let elementsPerToken = 2 * numFALayers * numKVHeads * headDim
        let quantizedBytes = elementsPerToken / 2
        let scaleOverhead = elementsPerToken / 64
        return quantizedBytes + scaleOverhead * 2
    }

    /// Fixed SSM state size for GDN layers in MB.
    public var gdnFixedStateMB: Double {
        guard numGDNLayers > 0 else { return 0 }
        return Double(numGDNLayers) * 0.5
    }

    /// Estimates total KV cache memory in MB for a context length and quantization mode.
    public func estimateKVCacheMB(contextLength: Int, quantized: Bool) -> Double {
        let bytesPerToken = quantized ? kvBytesPerTokenINT4 : kvBytesPerTokenFP16
        let kvMB = Double(bytesPerToken * contextLength) / (1024 * 1024)
        return kvMB + gdnFixedStateMB
    }

    /// Computes the maximum context length for the available memory budget in MB.
    public func maxContextLength(availableMemoryMB: Int, quantized: Bool) -> Int {
        let bytesPerToken = quantized ? kvBytesPerTokenINT4 : kvBytesPerTokenFP16
        guard bytesPerToken > 0 else { return 4096 }
        let availableBytes = (availableMemoryMB - Int(gdnFixedStateMB)) * 1024 * 1024
        return max(256, availableBytes / bytesPerToken)
    }

    /// Loads architecture metadata from a model directory's `config.json`.
    public static func load(from directory: URL) -> ModelArchInfo? {
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return parse(json: json)
    }

    /// Parses a `config.json` dictionary.
    static func parse(json: [String: Any]) -> ModelArchInfo {
        let modelType = json["model_type"] as? String ?? "unknown"

        let textConfig = json["text_config"] as? [String: Any] ?? json

        let numLayers = textConfig["num_hidden_layers"] as? Int
            ?? textConfig["n_layers"] as? Int
            ?? textConfig["num_layers"] as? Int
            ?? json["num_hidden_layers"] as? Int
            ?? 32

        let numKVHeads = textConfig["num_key_value_heads"] as? Int
            ?? textConfig["num_attention_heads"] as? Int  // MHA fallback
            ?? json["num_key_value_heads"] as? Int
            ?? 8

        let headDim: Int
        if let hd = textConfig["head_dim"] as? Int {
            headDim = hd
        } else if let hiddenSize = textConfig["hidden_size"] as? Int,
                  let numHeads = textConfig["num_attention_heads"] as? Int,
                  numHeads > 0 {
            headDim = hiddenSize / numHeads
        } else {
            headDim = 128
        }

        let numGDNLayers = detectGDNLayers(json: textConfig, modelType: modelType, totalLayers: numLayers)

        let numKVSharedLayers = textConfig["num_kv_shared_layers"] as? Int ?? 0

        return ModelArchInfo(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim,
            modelType: modelType,
            numGDNLayers: numGDNLayers,
            numKVSharedLayers: numKVSharedLayers
        )
    }

    /// Detects the number of GatedDeltaNet layers.
    private static func detectGDNLayers(json: [String: Any], modelType: String, totalLayers: Int) -> Int {
        if let layerTypes = json["layer_types"] as? [String] {
            return layerTypes.filter {
                $0 == "gated_deltanet" || $0 == "linear" || $0 == "linear_attention"
            }.count
        }
        if modelType.contains("qwen3_5") || modelType.contains("qwen3.5") {
            if let attnLayers = json["attention_layers"] as? [Int] {
                return totalLayers - attnLayers.count
            }
        }
        return 0
    }

    /// Conservative fallback used when `config.json` cannot be parsed.
    public static func fallback(modelSizeGB: Double) -> ModelArchInfo {
        let estimatedLayers: Int
        if modelSizeGB < 2 { estimatedLayers = 24 }
        else if modelSizeGB < 5 { estimatedLayers = 36 }
        else if modelSizeGB < 8 { estimatedLayers = 40 }
        else { estimatedLayers = 48 }

        return ModelArchInfo(
            numLayers: estimatedLayers,
            numKVHeads: 8,
            headDim: 128,
            modelType: "unknown",
            numGDNLayers: 0,
            numKVSharedLayers: 0
        )
    }
}
