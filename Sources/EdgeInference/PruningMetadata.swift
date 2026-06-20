// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Reads Edge Studio optimization metadata from a model's config.json.
///
/// Edge Studio (VirsualLLMInfer) writes optimization traces into config.json during
/// the 7-step pipeline. EdgeRuntime MUST read these to correctly load and run the model.
/// Generic upstream loaders will produce incorrect results with Edge Studio models.
public struct PruningMetadata: Sendable {

    /// Vocabulary was reduced from original_size → compact_size.
    /// token_id_remap maps compact IDs back to original model embedding IDs.
    public struct VocabPruning: Sendable {
        public let originalVocabSize: Int
        public let compactVocabSize: Int
        /// compact_token_id → original_embedding_row
        public let idRemap: [Int: Int]
    }

    /// Per-layer FFN intermediate sizes after neuron pruning.
    /// Replaces the global `intermediate_size` from config.json.
    /// Layer i has FFN size perLayerSizes[i].
    public struct NeuronPruning: Sendable {
        public let defaultIntermediateSize: Int
        public let perLayerSizes: [Int]
        /// Layer indices where size differs from default
        public var prunedLayers: [Int] {
            perLayerSizes.enumerated().compactMap {
                $0.element < defaultIntermediateSize ? $0.offset : nil
            }
        }
    }

    /// Some transformer layers were physically removed.
    /// The model has fewer layers than the original architecture.
    public struct LayerPruning: Sendable {
        public let originalLayerCount: Int
        public let currentLayerCount: Int
        /// Original layer indices that were removed (for provenance tracking)
        public let removedLayers: [Int]
    }

    /// Language and Vision weights split into separate files.
    /// Used for VLMs (Gemma-3-4B, Qwen3-VL-4B) to allow lazy-loading.
    public struct WeightSplit: Sendable {
        public let languageModelSizeMB: Double
        public let visionModelSizeMB: Double
    }

    public struct ResolutionReduction: Sendable {
        public let originalImageSize: Int
        public let targetImageSize: Int
    }

    public let modelDir: URL
    public let vocabPruning: VocabPruning?
    public let neuronPruning: NeuronPruning?
    public let layerPruning: LayerPruning?
    public let weightSplit: WeightSplit?
    public let resolutionReduction: ResolutionReduction?

    /// True if this model has ANY Edge Studio optimization applied.
    public var isEdgeOptimized: Bool {
        vocabPruning != nil || neuronPruning != nil ||
        layerPruning != nil || weightSplit != nil || resolutionReduction != nil
    }

    /// Load and parse Edge Studio optimization metadata from a model directory.
    /// Returns metadata even for standard models (all fields will be nil).
    public static func load(from modelDir: URL) throws -> PruningMetadata {
        let configURL = modelDir.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EdgeRuntimeError.loadFailed("config.json is not a valid JSON object")
        }
        return PruningMetadata(modelDir: modelDir, config: config)
    }

    init(modelDir: URL, config: [String: Any]) {
        self.modelDir = modelDir

        if let vp = config["vocab_pruning"] as? [String: Any] {
            let orig = vp["original_text_vocab_size"] as? Int
                    ?? vp["original_vocab_size"] as? Int ?? 0
            let compact = vp["compact_vocab_size"] as? Int ?? 0
            var remap: [Int: Int] = [:]
            if let remapRaw = vp["id_remap"] as? [String: Int] {
                for (k, v) in remapRaw {
                    if let ki = Int(k) { remap[ki] = v }
                }
            }
            vocabPruning = VocabPruning(
                originalVocabSize: orig,
                compactVocabSize: compact,
                idRemap: remap
            )
        } else {
            vocabPruning = nil
        }

        var np: NeuronPruning? = nil
        let subConfigs: [[String: Any]] = [
            config["text_config"] as? [String: Any],
            config["talker_config"] as? [String: Any],
            config,
        ].compactMap { $0 }
        for sub in subConfigs {
            if let plis = sub["per_layer_intermediate_sizes"] as? [Int], !plis.isEmpty {
                let defaultSize = sub["intermediate_size"] as? Int ?? 0
                np = NeuronPruning(defaultIntermediateSize: defaultSize, perLayerSizes: plis)
                break
            }
        }
        neuronPruning = np

        if let tlp = config["text_layer_pruning"] as? [String: Any] {
            let removed = tlp["removed_layers"] as? [Int] ?? []
            let old = tlp["old_num_layers"] as? Int ?? 0
            let new = tlp["new_num_layers"] as? Int ?? 0
            layerPruning = LayerPruning(
                originalLayerCount: old,
                currentLayerCount: new,
                removedLayers: removed
            )
        } else {
            layerPruning = nil
        }

        if let ws = config["weight_split"] as? [String: Any],
           ws["enabled"] as? Bool == true {
            weightSplit = WeightSplit(
                languageModelSizeMB: ws["language_model_size_mb"] as? Double ?? 0,
                visionModelSizeMB: ws["vision_model_size_mb"] as? Double ?? 0
            )
        } else {
            weightSplit = nil
        }

        if let rr = config["resolution_reduction"] as? [String: Any] {
            resolutionReduction = ResolutionReduction(
                originalImageSize: rr["original_image_size"] as? Int ?? 0,
                targetImageSize: rr["target_image_size"] as? Int ?? 0
            )
        } else {
            resolutionReduction = nil
        }
    }
}

extension PruningMetadata: CustomStringConvertible {
    public var description: String {
        guard isEdgeOptimized else { return "Standard model (no Edge Studio optimizations)" }
        var parts: [String] = ["[Edge Studio Model]"]
        if let vp = vocabPruning {
            parts.append("Vocab: \(vp.originalVocabSize)→\(vp.compactVocabSize)")
        }
        if let np = neuronPruning {
            parts.append("Neurons: \(np.prunedLayers.count) layers pruned")
        }
        if let lp = layerPruning {
            parts.append("Layers: \(lp.originalLayerCount)→\(lp.currentLayerCount)")
        }
        if weightSplit != nil {
            parts.append("WeightSplit: LM+Vision separated")
        }
        if let rr = resolutionReduction {
            parts.append("Resolution: \(rr.originalImageSize)→\(rr.targetImageSize)px")
        }
        return parts.joined(separator: " | ")
    }
}
