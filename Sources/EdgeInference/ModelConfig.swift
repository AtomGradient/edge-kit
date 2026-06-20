// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Model catalog entry used by the inference runtime.
public struct ModelConfig: Sendable {

    /// Stable local model identifier.
    public let modelID: String
    /// Hugging Face repository path.
    public let huggingFaceRepo: String
    /// Display name for UI and logs.
    public let familyName: String
    /// Approximate on-disk size in GB.
    public let sizeGB: Double
    /// Minimum available RAM required in GB.
    public let minRAMGB: Int
    /// Recommended device tier.
    public let tier: ModelTier

    /// All bundled model configurations.
    public static let all: [ModelConfig] = gemmaModels + qwen3Models + qwen35Models

    static let gemmaModels: [ModelConfig] = [
        .init(modelID: "gemma-3-4b-3bit",
              huggingFaceRepo: "AtomGradient/gemma-3-4b-it-qat-3bit-edge",
              familyName: "Gemma 3 4B (3bit)", sizeGB: 2.7, minRAMGB: 4, tier: .standard),
        .init(modelID: "gemma-3-4b-4bit",
              huggingFaceRepo: "AtomGradient/gemma-3-4b-it-qat-4bit-edge",
              familyName: "Gemma 3 4B (4bit)", sizeGB: 2.8, minRAMGB: 4, tier: .standard),
        .init(modelID: "gemma-3-4b-6bit",
              huggingFaceRepo: "AtomGradient/gemma-3-4b-it-qat-6bit-edge",
              familyName: "Gemma 3 4B (6bit)", sizeGB: 4.3, minRAMGB: 6, tier: .pro),
        .init(modelID: "gemma-3-4b-8bit",
              huggingFaceRepo: "AtomGradient/gemma-3-4b-it-qat-8bit-edge",
              familyName: "Gemma 3 4B (8bit)", sizeGB: 5.3, minRAMGB: 6, tier: .pro),
    ]

    static let qwen3Models: [ModelConfig] = [
        .init(modelID: "qwen3-4b-4bit",
              huggingFaceRepo: "AtomGradient/qwen3-4b-4bit-edge",
              familyName: "Qwen3 4B (4bit)", sizeGB: 2.1, minRAMGB: 4, tier: .standard),
        .init(modelID: "qwen3-8b-3bit",
              huggingFaceRepo: "AtomGradient/qwen3-8b-3bit-edge",
              familyName: "Qwen3 8B (3bit)", sizeGB: 3.4, minRAMGB: 5, tier: .pro),
        .init(modelID: "qwen3-8b-4bit",
              huggingFaceRepo: "AtomGradient/qwen3-8b-4bit-edge",
              familyName: "Qwen3 8B (4bit)", sizeGB: 4.3, minRAMGB: 6, tier: .max),
        .init(modelID: "qwen3-8b-6bit",
              huggingFaceRepo: "AtomGradient/qwen3-8b-6bit-edge",
              familyName: "Qwen3 8B (6bit)", sizeGB: 6.2, minRAMGB: 8, tier: .ultra),
        .init(modelID: "qwen3-vl-4b-4bit",
              huggingFaceRepo: "AtomGradient/qwen3-vl-4b-instruct-4bit-edge",
              familyName: "Qwen3 VL 4B (4bit)", sizeGB: 2.9, minRAMGB: 5, tier: .pro),
    ]

    static let qwen35Models: [ModelConfig] = [
        .init(modelID: "qwen3.5-0.8b",
              huggingFaceRepo: "AtomGradient/qwen3.5-0.8b-edge",
              familyName: "Qwen3.5 0.8B", sizeGB: 1.6, minRAMGB: 3, tier: .standard),
        .init(modelID: "qwen3.5-2b-8bit",
              huggingFaceRepo: "AtomGradient/qwen3.5-2b-8bit-edge",
              familyName: "Qwen3.5 2B (8bit)", sizeGB: 2.5, minRAMGB: 4, tier: .standard),
        .init(modelID: "qwen3.5-2b-bf16",
              huggingFaceRepo: "AtomGradient/qwen3.5-2b-bf16-edge",
              familyName: "Qwen3.5 2B (bf16)", sizeGB: 4.1, minRAMGB: 6, tier: .pro),
        .init(modelID: "qwen3.5-9b",
              huggingFaceRepo: "AtomGradient/qwen3.5-9b-edge",
              familyName: "Qwen3.5 9B", sizeGB: 18.0, minRAMGB: 20, tier: .ultra),
        .init(modelID: "qwen3.5-9b-8bit",
              huggingFaceRepo: "AtomGradient/qwen3.5-9b-8bit-edge",
              familyName: "Qwen3.5 9B (8bit)", sizeGB: 9.7, minRAMGB: 12, tier: .ultra),
    ]

    /// Finds a model configuration by local model identifier.
    public static func find(modelID: String) -> ModelConfig? {
        all.first { $0.modelID == modelID }
    }

    /// Returns the default model configuration for a tier.
    public static func config(for tier: ModelTier) -> ModelConfig {
        switch tier {
        case .standard: return find(modelID: "qwen3.5-0.8b")   ?? gemmaModels[0]
        case .pro:      return find(modelID: "qwen3.5-2b-8bit") ?? gemmaModels[1]
        case .max:      return find(modelID: "qwen3-8b-4bit")   ?? qwen3Models[2]
        case .ultra:    return find(modelID: "qwen3.5-9b-8bit") ?? qwen3Models[3]
        }
    }
}

/// Device capability tier used for model selection.
public enum ModelTier: String, CaseIterable, Sendable {
    case standard
    case pro
    case max
    case ultra

    /// Display label for UI and logs.
    var displayName: String {
        switch self {
        case .standard: return "Standard"
        case .pro:      return "Pro"
        case .max:      return "Max"
        case .ultra:    return "Ultra"
        }
    }
}

/// Selects a model tier from a device profile.
public enum ModelTierSelector {
    /// Recommends a model tier from available RAM.
    public static func recommend(for profile: DeviceProfile) -> ModelTier {
        let ram = profile.availableRAMGB
        switch ram {
        case ..<4:   return .standard
        case 4..<7:  return .pro
        case 7..<12: return .max
        default:     return .ultra
        }
    }
}
