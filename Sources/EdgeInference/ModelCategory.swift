// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Model category — determines which engine to use for inference.
public enum ModelCategory: String, Codable, Sendable, CaseIterable {
    case llm    // Text generation
    case vlm    // Vision + Language
    case tts    // Text-to-Speech (Qwen3TTS)
    case stt    // Speech-to-Text

    /// Detect model category from config.json in the given directory.
    public static func detect(from directory: URL) -> ModelCategory {
        let configURL = directory.appending(component: "config.json")
        guard let data = try? Data(contentsOf: configURL),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return .llm }

        if config["vision_config"] != nil {
            return .vlm
        }

        if let modelType = config["model_type"] as? String {
            let lower = modelType.lowercased()

            if lower.contains("tts") {
                return .tts
            }

            if lower.contains("asr") || lower == "sensevoice" {
                return .stt
            }
        }

        if config["talker_config"] != nil {
            return .tts
        }

        return .llm
    }
}
