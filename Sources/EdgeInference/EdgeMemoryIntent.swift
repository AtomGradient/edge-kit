// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Product-level memory policy intent.
///
/// Phase 1 lets runtime DSR/KV planning bias the cache window. Phase 2
/// planning contracts can additionally use this intent to plan compaction,
/// fact-store/tool recall, mesh-adaptation, and quality-loop behavior.
public enum EdgeMemoryIntent: String, CaseIterable, Codable, Sendable {
    /// Preserve more resident context when memory budget allows it.
    case longSession
    /// Bias toward exact/auditable recall when explicit fact-recall signals exist.
    case exactRecall
    /// Keep the device/model default planner behavior.
    case balanced
    /// Reduce resident state pressure for thermal/battery-sensitive sessions.
    case batteryFriendly

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self = EdgeMemoryIntent.parse(try container.decode(String.self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    /// Fail-closed parser for public APIs, env vars, and JSON config.
    public static func parse(_ rawValue: String?) -> EdgeMemoryIntent {
        guard let rawValue else { return .balanced }
        let normalized = rawValue
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        switch normalized {
        case "longsession", "long_session", "long":
            return .longSession
        case "exactrecall", "exact_recall", "exact":
            return .exactRecall
        case "batteryfriendly", "battery_friendly", "battery", "low_power", "lowpower":
            return .batteryFriendly
        case "balanced", "default", "":
            return .balanced
        default:
            return .balanced
        }
    }
}
