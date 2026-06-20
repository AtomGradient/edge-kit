// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/// EdgeRuntime-level scene hint retained for native KV policy planning.
///
/// The public fallback path no longer exposes EdgeStudio's private DSR cache controls,
/// so this type remains SDK metadata until the native runtime owns eviction.
public enum DSRSceneType: String, Sendable, Codable {
    case chat
    case code
    case image
    case translate
    case summary
    case creative
}
