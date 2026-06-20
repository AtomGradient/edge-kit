// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// KV cache memory policy planned for a loaded model and device.
public struct KVCacheMemoryPolicy: Sendable {

    public enum Mode: String, Sendable {
        /// Caller controls KV cache behavior explicitly.
        case manual
        /// Runtime plans KV cache behavior from the device and model profile.
        case auto
        /// Runtime uses stricter memory-saving defaults.
        case aggressive
    }

    /// Policy planning mode.
    public let mode: Mode

    /// Token threshold that enables KV cache quantization. `0` disables it.
    public let quantizeAtTokens: Int

    /// KV cache quantization bit width.
    public let kvBits: Int

    /// KV cache quantization group size.
    public let kvGroupSize: Int

    /// Maximum KV cache length in tokens. `nil` means unbounded.
    public let maxKVSize: Int?

    /// Recommended maximum context length.
    public let maxContextLength: Int

    /// Human-readable policy summary.
    public let reasoning: String

    /// Whether each eval step should wait synchronously.
    public let syncEval: Bool

    /// Whether native DSR eviction is enabled.
    public let useDSR: Bool

    /// Maximum DSR cache size. `nil` falls back to `maxKVSize`.
    public let dsrMaxCritical: Int?

    /// Heavy-hitter token budget. `nil` leaves it to native planning.
    public let dsrHeavyBudget: Int?

    /// Recent-token budget. `nil` leaves it to native planning.
    public let dsrRecentBudget: Int?

    /// Default scene hint used by native cache planning.
    public let dsrScene: DSRSceneType

    /// Product-level memory intent used when this policy was planned.
    public let memoryIntent: EdgeMemoryIntent

    public init(
        mode: Mode,
        quantizeAtTokens: Int,
        kvBits: Int,
        kvGroupSize: Int,
        maxKVSize: Int?,
        maxContextLength: Int,
        reasoning: String,
        syncEval: Bool,
        useDSR: Bool,
        dsrMaxCritical: Int?,
        dsrHeavyBudget: Int?,
        dsrRecentBudget: Int?,
        dsrScene: DSRSceneType,
        memoryIntent: EdgeMemoryIntent = .balanced
    ) {
        self.mode = mode
        self.quantizeAtTokens = quantizeAtTokens
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.maxKVSize = maxKVSize
        self.maxContextLength = maxContextLength
        self.reasoning = reasoning
        self.syncEval = syncEval
        self.useDSR = useDSR
        self.dsrMaxCritical = dsrMaxCritical
        self.dsrHeavyBudget = dsrHeavyBudget
        self.dsrRecentBudget = dsrRecentBudget
        self.dsrScene = dsrScene
        self.memoryIntent = memoryIntent
    }

    /// Manual policy with automatic KV cache management disabled.
    public static let manual = KVCacheMemoryPolicy(
        mode: .manual,
        quantizeAtTokens: 0,
        kvBits: 4,
        kvGroupSize: 64,
        maxKVSize: nil,
        maxContextLength: 8192,
        reasoning: "Manual mode — no automatic KV cache management",
        syncEval: false,
        useDSR: false,
        dsrMaxCritical: nil,
        dsrHeavyBudget: nil,
        dsrRecentBudget: nil,
        dsrScene: .chat
    )
}
