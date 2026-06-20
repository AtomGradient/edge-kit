// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Thermal-aware inference throttling.
///
/// Monitors device thermal state and adjusts inference behavior to reduce heat:
/// - nominal/fair: full speed
/// - serious: reduce max tokens, add small delay between tokens
/// - critical: pause generation, wait for cooldown
///
/// Usage:
/// ```swift
/// let thermal = ThermalManager()
/// if thermal.shouldThrottle {
///     // reduce max tokens or delay
/// }
/// if thermal.shouldPause {
///     // stop generation, show warning to user
/// }
/// ```
public struct ThermalManager: Sendable {

    /// Current thermal level
    public var level: ThermalLevel {
        #if canImport(UIKit)
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return .nominal
        case .fair:     return .fair
        case .serious:  return .serious
        case .critical: return .critical
        @unknown default: return .nominal
        }
        #else
        return .nominal  // macOS doesn't throttle as aggressively
        #endif
    }

    public enum ThermalLevel: String, Sendable {
        case nominal    // Full speed
        case fair       // Normal, slightly warm
        case serious    // Throttle recommended
        case critical   // Pause generation
    }

    /// Whether inference should be throttled (reduce max tokens / add delay)
    public var shouldThrottle: Bool {
        level == .serious || level == .critical
    }

    /// Whether inference should be paused entirely
    public var shouldPause: Bool {
        level == .critical
    }

    /// Recommended max tokens multiplier based on thermal state
    public var maxTokensMultiplier: Double {
        switch level {
        case .nominal:  return 1.0
        case .fair:     return 1.0
        case .serious:  return 0.5   // Halve max tokens
        case .critical: return 0.0   // Stop
        }
    }

    /// Recommended delay between token generations (seconds)
    public var interTokenDelay: TimeInterval {
        switch level {
        case .nominal:  return 0
        case .fair:     return 0
        case .serious:  return 0.01  // 10ms delay to reduce heat
        case .critical: return 1.0   // 1s delay (effectively paused)
        }
    }

    public init() {}
}
