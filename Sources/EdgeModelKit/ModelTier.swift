// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

/// Device-aware model tier selection manager.
public final class ModelTierManager: ObservableObject {

    public static let shared = ModelTierManager()

    @Published public private(set) var currentTier: ModelTier = .standard
    @Published public private(set) var availableTiers: [ModelTier] = []

    private init() {
        refresh()
    }

    /// Recomputes the recommended and available tiers for the current device.
    public func refresh() {
        let profile = DeviceProfile.current
        currentTier = ModelTierSelector.recommend(for: profile)
        availableTiers = ModelTier.allCases.filter { tier in
            ModelConfig.config(for: tier).minRAMGB <= profile.totalRAMGB
        }
    }

    /// Selects a tier when it is available on the current device.
    public func select(tier: ModelTier) {
        guard availableTiers.contains(tier) else { return }
        currentTier = tier
    }

    /// Model configuration for the selected tier.
    public var currentConfig: ModelConfig {
        ModelConfig.config(for: currentTier)
    }
}
