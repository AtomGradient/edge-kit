// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation

enum NativeQwenDSRPolicyPlanner {
    static func policies(
        for architecture: QwenHybridArchitecture,
        parameters: EdgeGenerateParameters
    ) throws -> [Int: QwenDSRKVCachePolicy] {
        guard parameters.useDSR,
              let maxSize = parameters.dsrMaxCritical,
              maxSize > 0
        else {
            return [:]
        }

        return try QwenDSRKVCachePolicy.layerAwarePolicies(
            for: architecture,
            maxSize: maxSize,
            heavyBudget: parameters.dsrHeavyBudget,
            recentBudget: parameters.dsrRecentBudget,
            sinkSize: 4,
            scene: qwenScene(parameters.dsrScene),
            evictionInterval: parameters.dsrEvictionInterval
        )
    }

    static func transientKVCapacity(
        parameters: EdgeGenerateParameters
    ) -> Int? {
        guard parameters.useDSR,
              let maxSize = parameters.dsrMaxCritical,
              maxSize > 0
        else {
            return nil
        }
        let transientTokens = max(parameters.prefillStepSize, parameters.dsrEvictionInterval)
        return maxSize + transientTokens + 1
    }

    static func qwenScene(_ scene: DSRSceneType) -> QwenDSRKVCacheScene {
        switch scene {
        case .chat:
            return .chat
        case .code:
            return .code
        case .image:
            return .image
        case .translate:
            return .translate
        case .summary:
            return .summary
        case .creative:
            return .creative
        }
    }
}
