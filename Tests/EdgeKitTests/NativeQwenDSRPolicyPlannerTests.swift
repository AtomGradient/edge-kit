// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import XCTest
@testable import EdgeInference

final class NativeQwenDSRPolicyPlannerTests: XCTestCase {
    func testBuildsPoliciesOnlyForFullAttentionLayers() throws {
        let architecture = try QwenHybridArchitecture(
            family: .qwen35,
            vocabularySize: 128,
            hiddenSize: 8,
            intermediateSize: 16,
            attentionHeadCount: 2,
            keyValueHeadCount: 1,
            headDimension: 4,
            contextLength: 64,
            rmsNormEpsilon: 1e-6,
            ropeTheta: 10_000,
            layerKinds: [.fullAttention, .gdn, .fullAttention]
        )
        let parameters = EdgeGenerateParameters(
            maxTokens: 16,
            prefillStepSize: 128,
            useDSR: true,
            dsrMaxCritical: 2_048,
            dsrScene: .image,
            dsrEvictionInterval: 32
        )

        let policies = try NativeQwenDSRPolicyPlanner.policies(
            for: architecture,
            parameters: parameters
        )

        XCTAssertEqual(policies.keys.sorted(), [0, 2])
        XCTAssertEqual(policies[0]?.sinkSize, 4)
        XCTAssertEqual(policies[0]?.evictionInterval, 32)
        XCTAssertNil(policies[1])
    }

    func testTransientCapacityLeavesRoomForPrefillBurst() {
        let parameters = EdgeGenerateParameters(
            maxTokens: 16,
            prefillStepSize: 512,
            useDSR: true,
            dsrMaxCritical: 3_584,
            dsrEvictionInterval: 32
        )

        XCTAssertEqual(
            NativeQwenDSRPolicyPlanner.transientKVCapacity(parameters: parameters),
            4_097
        )
    }
}
