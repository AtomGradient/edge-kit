// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class InferencePolicyTests: XCTestCase {

    func testEdgeMemoryIntentParsingIsFailClosed() {
        XCTAssertEqual(EdgeMemoryIntent.parse("longSession"), .longSession)
        XCTAssertEqual(EdgeMemoryIntent.parse("exact_recall"), .exactRecall)
        XCTAssertEqual(EdgeMemoryIntent.parse("battery-friendly"), .batteryFriendly)
        XCTAssertEqual(EdgeMemoryIntent.parse("unknown-future-intent"), .balanced)
        XCTAssertEqual(EdgeMemoryIntent.parse(nil), .balanced)
    }

    func testResolveDoesNotRelaxStaticMemoryPolicy() {
        let staticPolicy = KVCacheMemoryPolicy(
            mode: .aggressive,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            maxKVSize: nil,
            maxContextLength: 3584,
            reasoning: "test static aggressive policy",
            syncEval: false,
            useDSR: true,
            dsrMaxCritical: 3584,
            dsrHeavyBudget: nil,
            dsrRecentBudget: nil,
            dsrScene: .chat
        )
        let arch = ModelArchInfo(
            numLayers: 32,
            numKVHeads: 4,
            headDim: 256,
            modelType: "qwen3_5",
            numGDNLayers: 24,
            numKVSharedLayers: 0
        )
        let snapshot = InferencePolicy.DeviceSnapshot(
            availableMemoryMB: 2708,
            jetsamLimitMB: 6144,
            totalRAMGB: 8,
            thermalLevel: .nominal,
            measuredBandwidthGBs: nil
        )
        let context = InferencePolicy.TurnContext(
            turn: 2,
            cachedTokenCount: 91,
            archInfo: arch,
            scene: .chat,
            requestedMaxTokens: 1024
        )

        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: context,
            staticPolicy: staticPolicy
        )

        XCTAssertTrue(resolved.useDSR)
        XCTAssertEqual(resolved.kvBits, 4)
        XCTAssertEqual(resolved.quantizedKVStart, 0)
        XCTAssertFalse(resolved.syncEval)
        XCTAssertTrue(resolved.reasoning.contains("static"))
    }

    func testResolveDoesNotInventDSRWhenStaticPolicyDisablesIt() {
        let staticPolicy = KVCacheMemoryPolicy(
            mode: .auto,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            maxKVSize: nil,
            maxContextLength: 3584,
            reasoning: "legacy fallback",
            syncEval: true,
            useDSR: false,
            dsrMaxCritical: nil,
            dsrHeavyBudget: nil,
            dsrRecentBudget: nil,
            dsrScene: .chat
        )
        let arch = ModelArchInfo(
            numLayers: 32,
            numKVHeads: 4,
            headDim: 256,
            modelType: "qwen3_5",
            numGDNLayers: 24,
            numKVSharedLayers: 0
        )
        let snapshot = InferencePolicy.DeviceSnapshot(
            availableMemoryMB: 4096,
            jetsamLimitMB: 6144,
            totalRAMGB: 12,
            thermalLevel: .nominal,
            measuredBandwidthGBs: nil
        )
        let context = InferencePolicy.TurnContext(
            turn: 8,
            cachedTokenCount: 4096,
            archInfo: arch,
            scene: .chat,
            requestedMaxTokens: 1024
        )

        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: context,
            staticPolicy: staticPolicy
        )

        XCTAssertFalse(resolved.useDSR)
        XCTAssertNil(resolved.dsrMaxCritical)
        XCTAssertEqual(resolved.maxKVSize, 3584)
        XCTAssertEqual(resolved.kvBits, 4)
        XCTAssertEqual(resolved.quantizedKVStart, 0)
        XCTAssertTrue(resolved.reasoning.contains("DSR unavailable"))

        var parameters = EdgeGenerateParameters(maxTokens: 32, useDSR: false)
        resolved.apply(to: &parameters)
        XCTAssertFalse(parameters.useDSR)
        XCTAssertEqual(parameters.maxKVSize, 3584)
    }

    func testResolveHonorsExplicitDSRWindowOverrideWithStaticPolicy() {
        let staticPolicy = KVCacheMemoryPolicy(
            mode: .aggressive,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            maxKVSize: nil,
            maxContextLength: 3584,
            reasoning: "test static aggressive policy",
            syncEval: false,
            useDSR: true,
            dsrMaxCritical: 3584,
            dsrHeavyBudget: nil,
            dsrRecentBudget: nil,
            dsrScene: .chat
        )
        let arch = ModelArchInfo(
            numLayers: 32,
            numKVHeads: 4,
            headDim: 256,
            modelType: "qwen3_5",
            numGDNLayers: 24,
            numKVSharedLayers: 0
        )
        let snapshot = InferencePolicy.DeviceSnapshot(
            availableMemoryMB: 1024,
            jetsamLimitMB: 6144,
            totalRAMGB: 8,
            thermalLevel: .nominal,
            measuredBandwidthGBs: nil
        )
        let context = InferencePolicy.TurnContext(
            turn: 12,
            cachedTokenCount: 8192,
            archInfo: arch,
            scene: .chat,
            requestedMaxTokens: 1024
        )

        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: context,
            staticPolicy: staticPolicy,
            dsrMaxCriticalOverride: 6144
        )

        XCTAssertTrue(resolved.useDSR)
        XCTAssertEqual(resolved.dsrMaxCritical, 6144)
        XCTAssertEqual(resolved.kvBits, 4)
        XCTAssertEqual(resolved.quantizedKVStart, 0)
        XCTAssertTrue(resolved.reasoning.contains("override 6144"))

        var parameters = EdgeGenerateParameters(maxTokens: 32, useDSR: true, dsrMaxCritical: 6144)
        resolved.apply(to: &parameters)
        XCTAssertTrue(parameters.useDSR)
        XCTAssertEqual(parameters.dsrMaxCritical, 6144)
    }

    func testResolveUsesContextMemoryIntentWhenStaticPolicyIsAbsent() {
        let arch = ModelArchInfo(
            numLayers: 1,
            numKVHeads: 1,
            headDim: 64,
            modelType: "test",
            numGDNLayers: 0,
            numKVSharedLayers: 0
        )
        let snapshot = InferencePolicy.DeviceSnapshot(
            availableMemoryMB: 4096,
            jetsamLimitMB: 8192,
            totalRAMGB: 8,
            thermalLevel: .nominal,
            measuredBandwidthGBs: nil
        )
        let context = InferencePolicy.TurnContext(
            turn: 3,
            cachedTokenCount: 0,
            archInfo: arch,
            scene: .chat,
            requestedMaxTokens: 1024,
            memoryIntent: .longSession
        )

        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: context,
            staticPolicy: nil
        )

        XCTAssertTrue(resolved.useDSR)
        XCTAssertGreaterThan(resolved.dsrMaxCritical ?? 0, 8192)
        XCTAssertTrue(resolved.reasoning.contains("intent=longSession"))
    }

    func testExplicitDSRWindowOverrideCanForceStaticPolicyWindowFromDSRModeOffParams() {
        let staticPolicy = KVCacheMemoryPolicy(
            mode: .auto,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            maxKVSize: nil,
            maxContextLength: 18_048,
            reasoning: "iPad static policy",
            syncEval: false,
            useDSR: true,
            dsrMaxCritical: 18_048,
            dsrHeavyBudget: nil,
            dsrRecentBudget: nil,
            dsrScene: .chat
        )
        let arch = ModelArchInfo(
            numLayers: 32,
            numKVHeads: 4,
            headDim: 256,
            modelType: "qwen3_5",
            numGDNLayers: 24,
            numKVSharedLayers: 0
        )
        let snapshot = InferencePolicy.DeviceSnapshot(
            availableMemoryMB: 3000,
            jetsamLimitMB: 8192,
            totalRAMGB: 8,
            thermalLevel: .nominal,
            measuredBandwidthGBs: nil
        )
        let context = InferencePolicy.TurnContext(
            turn: 20,
            cachedTokenCount: 16_000,
            archInfo: arch,
            scene: .chat,
            requestedMaxTokens: 1024
        )
        let requested = EdgeGenerateParameters(maxTokens: 1024, useDSR: false, dsrMaxCritical: 8192)

        let resolved = InferencePolicy.resolve(
            snapshot: snapshot,
            context: context,
            staticPolicy: staticPolicy,
            dsrMaxCriticalOverride: requested.dsrMaxCritical
        )

        XCTAssertTrue(resolved.useDSR)
        XCTAssertEqual(resolved.dsrMaxCritical, 8192)
        XCTAssertTrue(resolved.reasoning.contains("override 8192"))

        var parameters = requested
        resolved.apply(to: &parameters)
        XCTAssertTrue(parameters.useDSR)
        XCTAssertEqual(parameters.dsrMaxCritical, 8192)
    }

    func testMemoryBudgetPlannerAdvertisesNativeDSR() {
        let plan = MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 15,
            maxMBPerBuffer: 15,
            memoryLimitBytes: 4 * 1024 * 1024 * 1024,
            cacheLimitBytes: 256 * 1024 * 1024,
            forwardPassReserveMB: 300,
            kvBudgetMB: 1024,
            dsrMaxCritical: 3584,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 512,
            syncEval: false,
            vendoredCommandBufferPrefillQMMEnabled: false,
            fusedGDNDecodeEnabled: false,
            mode: .auto,
            reasoning: "test plan",
            measuredBandwidthGBs: 77.0,
            memoryIntent: .longSession
        )

        let policy = MemoryBudgetPlanner.toKVPolicy(plan)

        XCTAssertTrue(policy.useDSR)
        XCTAssertEqual(policy.memoryIntent, .longSession)
        XCTAssertEqual(policy.dsrMaxCritical, 3584)
        XCTAssertNil(policy.maxKVSize)
        XCTAssertEqual(policy.maxContextLength, 3584)
        XCTAssertEqual(policy.quantizeAtTokens, 0)
        XCTAssertEqual(policy.kvBits, 4)
        XCTAssertEqual(policy.syncEval, false)
        XCTAssertTrue(policy.reasoning.contains("native-dsr=enabled"))

        var parameters = EdgeGenerateParameters(maxTokens: 32, useDSR: false)
        parameters.applyPolicy(policy)
        XCTAssertTrue(parameters.useDSR)
        XCTAssertEqual(parameters.dsrMaxCritical, 3584)
        XCTAssertNil(parameters.maxKVSize)
        XCTAssertFalse(parameters.syncEval)
    }

    func testMemoryIntentAdjustsDSRWindowWithinBudget() {
        let balanced = MemoryBudgetPlanner.dsrMaxCriticalForPlan(
            kvBudgetMB: 2048,
            modelSizeGB: 5.6,
            totalBudgetMB: 8192,
            isMac: false,
            isPhone: false
        )
        let longSession = MemoryBudgetPlanner.dsrMaxCriticalForPlan(
            kvBudgetMB: 2048,
            modelSizeGB: 5.6,
            totalBudgetMB: 8192,
            isMac: false,
            isPhone: false,
            intent: .longSession
        )
        let exactRecall = MemoryBudgetPlanner.dsrMaxCriticalForPlan(
            kvBudgetMB: 2048,
            modelSizeGB: 5.6,
            totalBudgetMB: 8192,
            isMac: false,
            isPhone: false,
            intent: .exactRecall
        )
        let batteryFriendly = MemoryBudgetPlanner.dsrMaxCriticalForPlan(
            kvBudgetMB: 2048,
            modelSizeGB: 5.6,
            totalBudgetMB: 8192,
            isMac: false,
            isPhone: false,
            intent: .batteryFriendly
        )

        XCTAssertGreaterThan(longSession, balanced)
        XCTAssertEqual(exactRecall, longSession)
        XCTAssertLessThan(batteryFriendly, balanced)
        XCTAssertGreaterThanOrEqual(batteryFriendly, 2048)
    }
}
