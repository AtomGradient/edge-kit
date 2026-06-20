// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import XCTest
@testable import EdgeInference

final class NativeRuntimeBridgeTests: XCTestCase {
    func testExposesNativeRuntimeVersionAndSchedulingDefaults() {
        XCTAssertEqual(NativeRuntimeBridge.runtimeVersion, EdgeEngine.version)
        XCTAssertFalse(NativeRuntimeBridge.runtimeVersion.isEmpty)

        let configuration = NativeRuntimeBridge.defaultMetalConfiguration(contextLengthHint: 12_288)

        XCTAssertEqual(configuration.effectiveMaxOpsPerCommandBuffer, 5)
        XCTAssertEqual(configuration.maxMBPerCommandBuffer, 40)
    }

    func testAppliesNativeMetalConfiguration() {
        let configuration = NativeRuntimeBridge.applyMetalConfiguration(
            NativeRuntimeBridge.metalConfiguration(
                maxOpsPerCommandBuffer: 15,
                maxMBPerCommandBuffer: 7,
                contextLengthHint: 12_288,
                dynamicOpsSchedule: DynamicOpsSchedule(
                    floor: 5,
                    contextLow: 4_096,
                    contextHigh: 12_288
                ),
                quantizedBufferCacheLimitBytes: 512 * 1_048_576,
                releaseQuantizedHostStorageAfterUpload: true
            )
        )

        XCTAssertEqual(configuration.effectiveMaxOpsPerCommandBuffer, 5)
        let current = NativeRuntimeBridge.currentMetalConfiguration
        XCTAssertEqual(current.maxOpsPerCommandBuffer, configuration.maxOpsPerCommandBuffer)
        XCTAssertEqual(current.maxMBPerCommandBuffer, configuration.maxMBPerCommandBuffer)
        XCTAssertEqual(current.contextLengthHint, configuration.contextLengthHint)
        XCTAssertEqual(current.effectiveMaxOpsPerCommandBuffer, configuration.effectiveMaxOpsPerCommandBuffer)
        XCTAssertEqual(current.dynamicOpsSchedule?.floor, configuration.dynamicOpsSchedule?.floor)
        XCTAssertEqual(current.dynamicOpsSchedule?.contextLow, configuration.dynamicOpsSchedule?.contextLow)
        XCTAssertEqual(current.dynamicOpsSchedule?.contextHigh, configuration.dynamicOpsSchedule?.contextHigh)
        XCTAssertEqual(current.quantizedBufferCacheLimitBytes, configuration.quantizedBufferCacheLimitBytes)
        XCTAssertEqual(current.releaseQuantizedHostStorageAfterUpload, true)
    }

    func testPlanMetalConfigurationCarriesQuantizedCachePolicy() {
        let plan = MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 15,
            maxMBPerBuffer: 15,
            memoryLimitBytes: 6 * 1_073_741_824,
            cacheLimitBytes: 256 * 1_048_576,
            forwardPassReserveMB: 300,
            kvBudgetMB: 1200,
            dsrMaxCritical: 3072,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 256,
            syncEval: true,
            vendoredCommandBufferPrefillQMMEnabled: true,
            fusedGDNDecodeEnabled: true,
            mode: .auto,
            reasoning: "fixture",
            measuredBandwidthGBs: 100
        )

        let configuration = NativeRuntimeBridge.metalConfiguration(for: plan)

        XCTAssertEqual(configuration.commandBufferBatchingEnabled, true)
        XCTAssertEqual(configuration.memoryLimitBytes, 6 * 1_073_741_824)
        XCTAssertEqual(configuration.quantizedBufferCacheLimitBytes, 256 * 1_048_576)
        XCTAssertEqual(configuration.releaseQuantizedHostStorageAfterUpload, true)
        XCTAssertEqual(configuration.useVendoredCommandBufferPrefillQMM, true)
        XCTAssertEqual(configuration.useFusedGDNDecode, true)
        XCTAssertEqual(configuration.maxInFlightCommandBuffers, 1)
    }

    @MainActor
    func testCmlxAttentionKVQuantizationDoesNotDependOnDSRPolicies() {
        let quantization = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: EdgeGenerateParameters(
                maxTokens: 32,
                kvBits: 4,
                kvGroupSize: 64,
                useDSR: false
            ),
            dsrPolicies: [:]
        )

        XCTAssertEqual(quantization?.bits, 4)
        XCTAssertEqual(quantization?.groupSize, 64)

        let disabled = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: EdgeGenerateParameters(
                maxTokens: 32,
                kvBits: 0,
                kvGroupSize: 64,
                useDSR: false
            ),
            dsrPolicies: [:]
        )
        XCTAssertNil(disabled)

        let delayedUnsupported = LLMEngine.cmlxAttentionCacheQuantization(
            parameters: EdgeGenerateParameters(
                maxTokens: 32,
                quantizedKVStart: 128,
                kvBits: 4,
                kvGroupSize: 64,
                useDSR: false
            ),
            dsrPolicies: [:]
        )
        XCTAssertNil(delayedUnsupported)
    }

    func testNativeRuntimeLoadOptionsRecordVendoredQuantizedMatmulOverride() {
        let plan = MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 15,
            maxMBPerBuffer: 15,
            memoryLimitBytes: 6 * 1_073_741_824,
            cacheLimitBytes: 256 * 1_048_576,
            forwardPassReserveMB: 300,
            kvBudgetMB: 1200,
            dsrMaxCritical: 3072,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 256,
            syncEval: false,
            vendoredCommandBufferPrefillQMMEnabled: false,
            fusedGDNDecodeEnabled: false,
            mode: .auto,
            reasoning: "fixture",
            measuredBandwidthGBs: 100
        )

        let updated = plan.applying(
            NativeRuntimeLoadOptions(
                vendoredQuantizedMatmulEnabled: true,
                vendoredQuantizedPrefillMatmulEnabled: true,
                vendoredCommandBufferPrefillQMMEnabled: true,
                singleCommandBufferPrefillEnabled: true,
                singleCommandBufferDecodeEnabled: true,
                prefillLayerCommandBufferBatchingEnabled: true,
                fusedGDNDecodeEnabled: true,
                cmlxFastRMSNormEnabled: true,
                cmlxLazyDecodeEnabled: true
            )
        )

        XCTAssertTrue(updated.reasoning.contains("override-vendoredQMM=on"))
        XCTAssertTrue(updated.reasoning.contains("override-vendoredPrefillQMM=on"))
        XCTAssertTrue(updated.reasoning.contains("override-vendoredCBPrefillQMM=on"))
        XCTAssertTrue(updated.reasoning.contains("override-singleCBPrefill=on"))
        XCTAssertTrue(updated.reasoning.contains("override-singleCBDecode=on"))
        XCTAssertTrue(updated.reasoning.contains("override-prefillLayerCB=on"))
        XCTAssertTrue(updated.reasoning.contains("override-fusedGDNDecode=on"))
        XCTAssertTrue(updated.reasoning.contains("override-cmlxFastRMSNorm=on"))
        XCTAssertTrue(updated.reasoning.contains("override-cmlxLazyDecode=on"))
        XCTAssertEqual(updated.vendoredCommandBufferPrefillQMMEnabled, true)
        XCTAssertEqual(updated.fusedGDNDecodeEnabled, true)
    }

    func testNativeRuntimeLoadOptionsOverrideCommandBufferBudgets() {
        let plan = MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 15,
            maxMBPerBuffer: 15,
            memoryLimitBytes: 6 * 1_073_741_824,
            cacheLimitBytes: 256 * 1_048_576,
            forwardPassReserveMB: 300,
            kvBudgetMB: 1200,
            dsrMaxCritical: 3072,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 256,
            syncEval: false,
            vendoredCommandBufferPrefillQMMEnabled: false,
            fusedGDNDecodeEnabled: false,
            mode: .auto,
            reasoning: "fixture",
            measuredBandwidthGBs: 100
        )

        let updated = plan.applying(
            NativeRuntimeLoadOptions(
                maxOpsPerBuffer: 24,
                maxMBPerBuffer: 32,
                maxInFlightCommandBuffers: 64
            )
        )

        XCTAssertEqual(updated.maxOpsPerBuffer, 24)
        XCTAssertEqual(updated.maxMBPerBuffer, 32)
        XCTAssertTrue(updated.reasoning.contains("override-maxOps=24"))
        XCTAssertTrue(updated.reasoning.contains("override-maxMB=32"))
        XCTAssertTrue(updated.reasoning.contains("override-maxInflight=64"))
    }

    func testNativeRuntimeLoadOptionsOverrideSyncEval() {
        let plan = MemoryBudgetPlanner.Plan(
            maxOpsPerBuffer: 20,
            maxMBPerBuffer: 10,
            memoryLimitBytes: 6 * 1_073_741_824,
            cacheLimitBytes: 64 * 1_048_576,
            forwardPassReserveMB: 50,
            kvBudgetMB: 600,
            dsrMaxCritical: 13_535,
            quantizeAtTokens: 0,
            kvBits: 4,
            kvGroupSize: 64,
            dynamicOpsEnabled: true,
            dynamicOpsFloor: 5,
            dynamicOpsCtxLow: 4096,
            dynamicOpsCtxHigh: 12288,
            prefillStepSize: 256,
            syncEval: true,
            vendoredCommandBufferPrefillQMMEnabled: true,
            fusedGDNDecodeEnabled: true,
            mode: .aggressive,
            reasoning: "fixture syncEval=on",
            measuredBandwidthGBs: 0
        )

        let syncOff = plan.applying(NativeRuntimeLoadOptions(syncEval: false))
        XCTAssertFalse(syncOff.syncEval)
        XCTAssertTrue(syncOff.reasoning.contains("override-syncEval=off"))

        let syncOn = syncOff.applying(NativeRuntimeLoadOptions(syncEval: true))
        XCTAssertTrue(syncOn.syncEval)
        XCTAssertTrue(syncOn.reasoning.contains("override-syncEval=on"))
    }

    func testQuantizedCachePlannerKeepsIPhoneConservativeAndNonPhoneHotWeights() {
        let qwen4BOnIPhoneAir = MemoryBudgetPlanner.quantizedBufferCacheLimitMB(
            totalBudgetMB: 6144,
            headroomMB: 2100,
            modelSizeGB: 3.8,
            isPhone: true,
            isMac: false,
            chipStableAtLongCtx: true,
            mode: .auto
        )
        XCTAssertEqual(qwen4BOnIPhoneAir, 64)

        let qwen9BOnIPhoneAir = MemoryBudgetPlanner.quantizedBufferCacheLimitMB(
            totalBudgetMB: 6144,
            headroomMB: 700,
            modelSizeGB: 5.6,
            isPhone: true,
            isMac: false,
            chipStableAtLongCtx: true,
            mode: .aggressive
        )
        XCTAssertEqual(qwen9BOnIPhoneAir, 64)

        let qwen9BOnLegacyTightPhone = MemoryBudgetPlanner.quantizedBufferCacheLimitMB(
            totalBudgetMB: 6144,
            headroomMB: 700,
            modelSizeGB: 5.6,
            isPhone: true,
            isMac: false,
            chipStableAtLongCtx: false,
            mode: .aggressive
        )
        XCTAssertEqual(qwen9BOnLegacyTightPhone, 64)

        let qwen9BOnNonPhoneStableDevice = MemoryBudgetPlanner.quantizedBufferCacheLimitMB(
            totalBudgetMB: 8192,
            headroomMB: 2700,
            modelSizeGB: 5.6,
            isPhone: false,
            isMac: false,
            chipStableAtLongCtx: true,
            mode: .auto
        )
        XCTAssertGreaterThanOrEqual(qwen9BOnNonPhoneStableDevice, 5376)
    }

    func testDSRWindowIsBudgetDerivedWithout3584Cap() {
        let dsrForAirPlannerV2Budget = MemoryBudgetPlanner.dsrMaxCriticalForBudget(
            kvBudgetMB: 636,
            modelSizeGB: 5.6
        )
        XCTAssertGreaterThan(dsrForAirPlannerV2Budget, 3584)

        let expected = Int(Double(636) * 1024.0 * 1024.0 / (5.6 * 8.0 * 1024.0))
        XCTAssertEqual(dsrForAirPlannerV2Budget, expected)

        XCTAssertEqual(
            MemoryBudgetPlanner.dsrMaxCriticalForBudget(kvBudgetMB: 1, modelSizeGB: 5.6),
            2048
        )
    }

    func testNonPhoneDSRWindowScalesWithJetsamBudget() {
        let m3BudgetDerived = MemoryBudgetPlanner.dsrMaxCriticalForBudget(
            kvBudgetMB: 2184,
            modelSizeGB: 5.6
        )
        let iPadCap = MemoryBudgetPlanner.nonPhoneMobileDSRCap(totalBudgetMB: 8192)
        XCTAssertGreaterThan(m3BudgetDerived, iPadCap)
        XCTAssertEqual(iPadCap, 18_048)

        XCTAssertEqual(
            MemoryBudgetPlanner.dsrMaxCriticalForPlan(
                kvBudgetMB: 2184,
                modelSizeGB: 5.6,
                totalBudgetMB: 8192,
                isMac: false,
                isPhone: false
            ),
            iPadCap
        )

        XCTAssertEqual(
            MemoryBudgetPlanner.dsrMaxCriticalForPlan(
                kvBudgetMB: 586,
                modelSizeGB: 5.6,
                totalBudgetMB: 6144,
                isMac: false,
                isPhone: true
            ),
            MemoryBudgetPlanner.dsrMaxCriticalForBudget(kvBudgetMB: 586, modelSizeGB: 5.6)
        )

        XCTAssertEqual(
            MemoryBudgetPlanner.dsrMaxCriticalForPlan(
                kvBudgetMB: 2184,
                modelSizeGB: 5.6,
                totalBudgetMB: 8192,
                isMac: true,
                isPhone: false
            ),
            m3BudgetDerived
        )
    }

    func testDynamicOpsEnabledForAllChipStableModels() {
        XCTAssertTrue(
            MemoryBudgetPlanner.dynamicOpsEnabledForPlan(
                maxOps: 15,
                isMac: false,
                modelSizeGB: 3.8,
                chipStableAtLongCtx: true
            )
        )
        XCTAssertTrue(
            MemoryBudgetPlanner.dynamicOpsEnabledForPlan(
                maxOps: 15,
                isMac: false,
                modelSizeGB: 5.6,
                chipStableAtLongCtx: true
            )
        )
        XCTAssertFalse(
            MemoryBudgetPlanner.dynamicOpsEnabledForPlan(
                maxOps: 700,
                isMac: true,
                modelSizeGB: 5.6,
                chipStableAtLongCtx: true
            )
        )
    }

    func testHighThroughputPhoneRequiresBandwidthAndApple10Family() {
        XCTAssertTrue(
            MemoryBudgetPlanner.isHighThroughputPhone(
                effectiveBandwidthGBs: 55,
                metalFamilyTier: 10
            )
        )
        XCTAssertTrue(
            MemoryBudgetPlanner.isHighThroughputPhone(
                effectiveBandwidthGBs: 49,
                metalFamilyTier: 10
            )
        )
        XCTAssertFalse(
            MemoryBudgetPlanner.isHighThroughputPhone(
                effectiveBandwidthGBs: 43,
                metalFamilyTier: 9
            )
        )
        XCTAssertFalse(
            MemoryBudgetPlanner.isHighThroughputPhone(
                effectiveBandwidthGBs: 90,
                metalFamilyTier: 9
            )
        )
    }

    func testHighThroughputPhoneUsesPlannerV2Baseline() {
        let phoneBudget = MemoryBudgetPlanner.phoneCommandBufferBudget(highThroughputPhone: true)
        XCTAssertEqual(phoneBudget.maxOps, 20)
        XCTAssertEqual(phoneBudget.maxMB, 10)
        XCTAssertEqual(phoneBudget.peakReserveMB, 50)

        XCTAssertEqual(
            MemoryBudgetPlanner.prefillStepSizeForPlan(
                isMac: false,
                isHighThroughputPhone: true,
                headroomMB: 636
            ),
            256
        )

        XCTAssertTrue(
            MemoryBudgetPlanner.syncEvalForPlan(
                isHighThroughputPhone: true,
                chipStableAtLongCtx: true,
                headroomMB: 636
            )
        )

        let conservativeBudget = MemoryBudgetPlanner.phoneCommandBufferBudget(highThroughputPhone: false)
        XCTAssertEqual(conservativeBudget.maxOps, 5)
        XCTAssertEqual(conservativeBudget.maxMB, 5)
        XCTAssertEqual(conservativeBudget.peakReserveMB, 150)

        XCTAssertEqual(
            MemoryBudgetPlanner.prefillStepSizeForPlan(
                isMac: false,
                isHighThroughputPhone: false,
                headroomMB: 636
            ),
            128
        )

        XCTAssertFalse(
            MemoryBudgetPlanner.syncEvalForPlan(
                isHighThroughputPhone: false,
                chipStableAtLongCtx: true,
                headroomMB: 636
            )
        )
    }

    func testOnlineCalibratorRelaxesOneSafeKnobAtATime() {
        let defaults = UserDefaults(suiteName: "OnlineCalibratorTests.\(UUID().uuidString)")!
        let calibrator = OnlineCalibrator(
            storageIdentity: "test",
            initialOverrides: OnlineCalibrator.CalibrationOverrides(
                maxOpsPerBuffer: 5,
                prefillStepSize: 128,
                dynamicOpsFloor: 5
            ),
            userDefaults: defaults
        )

        XCTAssertNil(calibrator.record(.init(turnNumber: 1, tps: 20, contextLength: 512, thermalState: "nominal", availableMemoryMB: 1200)))
        XCTAssertNil(calibrator.record(.init(turnNumber: 2, tps: 14, contextLength: 1024, thermalState: "nominal", availableMemoryMB: 1200)))
        let decision = calibrator.record(.init(turnNumber: 3, tps: 14, contextLength: 2048, thermalState: "nominal", availableMemoryMB: 1200))

        XCTAssertEqual(decision?.overrides.maxOpsPerBuffer, 10)
        XCTAssertEqual(decision?.overrides.prefillStepSize, 128)
        XCTAssertEqual(decision?.overrides.dynamicOpsFloor, 5)
    }

    func testOnlineCalibratorIgnoresLateContextDecayUntilConsecutiveSteadyDecline() {
        let defaults = UserDefaults(suiteName: "OnlineCalibratorTests.\(UUID().uuidString)")!
        let calibrator = OnlineCalibrator(
            storageIdentity: "test",
            initialOverrides: OnlineCalibrator.CalibrationOverrides(
                maxOpsPerBuffer: 15,
                prefillStepSize: 512,
                dynamicOpsFloor: 5
            ),
            userDefaults: defaults
        )

        XCTAssertNil(calibrator.record(.init(turnNumber: 1, tps: 18, contextLength: 512, thermalState: "nominal", availableMemoryMB: 3000)))
        XCTAssertNil(calibrator.record(.init(turnNumber: 2, tps: 9.8, contextLength: 2048, thermalState: "nominal", availableMemoryMB: 3000)))
        XCTAssertNil(calibrator.record(.init(turnNumber: 3, tps: 9.7, contextLength: 3072, thermalState: "nominal", availableMemoryMB: 3000)))

        let decision = calibrator.record(.init(turnNumber: 4, tps: 9.6, contextLength: 4096, thermalState: "nominal", availableMemoryMB: 3000))
        XCTAssertEqual(decision?.overrides.maxOpsPerBuffer, 10)
        XCTAssertEqual(decision?.overrides.prefillStepSize, 512)
        XCTAssertEqual(decision?.overrides.dynamicOpsFloor, 5)

        XCTAssertNil(calibrator.record(.init(turnNumber: 5, tps: 7, contextLength: 8192, thermalState: "nominal", availableMemoryMB: 3000)))
        XCTAssertEqual(calibrator.currentOverrides.maxOpsPerBuffer, 10)
    }

    func testOnlineCalibratorDoesNotLowerDynamicFloorForHealthyLowTps() {
        let defaults = UserDefaults(suiteName: "OnlineCalibratorTests.\(UUID().uuidString)")!
        let calibrator = OnlineCalibrator(
            storageIdentity: "test",
            initialOverrides: OnlineCalibrator.CalibrationOverrides(
                maxOpsPerBuffer: 5,
                prefillStepSize: 128,
                dynamicOpsFloor: 5
            ),
            userDefaults: defaults
        )

        XCTAssertNil(calibrator.record(.init(turnNumber: 2, tps: 7.8, contextLength: 2048, thermalState: "nominal", availableMemoryMB: 1200)))
        XCTAssertNil(calibrator.record(.init(turnNumber: 3, tps: 7.6, contextLength: 3072, thermalState: "nominal", availableMemoryMB: 1200)))
        XCTAssertNil(calibrator.record(.init(turnNumber: 4, tps: 7.5, contextLength: 4096, thermalState: "nominal", availableMemoryMB: 1200)))

        XCTAssertEqual(calibrator.currentOverrides.dynamicOpsFloor, 5)
    }

    func testOnlineCalibratorLowersDynamicFloorUnderThermalOrMemoryPressure() {
        let defaults = UserDefaults(suiteName: "OnlineCalibratorTests.\(UUID().uuidString)")!
        let calibrator = OnlineCalibrator(
            storageIdentity: "test",
            initialOverrides: OnlineCalibrator.CalibrationOverrides(
                maxOpsPerBuffer: 5,
                prefillStepSize: 128,
                dynamicOpsFloor: 5
            ),
            userDefaults: defaults
        )

        XCTAssertNil(calibrator.record(.init(turnNumber: 2, tps: 8.0, contextLength: 2048, thermalState: "nominal", availableMemoryMB: 1200)))
        let decision = calibrator.record(.init(turnNumber: 3, tps: 7.9, contextLength: 3072, thermalState: "fair", availableMemoryMB: 1200))

        XCTAssertEqual(decision?.overrides.maxOpsPerBuffer, 5)
        XCTAssertEqual(decision?.overrides.prefillStepSize, 128)
        XCTAssertEqual(decision?.overrides.dynamicOpsFloor, 3)
    }

    func testPlannerEnablesNativePrefillAndDecodeFastPathsOnChipStableDevices() {
        XCTAssertTrue(
            MemoryBudgetPlanner.vendoredCommandBufferPrefillQMMEnabledForPlan(
                isMac: false,
                metalFamilyTier: 10,
                cacheLimitMB: 4096
            )
        )
        XCTAssertFalse(
            MemoryBudgetPlanner.vendoredCommandBufferPrefillQMMEnabledForPlan(
                isMac: false,
                metalFamilyTier: 8,
                cacheLimitMB: 4096
            )
        )
        XCTAssertFalse(
            MemoryBudgetPlanner.vendoredCommandBufferPrefillQMMEnabledForPlan(
                isMac: false,
                metalFamilyTier: 10,
                cacheLimitMB: 0
            )
        )
        XCTAssertTrue(
            MemoryBudgetPlanner.fusedGDNDecodeEnabledForPlan(
                isMac: false,
                metalFamilyTier: 10
            )
        )
    }

    func testModelSizeEstimateUsesCurrentWeightIndexAndIgnoresStaleShards() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-size-index-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data(repeating: 0, count: 1_024).write(
            to: directory.appendingPathComponent("current-a.safetensors")
        )
        try Data(repeating: 0, count: 2_048).write(
            to: directory.appendingPathComponent("current-b.safetensors")
        )
        try Data(repeating: 0, count: 8_192).write(
            to: directory.appendingPathComponent("stale-previous-model.safetensors")
        )
        try Data("""
        {
          "metadata": {},
          "weight_map": {
            "model.embed_tokens.weight": "current-a.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "current-b.safetensors",
            "model.layers.0.self_attn.k_proj.weight": "current-b.safetensors"
          }
        }
        """.utf8).write(to: directory.appendingPathComponent("model.safetensors.index.json"))

        let estimatedBytes = LLMEngine.estimateModelSizeGB(directory: directory) * 1024 * 1024 * 1024

        XCTAssertEqual(estimatedBytes, 3_072, accuracy: 0.001)
    }

    func testModelSizeEstimateFallsBackToAllSafetensorsWithoutIndex() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-size-fallback-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data(repeating: 0, count: 1_024).write(
            to: directory.appendingPathComponent("a.safetensors")
        )
        try Data(repeating: 0, count: 2_048).write(
            to: directory.appendingPathComponent("b.safetensors")
        )

        let estimatedBytes = LLMEngine.estimateModelSizeGB(directory: directory) * 1024 * 1024 * 1024

        XCTAssertEqual(estimatedBytes, 3_072, accuracy: 0.001)
    }

    func testBuildsQwenExecutionPlanFromConfigJSON() throws {
        let json = """
        {
          "model_type": "qwen3.5",
          "vocab_size": 128,
          "hidden_size": 8,
          "intermediate_size": 32,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "context_length": 64,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000,
          "edgeruntime_layer_plan": ["full_attention", "gdn"]
        }
        """
        let plan = try NativeRuntimeBridge.makeQwenExecutionPlan(
            configData: XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(plan.fullAttentionSteps.map(\.layerIndex), [0])
        XCTAssertEqual(plan.gdnSteps.map(\.layerIndex), [1])
    }

    func testBuildsQwenExecutionPlanFromPublicTextConfigLayerTypes() throws {
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
              "rope_theta": 10000000
            },
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
          }
        }
        """
        let plan = try NativeRuntimeBridge.makeQwenExecutionPlan(
            configData: XCTUnwrap(json.data(using: .utf8)),
            family: .qwen36
        )

        XCTAssertEqual(plan.architecture.family.rawValue, QwenModelFamily.qwen36.rawValue)
        XCTAssertEqual(plan.architecture.attentionHeadDimension, 256)
        XCTAssertEqual(plan.fullAttentionSteps.map(\.layerIndex), [3])
        XCTAssertEqual(plan.gdnSteps.map(\.layerIndex), [0, 1, 2])
    }

    func testBuildsQwen35ExecutionPlanFromPublicTextConfigWithoutFamilyOverride() throws {
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
              "rope_theta": 10000000
            },
            "layer_types": [
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention",
              "linear_attention",
              "linear_attention",
              "linear_attention",
              "full_attention"
            ]
          }
        }
        """
        let plan = try NativeRuntimeBridge.makeQwenExecutionPlan(
            configData: XCTUnwrap(json.data(using: .utf8))
        )

        XCTAssertEqual(plan.architecture.family.rawValue, QwenModelFamily.qwen35.rawValue)
        XCTAssertEqual(plan.architecture.layerCount, 32)
        XCTAssertEqual(plan.fullAttentionSteps.map(\.layerIndex), [3, 7, 11, 15, 19, 23, 27, 31])
        XCTAssertEqual(plan.gdnSteps.count, 24)
        XCTAssertEqual(plan.architecture.attentionHeadDimension, 256)
    }

    func testExposesNativeSpeechPlans() throws {
        let asr = try NativeRuntimeBridge.makeQwen3ASRPlan()
        let tts = try NativeRuntimeBridge.makeQwen3TTSPlan()
        let features = try NativeRuntimeBridge.qwenASRFeatureConfiguration()

        XCTAssertEqual(asr.modelFamily, .qwen3ASR)
        XCTAssertEqual(asr.modality, .asr)
        XCTAssertEqual(asr.preferredSampleRate, 16_000)
        XCTAssertFalse(asr.supportsStreaming)
        XCTAssertEqual(tts.modelFamily, .qwen3TTS)
        XCTAssertEqual(tts.modality, .tts)
        XCTAssertEqual(tts.preferredSampleRate, 24_000)
        XCTAssertFalse(tts.supportsStreaming)
        XCTAssertEqual(features.sampleRate, 16_000)
        XCTAssertEqual(features.melBinCount, 128)
    }

    func testReleaseReadinessSeparatesNativeSchedulerFromLegacyFallback() throws {
        let report = NativeRuntimeBridge.releaseReadinessReport()

        XCTAssertFalse(report.isReleaseReady)
        XCTAssertTrue(report.nativeMetalSchedulingAppliesToNativeRuntime)
        XCTAssertFalse(report.legacyFallbackUsesNativeMetalScheduling)
        XCTAssertEqual(
            report.gate("iphone-air-9b-t20-device-gate")?.status,
            .blocked
        )
        XCTAssertEqual(
            report.gate("grdb-cross-language-hash-gate")?.status,
            .blocked
        )
        XCTAssertEqual(
            report.gate("stt-dogfood-decision-gate")?.status,
            .pending
        )
        XCTAssertEqual(
            report.gate("native-dsr-kv-retention-gate")?.status,
            .blocked
        )
    }

    func testReleaseReadinessReportsDeferredNativeCapabilities() throws {
        let report = NativeRuntimeBridge.releaseReadinessReport()

        XCTAssertEqual(
            report.deferredCapabilities.map(\.id),
            [
                "dsr-kv-cache-bounding",
                "hidden-state-capture",
                "activation-steering-injection",
                "dynamic-gdn-offload",
                "low-rank-gdn-state",
            ]
        )
        XCTAssertTrue(
            try XCTUnwrap(report.deferredCapability("dsr-kv-cache-bounding"))
                .legacyFallbackBehavior
                .contains("no EdgeStudio DSR cache path")
        )
        XCTAssertTrue(
            try XCTUnwrap(report.deferredCapability("hidden-state-capture"))
                .legacyFallbackBehavior
                .contains("callWithCapture")
        )
        XCTAssertTrue(
            try XCTUnwrap(report.deferredCapability("activation-steering-injection"))
                .legacyFallbackBehavior
                .contains("stackResiduals")
        )
    }

    func testReleaseReadinessReportDecodesOldPayloadWithoutDeferredCapabilities() throws {
        let data = Data(
            """
            {
              "runtimeVersion": "0.1.4",
              "nativeMetalSchedulingAppliesToNativeRuntime": true,
              "legacyFallbackUsesNativeMetalScheduling": false,
              "gates": []
            }
            """.utf8
        )

        let report = try JSONDecoder().decode(NativeRuntimeReadinessReport.self, from: data)

        XCTAssertEqual(report.runtimeVersion, "0.1.4")
        XCTAssertTrue(report.nativeMetalSchedulingAppliesToNativeRuntime)
        XCTAssertFalse(report.legacyFallbackUsesNativeMetalScheduling)
        XCTAssertEqual(report.gates, [])
        XCTAssertEqual(report.deferredCapabilities, [])
    }

    func testReleaseReadinessPassesOnlyWhenAllEvidenceIsRecorded() throws {
        let report = NativeRuntimeBridge.releaseReadinessReport(
            hasIPhoneAir9BT20DeviceGate: true,
            hasGRDBCrossLanguageHashGate: true,
            hasLLMVLMSmokeGate: true,
            hasNativeDSRKVRetentionGate: true,
            hasSTTDogfoodDecision: true
        )

        XCTAssertTrue(report.isReleaseReady)
        XCTAssertTrue(report.gates.allSatisfy { $0.status == .passed })
    }
}
