// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Runtime-only throughput feedback loop for safe inference knobs.
///
/// The calibrator is deliberately conservative: it only adjusts command buffer
/// scheduling and prefill chunk size, never quality-affecting cache windows.
public final class OnlineCalibrator: @unchecked Sendable {
    public struct CalibrationOverrides: Codable, Equatable, Sendable {
        public let maxOpsPerBuffer: Int
        public let prefillStepSize: Int
        public let dynamicOpsFloor: Int

        var nativeLoadOptions: NativeRuntimeLoadOptions {
            NativeRuntimeLoadOptions(
                maxOpsPerBuffer: maxOpsPerBuffer,
                maxMBPerBuffer: maxOpsPerBuffer,
                dynamicOpsFloor: dynamicOpsFloor,
                prefillStepSize: prefillStepSize
            )
        }
    }

    public struct Sample: Sendable {
        public let turnNumber: Int
        public let tps: Double
        public let contextLength: Int
        public let thermalState: String
        public let availableMemoryMB: Int

        public init(
            turnNumber: Int,
            tps: Double,
            contextLength: Int,
            thermalState: String,
            availableMemoryMB: Int
        ) {
            self.turnNumber = turnNumber
            self.tps = tps
            self.contextLength = contextLength
            self.thermalState = thermalState
            self.availableMemoryMB = availableMemoryMB
        }
    }

    public struct Decision: Sendable {
        public let averageTps: Double
        public let overrides: CalibrationOverrides
        public let reason: String
    }

    private struct PersistedState: Codable {
        var avgTps: Double
        var sampleCount: Int
        var lastMaxOps: Int
        var lastPrefillStep: Int
        var lastDynamicOpsFloor: Int
        var lastUpdated: Date
    }

    private static let storagePrefix = "com.atomgradient.edgekit.online-calibration.v2_2"

    private let storageKey: String
    private let userDefaults: UserDefaults
    private var samples: [Sample] = []
    private var lastCommittedAverageTps: Double?
    private var persistedSampleCount = 0
    private var current: CalibrationOverrides

    public init(
        storageIdentity: String,
        initialOverrides: CalibrationOverrides,
        userDefaults: UserDefaults = .standard
    ) {
        self.storageKey = "\(Self.storagePrefix).\(storageIdentity)"
        self.userDefaults = userDefaults
        if let data = userDefaults.data(forKey: storageKey),
           let state = try? JSONDecoder().decode(PersistedState.self, from: data) {
            self.current = CalibrationOverrides(
                maxOpsPerBuffer: state.lastMaxOps,
                prefillStepSize: state.lastPrefillStep,
                dynamicOpsFloor: state.lastDynamicOpsFloor
            )
            self.lastCommittedAverageTps = state.avgTps
            self.persistedSampleCount = state.sampleCount
        } else {
            self.current = initialOverrides
        }
    }

    public static func storageIdentity(
        deviceModel: String,
        modelIdentifier: String,
        quantization: String
    ) -> String {
        [deviceModel, modelIdentifier, quantization]
            .map { sanitize($0) }
            .joined(separator: "_")
    }

    public var currentOverrides: CalibrationOverrides {
        current
    }

    @discardableResult
    public func record(_ sample: Sample) -> Decision? {
        guard sample.tps.isFinite, sample.tps > 0 else { return nil }
        guard (2...4).contains(sample.turnNumber) else { return nil }

        samples.append(sample)
        persistedSampleCount += 1
        if samples.count > 3 {
            samples.removeFirst(samples.count - 3)
        }
        guard samples.count >= 2 else {
            persist(avgTps: sample.tps)
            return nil
        }

        let avg = samples.map(\.tps).reduce(0, +) / Double(samples.count)
        let target = targetOverrides(for: avg, latest: sample)
        guard target != current else {
            persist(avgTps: avg)
            return nil
        }

        let directionIsRetreat = target.maxOpsPerBuffer < current.maxOpsPerBuffer
            || target.prefillStepSize < current.prefillStepSize
            || target.dynamicOpsFloor < current.dynamicOpsFloor
        let thermalBad = Self.isThermalDegraded(sample.thermalState)
        let memoryBad = Self.isMemoryDegraded(sample.availableMemoryMB)

        let previousAvg = lastCommittedAverageTps
        let relativeChange = previousAvg.map { avg > 0 ? abs(avg - $0) / max($0, 0.001) : 0 } ?? 1
        let hasConsecutiveDecline = Self.hasConsecutiveDecline(samples)
        let canChange: Bool
        if directionIsRetreat {
            canChange = thermalBad
                || memoryBad
                || (previousAvg != nil && relativeChange > 0.20)
                || avg < 9.0
                || hasConsecutiveDecline
        } else {
            canChange = !thermalBad && !memoryBad && (previousAvg == nil || relativeChange > 0.10 || avg >= 12.0)
        }
        guard canChange else {
            persist(avgTps: avg)
            return nil
        }

        let next = oneStep(from: current, toward: target)
        current = next
        lastCommittedAverageTps = avg
        persist(avgTps: avg)
        return Decision(
            averageTps: avg,
            overrides: next,
            reason: directionIsRetreat ? "retreat" : "relax"
        )
    }

    private func targetOverrides(for avgTps: Double, latest: Sample) -> CalibrationOverrides {
        let pressureDegraded = Self.isThermalDegraded(latest.thermalState)
            || Self.isMemoryDegraded(latest.availableMemoryMB)
        let healthyFloor = max(current.dynamicOpsFloor, 5)
        if pressureDegraded {
            return CalibrationOverrides(maxOpsPerBuffer: 5, prefillStepSize: 128, dynamicOpsFloor: 3)
        }
        if avgTps >= 12.0, latest.availableMemoryMB >= 1_000 {
            return CalibrationOverrides(
                maxOpsPerBuffer: 15,
                prefillStepSize: latest.availableMemoryMB >= 2_000 ? 512 : 256,
                dynamicOpsFloor: 8
            )
        }
        if avgTps >= 9.6, latest.availableMemoryMB >= 850 {
            return CalibrationOverrides(maxOpsPerBuffer: 10, prefillStepSize: 256, dynamicOpsFloor: healthyFloor)
        }
        return CalibrationOverrides(maxOpsPerBuffer: 5, prefillStepSize: 128, dynamicOpsFloor: healthyFloor)
    }

    private func oneStep(
        from current: CalibrationOverrides,
        toward target: CalibrationOverrides
    ) -> CalibrationOverrides {
        if current.maxOpsPerBuffer != target.maxOpsPerBuffer {
            return CalibrationOverrides(
                maxOpsPerBuffer: step(current.maxOpsPerBuffer, toward: target.maxOpsPerBuffer, ladder: [5, 10, 15]),
                prefillStepSize: current.prefillStepSize,
                dynamicOpsFloor: current.dynamicOpsFloor
            )
        }
        if current.prefillStepSize != target.prefillStepSize {
            return CalibrationOverrides(
                maxOpsPerBuffer: current.maxOpsPerBuffer,
                prefillStepSize: step(current.prefillStepSize, toward: target.prefillStepSize, ladder: [128, 256, 512]),
                dynamicOpsFloor: current.dynamicOpsFloor
            )
        }
        if current.dynamicOpsFloor != target.dynamicOpsFloor {
            return CalibrationOverrides(
                maxOpsPerBuffer: current.maxOpsPerBuffer,
                prefillStepSize: current.prefillStepSize,
                dynamicOpsFloor: step(current.dynamicOpsFloor, toward: target.dynamicOpsFloor, ladder: [3, 5, 8])
            )
        }
        return current
    }

    private func step(_ value: Int, toward target: Int, ladder: [Int]) -> Int {
        guard let currentIndex = ladder.firstIndex(of: value),
              let targetIndex = ladder.firstIndex(of: target),
              currentIndex != targetIndex else {
            return target
        }
        return ladder[currentIndex + (targetIndex > currentIndex ? 1 : -1)]
    }

    private func persist(avgTps: Double) {
        let state = PersistedState(
            avgTps: avgTps,
            sampleCount: persistedSampleCount,
            lastMaxOps: current.maxOpsPerBuffer,
            lastPrefillStep: current.prefillStepSize,
            lastDynamicOpsFloor: current.dynamicOpsFloor,
            lastUpdated: Date()
        )
        if let data = try? JSONEncoder().encode(state) {
            userDefaults.set(data, forKey: storageKey)
        }
    }

    private static func isThermalDegraded(_ thermalState: String) -> Bool {
        let lower = thermalState.lowercased()
        return lower.contains("fair") || lower.contains("serious") || lower.contains("critical")
    }

    private static func isMemoryDegraded(_ availableMemoryMB: Int) -> Bool {
        availableMemoryMB < 700
    }

    private static func hasConsecutiveDecline(_ samples: [Sample]) -> Bool {
        guard samples.count >= 3 else { return false }
        let tail = samples.suffix(3)
        guard tail.count == 3 else { return false }
        let values = tail.map(\.tps)
        return values[1] < values[0] && values[2] < values[1]
    }

    private static func sanitize(_ component: String) -> String {
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        let scalars = component.unicodeScalars.map { scalar -> Character in
            allowed.contains(Character(scalar)) ? Character(scalar) : "_"
        }
        return String(scalars).isEmpty ? "unknown" : String(scalars)
    }
}
