// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

/// SDK-level orchestration for applying a received Halo capsule package.
///
/// The coordinator owns the common wire protocol sequence:
/// `.received -> restore package -> .applied/.failed`. Runtime-specific restore
/// remains injected by the app or EdgeHalo integration layer.
public struct HaloCapsuleAutoRestoreCoordinator: Sendable {
    public struct Configuration: Sendable {
        public var runtimeVersion: String?
        public var now: @Sendable () -> Date
        public var maxErrorMessageLength: Int

        public init(
            runtimeVersion: String? = EdgeKitRuntime.version,
            now: @escaping @Sendable () -> Date = { Date() },
            maxErrorMessageLength: Int = 500
        ) {
            self.runtimeVersion = runtimeVersion
            self.now = now
            self.maxErrorMessageLength = max(0, maxErrorMessageLength)
        }
    }

    public struct RestoreResult: Equatable, Sendable {
        public var prefixTokenCount: Int?

        public init(prefixTokenCount: Int? = nil) {
            self.prefixTokenCount = prefixTokenCount
        }
    }

    public enum Outcome: Equatable, Sendable {
        case applied(RestoreResult)
        case failed(errorCode: String, errorMessage: String)
    }

    public struct Result: Equatable, Sendable {
        public var receivedPayload: HaloCapsuleApplyStatusPayload
        public var terminalPayload: HaloCapsuleApplyStatusPayload
        public var outcome: Outcome

        public init(
            receivedPayload: HaloCapsuleApplyStatusPayload,
            terminalPayload: HaloCapsuleApplyStatusPayload,
            outcome: Outcome
        ) {
            self.receivedPayload = receivedPayload
            self.terminalPayload = terminalPayload
            self.outcome = outcome
        }
    }

    public typealias RestorePackage = @Sendable (
        HaloCapsuleInboundTransferSession.CompletedPackage
    ) async throws -> RestoreResult
    public typealias SendApplyStatus = @Sendable (
        HaloCapsuleApplyStatusPayload
    ) async throws -> Void

    private let configuration: Configuration
    private let restorePackage: RestorePackage
    private let sendApplyStatus: SendApplyStatus

    public init(
        configuration: Configuration = Configuration(),
        restorePackage: @escaping RestorePackage,
        sendApplyStatus: @escaping SendApplyStatus
    ) {
        self.configuration = configuration
        self.restorePackage = restorePackage
        self.sendApplyStatus = sendApplyStatus
    }

    @discardableResult
    public func apply(
        _ completed: HaloCapsuleInboundTransferSession.CompletedPackage
    ) async throws -> Result {
        let received = makeStatusPayload(
            for: completed,
            status: .received
        )
        try await sendApplyStatus(received)

        let restoreResult: RestoreResult
        do {
            restoreResult = try await restorePackage(completed)
        } catch {
            let errorMessage = truncatedErrorMessage(error)
            let failed = makeStatusPayload(
                for: completed,
                status: .failed,
                errorCode: "restore_failed",
                errorMessage: errorMessage
            )
            try await sendApplyStatus(failed)
            return Result(
                receivedPayload: received,
                terminalPayload: failed,
                outcome: .failed(errorCode: "restore_failed", errorMessage: errorMessage)
            )
        }

        let applied = makeStatusPayload(
            for: completed,
            status: .applied,
            prefixTokenCount: restoreResult.prefixTokenCount
        )
        try await sendApplyStatus(applied)
        return Result(
            receivedPayload: received,
            terminalPayload: applied,
            outcome: .applied(restoreResult)
        )
    }

    private func makeStatusPayload(
        for completed: HaloCapsuleInboundTransferSession.CompletedPackage,
        status: HaloCapsuleApplyStatusValue,
        prefixTokenCount: Int? = nil,
        errorCode: String? = nil,
        errorMessage: String? = nil
    ) -> HaloCapsuleApplyStatusPayload {
        let message = completed.message
        return HaloCapsuleApplyStatusPayload(
            transferID: message.transferID,
            capsuleID: message.capsule.capsuleID,
            status: status,
            artifactSHA256: message.capsule.artifact.sha256,
            canonicalSHA256: completed.completeAck.canonicalSHA256,
            runtimeVersion: configuration.runtimeVersion,
            prefixTokenCount: prefixTokenCount,
            appliedAtUnixSeconds: configuration.now().timeIntervalSince1970,
            errorCode: errorCode,
            errorMessage: errorMessage
        )
    }

    private func truncatedErrorMessage(_ error: Error) -> String {
        let message = String(describing: error)
        return String(message.prefix(configuration.maxErrorMessageLength))
    }
}
