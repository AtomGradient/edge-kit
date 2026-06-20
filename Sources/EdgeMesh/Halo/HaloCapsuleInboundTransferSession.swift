// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Small state machine for receiving one host-pushed Halo capsule package.
///
/// `HaloCapsulePackageReceiver` owns package validation and disk writes. This
/// wrapper adds the mesh-facing concerns: ignore unrelated frames, emit offer /
/// complete ACK frames, and expose the completed package directory to the app
/// runtime that performs Neural Imprint restore.
public final class HaloCapsuleInboundTransferSession: @unchecked Sendable {
    public struct CompletedPackage: Equatable, Sendable {
        public let message: HaloCapsuleMeshMessage
        public let packageDirectory: URL
        public let completeAck: HaloCapsuleTransferAck

        public init(
            message: HaloCapsuleMeshMessage,
            packageDirectory: URL,
            completeAck: HaloCapsuleTransferAck
        ) {
            self.message = message
            self.packageDirectory = packageDirectory
            self.completeAck = completeAck
        }
    }

    public struct Result: Equatable, Sendable {
        public let handled: Bool
        public let outgoingFrames: [Data]
        public let acceptedOffer: HaloCapsuleMeshMessage?
        public let rejectedOfferAck: HaloCapsuleTransferAck?
        public let completedPackage: CompletedPackage?

        public init(
            handled: Bool,
            outgoingFrames: [Data] = [],
            acceptedOffer: HaloCapsuleMeshMessage? = nil,
            rejectedOfferAck: HaloCapsuleTransferAck? = nil,
            completedPackage: CompletedPackage? = nil
        ) {
            self.handled = handled
            self.outgoingFrames = outgoingFrames
            self.acceptedOffer = acceptedOffer
            self.rejectedOfferAck = rejectedOfferAck
            self.completedPackage = completedPackage
        }

        public static let ignored = Result(handled: false)
    }

    public typealias ConfigurationProvider = (HaloCapsuleMeshMessage) throws -> HaloCapsulePackageReceiver.Configuration

    private final class ActiveState {
        let receiver: HaloCapsulePackageReceiver
        let packageDirectory: URL
        var message: HaloCapsuleMeshMessage?

        init(receiver: HaloCapsulePackageReceiver, packageDirectory: URL) {
            self.receiver = receiver
            self.packageDirectory = packageDirectory
        }
    }

    private let configurationProvider: ConfigurationProvider
    private var activeState: ActiveState?

    public init(configurationProvider: @escaping ConfigurationProvider) {
        self.configurationProvider = configurationProvider
    }

    public var hasActiveTransfer: Bool {
        activeState != nil
    }

    public var activeOffer: HaloCapsuleMeshMessage? {
        activeState?.message
    }

    public func reset() {
        activeState = nil
    }

    public func receive(_ frame: Data) throws -> Result {
        let decoded: HaloCapsuleTransferFrame
        do {
            decoded = try HaloCapsulePackageTransfer.decodeFrame(frame)
        } catch HaloCapsuleTransportError.unsupportedMessageKind {
            return .ignored
        }

        switch decoded {
        case .offer(let offer):
            let configuration = try configurationProvider(offer)
            let state = ActiveState(
                receiver: HaloCapsulePackageReceiver(configuration: configuration),
                packageDirectory: configuration.destinationDirectory
            )
            activeState = state
            return try process(frame, state: state)

        case .chunkHeader, .binary, .complete:
            guard let state = activeState else {
                return .ignored
            }
            return try process(frame, state: state)

        case .offerAck, .completeAck, .applyStatus:
            return .ignored
        }
    }

    private func process(_ frame: Data, state: ActiveState) throws -> Result {
        let event = try state.receiver.receive(frame)
        switch event {
        case .offerAccepted(let ack):
            guard case .offer(let offer) = try HaloCapsulePackageTransfer.decodeFrame(frame) else {
                return Result(handled: true)
            }
            state.message = offer
            return Result(
                handled: true,
                outgoingFrames: [try state.receiver.offerAckFrame(ack)],
                acceptedOffer: offer
            )

        case .offerRejected(let ack):
            activeState = nil
            return Result(
                handled: true,
                outgoingFrames: [try state.receiver.offerAckFrame(ack)],
                rejectedOfferAck: ack
            )

        case .chunkHeaderAccepted, .chunkWritten:
            return Result(handled: true)

        case .completed(let message, let ack):
            activeState = nil
            let completed = CompletedPackage(
                message: message,
                packageDirectory: state.packageDirectory,
                completeAck: ack
            )
            return Result(
                handled: true,
                outgoingFrames: [try state.receiver.completeAckFrame(ack)],
                completedPackage: completed
            )
        }
    }
}
