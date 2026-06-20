// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

@available(iOS 17.0, macOS 14.0, *)
public final class EventSyncClient: @unchecked Sendable {
    public typealias SendFrame = (Data) throws -> Void
    public typealias RegisterFrameHandler = (@escaping (Data) -> Void) -> Void

    public struct Configuration: Sendable {
        public let uploadOp: String
        public let ackOp: String
        public let ackTimeout: TimeInterval

        public init(
            uploadOp: String = "event_upload",
            ackOp: String = "event_upload_ack",
            ackTimeout: TimeInterval = 30
        ) {
            self.uploadOp = uploadOp
            self.ackOp = ackOp
            self.ackTimeout = ackTimeout
        }
    }

    public enum SyncError: Error, Equatable, LocalizedError, Sendable {
        case uploadInFlight
        case ackTimeout(seconds: TimeInterval)
        case ackWaitCancelled

        public var errorDescription: String? {
            switch self {
            case .uploadInFlight:
                return "event upload already in flight"
            case .ackTimeout(let seconds):
                return "event upload ack timed out after \(Int(seconds))s"
            case .ackWaitCancelled:
                return "event upload ack wait cancelled"
            }
        }
    }

    private struct UploadEnvelope: Encodable {
        let op: String
        let payload: EventUploadBatch
    }

    private struct AckEnvelope: Decodable {
        let op: String
        let payload: EventUploadAck
    }

    private struct PendingAck {
        let resume: (Result<EventUploadAck, Error>) -> Void
    }

    private let collector: DataCollector
    private let sendFrame: SendFrame
    private let configuration: Configuration
    private let lock = NSLock()
    private var pendingAck: PendingAck?

    public init(
        collector: DataCollector,
        sendFrame: @escaping SendFrame,
        registerFrameHandler: RegisterFrameHandler,
        configuration: Configuration = Configuration()
    ) {
        self.collector = collector
        self.sendFrame = sendFrame
        self.configuration = configuration
        registerFrameHandler { [weak self] data in
            self?.handleFrame(data)
        }
    }

    public convenience init(
        collector: DataCollector,
        connection: MeshConnection,
        configuration: Configuration = Configuration()
    ) {
        self.init(
            collector: collector,
            sendFrame: { data in try connection.send(data) },
            registerFrameHandler: { handler in connection.onFrame(handler) },
            configuration: configuration
        )
    }

    @discardableResult
    public func flush(
        to peerId: String,
        batchSize: Int = 100
    ) async throws -> Int {
        try await collector.flush(to: peerId, batchSize: batchSize) { [self] batch in
            try await sendBatchAndWaitForAck(batch)
        }
    }

    public static func makeEventUploadFrame(
        batch: EventUploadBatch,
        op: String = "event_upload"
    ) throws -> Data {
        try JSONEncoder().encode(UploadEnvelope(op: op, payload: batch))
    }

    public static func decodeEventUploadAckFrame(
        _ data: Data,
        ackOp: String = "event_upload_ack"
    ) -> EventUploadAck? {
        guard let envelope = try? JSONDecoder().decode(AckEnvelope.self, from: data),
              envelope.op == ackOp else {
            return nil
        }
        return envelope.payload
    }

    private func sendBatchAndWaitForAck(_ batch: EventUploadBatch) async throws -> EventUploadAck {
        let frame = try Self.makeEventUploadFrame(batch: batch, op: configuration.uploadOp)
        let timeout = configuration.ackTimeout

        return try await withThrowingTaskGroup(of: EventUploadAck.self) { group in
            group.addTask { [self] in
                try await installPendingAckAndSend(frame)
            }
            group.addTask { [self] in
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                let error = SyncError.ackTimeout(seconds: timeout)
                completePending(.failure(error))
                throw error
            }

            guard let result = try await group.next() else {
                throw SyncError.ackWaitCancelled
            }
            group.cancelAll()
            clearPending()
            return result
        }
    }

    private func installPendingAckAndSend(_ frame: Data) async throws -> EventUploadAck {
        try await withCheckedThrowingContinuation { continuation in
            let pending = PendingAck { result in
                continuation.resume(with: result)
            }

            lock.lock()
            if pendingAck != nil {
                lock.unlock()
                continuation.resume(throwing: SyncError.uploadInFlight)
                return
            }
            pendingAck = pending
            lock.unlock()

            do {
                try sendFrame(frame)
            } catch {
                completePending(.failure(error))
            }
        }
    }

    private func handleFrame(_ data: Data) {
        guard let ack = Self.decodeEventUploadAckFrame(data, ackOp: configuration.ackOp) else {
            return
        }
        completePending(.success(ack))
    }

    private func completePending(_ result: Result<EventUploadAck, Error>) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        pending?.resume(result)
    }

    private func clearPending() {
        lock.lock()
        pendingAck = nil
        lock.unlock()
    }
}
