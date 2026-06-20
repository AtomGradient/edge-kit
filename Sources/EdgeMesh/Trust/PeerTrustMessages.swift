// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum PeerTrustOps {
    public static let deleted = "peer_trust_deleted"
    public static let deletedAck = "peer_trust_deleted_ack"
}

public struct PeerTrustDeletedPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.peer_trust_deleted.v1"

    public var schemaVersion: String
    public var peerID: String
    public var reason: String?
    public var deletedAtUnixSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case peerID = "peer_id"
        case reason
        case deletedAtUnixSeconds = "deleted_at_unix_seconds"
    }

    public init(
        peerID: String,
        reason: String? = nil,
        deletedAtUnixSeconds: Double? = Date().timeIntervalSince1970
    ) {
        self.schemaVersion = Self.schemaVersion
        self.peerID = peerID
        self.reason = reason
        self.deletedAtUnixSeconds = deletedAtUnixSeconds
    }
}

public struct PeerTrustDeletedMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: PeerTrustDeletedPayload

    public init(payload: PeerTrustDeletedPayload) {
        self.op = PeerTrustOps.deleted
        self.payload = payload
    }
}

public struct PeerTrustDeletedAckPayload: Codable, Equatable, Sendable {
    public var ok: Bool
    public var peerID: String
    public var wasKnown: Bool

    enum CodingKeys: String, CodingKey {
        case ok
        case peerID = "peer_id"
        case wasKnown = "was_known"
    }

    public init(ok: Bool, peerID: String, wasKnown: Bool) {
        self.ok = ok
        self.peerID = peerID
        self.wasKnown = wasKnown
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendPeerTrustDeleted(_ payload: PeerTrustDeletedPayload) throws {
        try sendJSON(PeerTrustDeletedMessage(payload: payload))
    }

    func sendPeerTrustDeletedAndWait(_ payload: PeerTrustDeletedPayload) async throws {
        try await sendJSONAndWait(PeerTrustDeletedMessage(payload: payload))
    }
}

@available(iOS 17.0, macOS 14.0, *)
public final class PeerTrustDeleteClient: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var ackTimeout: TimeInterval

        public init(ackTimeout: TimeInterval = 3) {
            self.ackTimeout = ackTimeout
        }
    }

    public enum ClientError: Error, LocalizedError, Equatable {
        case ackTimeout(seconds: TimeInterval)
        case ackWaitCancelled
        case alreadyWaiting
        case remoteError(code: String, message: String)

        public var errorDescription: String? {
            switch self {
            case .ackTimeout(let seconds):
                return "peer_trust_deleted ack timed out after \(Int(seconds))s"
            case .ackWaitCancelled:
                return "peer_trust_deleted ack wait cancelled"
            case .alreadyWaiting:
                return "peer_trust_deleted ack wait already in progress"
            case .remoteError(let code, let message):
                return "peer_trust_deleted remote error \(code): \(message)"
            }
        }
    }

    private struct AckEnvelope: Decodable {
        let op: String
        let payload: PeerTrustDeletedAckPayload
    }

    private struct ErrorPayload: Decodable {
        let code: String
        let message: String
    }

    private struct ErrorEnvelope: Decodable {
        let op: String
        let payload: ErrorPayload
    }

    private struct PendingAck {
        let peerID: String
        let resume: (Result<PeerTrustDeletedAckPayload, Error>) -> Void
    }

    private let connection: MeshConnection
    private let configuration: Configuration
    private let lock = NSLock()
    private var pendingAck: PendingAck?

    public init(
        connection: MeshConnection,
        configuration: Configuration = Configuration()
    ) {
        self.connection = connection
        self.configuration = configuration
        connection.onFrame { [weak self] data in
            self?.handleFrame(data)
        }
    }

    public func sendAndWaitForAck(
        _ payload: PeerTrustDeletedPayload
    ) async throws -> PeerTrustDeletedAckPayload {
        let timeout = configuration.ackTimeout
        return try await withThrowingTaskGroup(of: PeerTrustDeletedAckPayload.self) { group in
            defer {
                group.cancelAll()
                clearPending(.failure(ClientError.ackWaitCancelled))
            }

            group.addTask {
                try await self.installPendingAckAndSend(payload)
            }
            group.addTask {
                let nanoseconds = UInt64(max(timeout, 0.001) * 1_000_000_000)
                try await Task.sleep(nanoseconds: nanoseconds)
                throw ClientError.ackTimeout(seconds: timeout)
            }

            guard let result = try await group.next() else {
                throw ClientError.ackWaitCancelled
            }
            return result
        }
    }

    public static func decodeAckFrame(_ data: Data) -> PeerTrustDeletedAckPayload? {
        guard let envelope = try? JSONDecoder().decode(AckEnvelope.self, from: data),
              envelope.op == PeerTrustOps.deletedAck else {
            return nil
        }
        return envelope.payload
    }

    private func installPendingAckAndSend(
        _ payload: PeerTrustDeletedPayload
    ) async throws -> PeerTrustDeletedAckPayload {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            if pendingAck != nil {
                lock.unlock()
                continuation.resume(throwing: ClientError.alreadyWaiting)
                return
            }
            pendingAck = PendingAck(peerID: payload.peerID) { result in
                switch result {
                case .success(let ack):
                    continuation.resume(returning: ack)
                case .failure(let error):
                    continuation.resume(throwing: error)
                }
            }
            lock.unlock()

            Task {
                do {
                    try await connection.sendPeerTrustDeletedAndWait(payload)
                } catch {
                    completePending(.failure(error))
                }
            }
        }
    }

    private func handleFrame(_ data: Data) {
        if let ack = Self.decodeAckFrame(data) {
            lock.lock()
            let pending = pendingAck
            lock.unlock()
            guard pending?.peerID == ack.peerID else {
                return
            }
            completePending(.success(ack))
            return
        }

        guard let envelope = try? JSONDecoder().decode(ErrorEnvelope.self, from: data),
              envelope.op == "error" else {
            return
        }
        completePending(.failure(ClientError.remoteError(
            code: envelope.payload.code,
            message: envelope.payload.message
        )))
    }

    private func completePending(_ result: Result<PeerTrustDeletedAckPayload, Error>) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        pending?.resume(result)
    }

    private func clearPending(_ result: Result<PeerTrustDeletedAckPayload, Error>? = nil) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        if let result {
            pending?.resume(result)
        }
    }
}
