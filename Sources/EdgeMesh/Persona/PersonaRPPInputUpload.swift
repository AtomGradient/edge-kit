// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public enum PersonaRPPInputUploadOps {
    public static let upload = "persona_rpp_input_upload"
    public static let uploadAck = "persona_rpp_input_upload_ack"
}

public enum PersonaRPPInputSourceKind: String, Codable, Equatable, Sendable {
    case appFacts = "app_facts"
    case importedFacts = "imported_facts"
    case correctionOverlay = "correction_overlay"
}

public struct PersonaRPPInputRecord: Codable, Equatable, Sendable {
    public var recordID: String
    public var kind: String?
    public var text: String
    public var tags: [String]?

    private enum CodingKeys: String, CodingKey {
        case recordID = "record_id"
        case kind
        case text
        case tags
    }

    public init(
        recordID: String,
        kind: String? = nil,
        text: String,
        tags: [String]? = nil
    ) {
        self.recordID = recordID
        self.kind = kind
        self.text = text
        self.tags = tags
    }
}

public struct PersonaRPPInputUploadPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.persona_rpp_input.v1"

    public var schemaVersion: String
    public var peerID: String
    public var appID: String
    public var baseModelID: String
    public var sourceKind: PersonaRPPInputSourceKind
    public var createdAt: Double
    public var records: [PersonaRPPInputRecord]
    public var recordsSHA256: String
    public var inputNote: String?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case peerID = "peer_id"
        case appID = "app_id"
        case baseModelID = "base_model_id"
        case sourceKind = "source_kind"
        case createdAt = "created_at"
        case records
        case recordsSHA256 = "records_sha256"
        case inputNote = "input_note"
    }

    public init(
        peerID: String,
        appID: String,
        baseModelID: String,
        sourceKind: PersonaRPPInputSourceKind,
        records: [PersonaRPPInputRecord],
        recordsSHA256: String? = nil,
        inputNote: String? = nil,
        createdAt: Double = Date().timeIntervalSince1970
    ) throws {
        self.schemaVersion = Self.schemaVersion
        self.peerID = peerID
        self.appID = appID
        self.baseModelID = baseModelID
        self.sourceKind = sourceKind
        self.createdAt = createdAt
        self.records = records
        self.recordsSHA256 = try recordsSHA256 ?? Self.recordsSHA256(records)
        self.inputNote = inputNote
    }

    public static func recordsSHA256(_ records: [PersonaRPPInputRecord]) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(records)
        return sha256Hex(data)
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

public struct PersonaRPPInputUploadMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: PersonaRPPInputUploadPayload

    public init(payload: PersonaRPPInputUploadPayload) {
        self.op = PersonaRPPInputUploadOps.upload
        self.payload = payload
    }
}

public struct PersonaRPPInputUploadAckPayload: Codable, Equatable, Sendable {
    public var ok: Bool
    public var peerID: String
    public var inputID: String?
    public var inputSHA256: String?
    public var recordsSHA256: String?
    public var recordCount: Int?
    public var message: String?

    private enum CodingKeys: String, CodingKey {
        case ok
        case peerID = "peer_id"
        case inputID = "input_id"
        case inputSHA256 = "input_sha256"
        case recordsSHA256 = "records_sha256"
        case recordCount = "record_count"
        case message
    }

    public init(
        ok: Bool,
        peerID: String,
        inputID: String? = nil,
        inputSHA256: String? = nil,
        recordsSHA256: String? = nil,
        recordCount: Int? = nil,
        message: String? = nil
    ) {
        self.ok = ok
        self.peerID = peerID
        self.inputID = inputID
        self.inputSHA256 = inputSHA256
        self.recordsSHA256 = recordsSHA256
        self.recordCount = recordCount
        self.message = message
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendPersonaRPPInputUpload(_ payload: PersonaRPPInputUploadPayload) throws {
        try sendJSON(PersonaRPPInputUploadMessage(payload: payload))
    }

    func sendPersonaRPPInputUploadAndWait(_ payload: PersonaRPPInputUploadPayload) async throws {
        try await sendJSONAndWait(PersonaRPPInputUploadMessage(payload: payload))
    }
}

@available(iOS 17.0, macOS 14.0, *)
public final class PersonaRPPInputUploadClient: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var ackTimeout: TimeInterval

        public init(ackTimeout: TimeInterval = 8) {
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
                return "persona_rpp_input_upload ack timed out after \(Int(seconds))s"
            case .ackWaitCancelled:
                return "persona_rpp_input_upload ack wait cancelled"
            case .alreadyWaiting:
                return "persona_rpp_input_upload ack wait already in progress"
            case .remoteError(let code, let message):
                return "persona_rpp_input_upload remote error \(code): \(message)"
            }
        }
    }

    private struct AckEnvelope: Decodable {
        let op: String
        let payload: PersonaRPPInputUploadAckPayload
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
        let resume: (Result<PersonaRPPInputUploadAckPayload, Error>) -> Void
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
        _ payload: PersonaRPPInputUploadPayload
    ) async throws -> PersonaRPPInputUploadAckPayload {
        let timeout = configuration.ackTimeout
        return try await withThrowingTaskGroup(of: PersonaRPPInputUploadAckPayload.self) { group in
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

    public static func decodeAckFrame(_ data: Data) -> PersonaRPPInputUploadAckPayload? {
        guard let envelope = try? JSONDecoder().decode(AckEnvelope.self, from: data),
              envelope.op == PersonaRPPInputUploadOps.uploadAck else {
            return nil
        }
        return envelope.payload
    }

    private func installPendingAckAndSend(
        _ payload: PersonaRPPInputUploadPayload
    ) async throws -> PersonaRPPInputUploadAckPayload {
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
                    try await connection.sendPersonaRPPInputUploadAndWait(payload)
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

    private func completePending(_ result: Result<PersonaRPPInputUploadAckPayload, Error>) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        pending?.resume(result)
    }

    private func clearPending(_ result: Result<PersonaRPPInputUploadAckPayload, Error>? = nil) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        if let result {
            pending?.resume(result)
        }
    }
}
