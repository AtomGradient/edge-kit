// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import EdgeInference
import Foundation

public enum PersonaSourceUploadOps {
    public static let upload = "persona_source_upload"
    public static let uploadAck = "persona_source_upload_ack"
}

public enum PersonaSourceKind: String, Codable, Equatable, Sendable {
    case toolSchemaOnly = "tool_schema_only"
    case deviceRPPProfile = "device_rpp_profile"
    case hostRPPProfile = "host_rpp_profile"
}

public struct PersonaSourceUploadPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.persona_source_upload.v1"

    public var schemaVersion: String
    public var peerID: String
    public var appID: String
    public var baseModelID: String
    public var toolSchemaExport: ToolSchemaExport
    public var toolSchemaSHA256: String
    public var profileBody: String?
    public var profileBodySHA256: String?
    public var rppRunID: String?
    public var sourceKind: PersonaSourceKind
    public var createdAt: Double

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case peerID = "peer_id"
        case appID = "app_id"
        case baseModelID = "base_model_id"
        case toolSchemaExport = "tool_schema_export"
        case toolSchemaSHA256 = "tool_schema_sha256"
        case profileBody = "profile_body"
        case profileBodySHA256 = "profile_body_sha256"
        case rppRunID = "rpp_run_id"
        case sourceKind = "source_kind"
        case createdAt = "created_at"
    }

    public init(
        peerID: String,
        appID: String,
        baseModelID: String,
        toolSchemaSnapshot: ToolSchemaSnapshot,
        profileBody: String? = nil,
        rppRunID: String? = nil,
        sourceKind: PersonaSourceKind,
        createdAt: Double = Date().timeIntervalSince1970
    ) {
        self.init(
            peerID: peerID,
            appID: appID,
            baseModelID: baseModelID,
            toolSchemaExport: toolSchemaSnapshot.export,
            toolSchemaSHA256: toolSchemaSnapshot.sha256,
            profileBody: profileBody,
            profileBodySHA256: profileBody.map(Self.sha256Hex),
            rppRunID: rppRunID,
            sourceKind: sourceKind,
            createdAt: createdAt
        )
    }

    public init(
        peerID: String,
        appID: String,
        baseModelID: String,
        toolSchemaExport: ToolSchemaExport,
        toolSchemaSHA256: String,
        profileBody: String? = nil,
        profileBodySHA256: String? = nil,
        rppRunID: String? = nil,
        sourceKind: PersonaSourceKind,
        createdAt: Double = Date().timeIntervalSince1970
    ) {
        self.schemaVersion = Self.schemaVersion
        self.peerID = peerID
        self.appID = appID
        self.baseModelID = baseModelID
        self.toolSchemaExport = toolSchemaExport
        self.toolSchemaSHA256 = toolSchemaSHA256
        self.profileBody = profileBody
        self.profileBodySHA256 = profileBodySHA256 ?? profileBody.map(Self.sha256Hex)
        self.rppRunID = rppRunID
        self.sourceKind = sourceKind
        self.createdAt = createdAt
    }

    public static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

public struct PersonaSourceUploadMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: PersonaSourceUploadPayload

    public init(payload: PersonaSourceUploadPayload) {
        self.op = PersonaSourceUploadOps.upload
        self.payload = payload
    }
}

public struct PersonaSourceUploadAckPayload: Codable, Equatable, Sendable {
    public var ok: Bool
    public var peerID: String
    public var sourceID: String?
    public var sourceSHA256: String?
    public var message: String?

    private enum CodingKeys: String, CodingKey {
        case ok
        case peerID = "peer_id"
        case sourceID = "source_id"
        case sourceSHA256 = "source_sha256"
        case message
    }

    public init(
        ok: Bool,
        peerID: String,
        sourceID: String? = nil,
        sourceSHA256: String? = nil,
        message: String? = nil
    ) {
        self.ok = ok
        self.peerID = peerID
        self.sourceID = sourceID
        self.sourceSHA256 = sourceSHA256
        self.message = message
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendPersonaSourceUpload(_ payload: PersonaSourceUploadPayload) throws {
        try sendJSON(PersonaSourceUploadMessage(payload: payload))
    }

    func sendPersonaSourceUploadAndWait(_ payload: PersonaSourceUploadPayload) async throws {
        try await sendJSONAndWait(PersonaSourceUploadMessage(payload: payload))
    }
}

@available(iOS 17.0, macOS 14.0, *)
public final class PersonaSourceUploadClient: @unchecked Sendable {
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
                return "persona_source_upload ack timed out after \(Int(seconds))s"
            case .ackWaitCancelled:
                return "persona_source_upload ack wait cancelled"
            case .alreadyWaiting:
                return "persona_source_upload ack wait already in progress"
            case .remoteError(let code, let message):
                return "persona_source_upload remote error \(code): \(message)"
            }
        }
    }

    private struct AckEnvelope: Decodable {
        let op: String
        let payload: PersonaSourceUploadAckPayload
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
        let resume: (Result<PersonaSourceUploadAckPayload, Error>) -> Void
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
        _ payload: PersonaSourceUploadPayload
    ) async throws -> PersonaSourceUploadAckPayload {
        let timeout = configuration.ackTimeout
        return try await withThrowingTaskGroup(of: PersonaSourceUploadAckPayload.self) { group in
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

    public static func decodeAckFrame(_ data: Data) -> PersonaSourceUploadAckPayload? {
        guard let envelope = try? JSONDecoder().decode(AckEnvelope.self, from: data),
              envelope.op == PersonaSourceUploadOps.uploadAck else {
            return nil
        }
        return envelope.payload
    }

    private func installPendingAckAndSend(
        _ payload: PersonaSourceUploadPayload
    ) async throws -> PersonaSourceUploadAckPayload {
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
                    try await connection.sendPersonaSourceUploadAndWait(payload)
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

    private func completePending(_ result: Result<PersonaSourceUploadAckPayload, Error>) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        pending?.resume(result)
    }

    private func clearPending(_ result: Result<PersonaSourceUploadAckPayload, Error>? = nil) {
        lock.lock()
        let pending = pendingAck
        pendingAck = nil
        lock.unlock()
        if let result {
            pending?.resume(result)
        }
    }
}
