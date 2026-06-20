// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public enum HaloCapsuleTransferOps {
    public static let offer = HaloCapsuleMeshMessage.offerKind
    public static let offerAck = "halo_capsule_offer_ack"
    public static let chunk = "halo_capsule_chunk"
    public static let complete = "halo_capsule_complete"
    public static let completeAck = "halo_capsule_complete_ack"
    public static let applyStatus = "halo_capsule_apply_status"
}

public struct HaloCapsuleTransferAck: Codable, Equatable, Sendable {
    public var transferID: String
    public var accepted: Bool
    public var reason: String?
    public var canonicalSHA256: String?

    enum CodingKeys: String, CodingKey {
        case transferID = "transfer_id"
        case accepted
        case reason
        case canonicalSHA256 = "canonical_sha256"
    }

    public init(
        transferID: String,
        accepted: Bool,
        reason: String? = nil,
        canonicalSHA256: String? = nil
    ) {
        self.transferID = transferID
        self.accepted = accepted
        self.reason = reason
        self.canonicalSHA256 = canonicalSHA256
    }
}

public struct HaloCapsuleChunkHeader: Codable, Equatable, Sendable {
    public var transferID: String
    public var fileName: String
    public var offset: Int
    public var byteCount: Int
    public var sha256: String

    enum CodingKeys: String, CodingKey {
        case transferID = "transfer_id"
        case fileName = "file_name"
        case offset
        case byteCount = "byte_count"
        case sha256
    }

    public init(
        transferID: String,
        fileName: String,
        offset: Int,
        byteCount: Int,
        sha256: String
    ) {
        self.transferID = transferID
        self.fileName = fileName
        self.offset = offset
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct HaloCapsuleTransferComplete: Codable, Equatable, Sendable {
    public var transferID: String

    enum CodingKeys: String, CodingKey {
        case transferID = "transfer_id"
    }

    public init(transferID: String) {
        self.transferID = transferID
    }
}

public enum HaloCapsuleApplyStatusValue: String, Codable, Equatable, Sendable {
    case received
    case applied
    case failed
}

public struct HaloCapsuleApplyStatusPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.halo_capsule_apply_status.v1"

    public var schemaVersion: String
    public var transferID: String
    public var capsuleID: String
    public var status: HaloCapsuleApplyStatusValue
    public var artifactSHA256: String?
    public var canonicalSHA256: String?
    public var runtimeVersion: String?
    public var prefixTokenCount: Int?
    public var appliedAtUnixSeconds: Double?
    public var errorCode: String?
    public var errorMessage: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case transferID = "transfer_id"
        case capsuleID = "capsule_id"
        case status
        case artifactSHA256 = "artifact_sha256"
        case canonicalSHA256 = "canonical_sha256"
        case runtimeVersion = "runtime_version"
        case prefixTokenCount = "prefix_token_count"
        case appliedAtUnixSeconds = "applied_at_unix_seconds"
        case errorCode = "error_code"
        case errorMessage = "error_message"
    }

    public init(
        schemaVersion: String = Self.schemaVersion,
        transferID: String,
        capsuleID: String,
        status: HaloCapsuleApplyStatusValue,
        artifactSHA256: String? = nil,
        canonicalSHA256: String? = nil,
        runtimeVersion: String? = nil,
        prefixTokenCount: Int? = nil,
        appliedAtUnixSeconds: Double? = nil,
        errorCode: String? = nil,
        errorMessage: String? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.transferID = transferID
        self.capsuleID = capsuleID
        self.status = status
        self.artifactSHA256 = artifactSHA256
        self.canonicalSHA256 = canonicalSHA256
        self.runtimeVersion = runtimeVersion
        self.prefixTokenCount = prefixTokenCount
        self.appliedAtUnixSeconds = appliedAtUnixSeconds
        self.errorCode = errorCode
        self.errorMessage = errorMessage
    }
}

public struct HaloCapsuleApplyStatusMessage: Codable, Equatable, Sendable {
    public static let operation = HaloCapsuleTransferOps.applyStatus

    public let op: String
    public let payload: HaloCapsuleApplyStatusPayload

    public init(payload: HaloCapsuleApplyStatusPayload) {
        self.op = Self.operation
        self.payload = payload
    }
}

public enum HaloCapsuleTransferFrame: Equatable, Sendable {
    case offer(HaloCapsuleMeshMessage)
    case offerAck(HaloCapsuleTransferAck)
    case chunkHeader(HaloCapsuleChunkHeader)
    case binary(Data)
    case complete(HaloCapsuleTransferComplete)
    case completeAck(HaloCapsuleTransferAck)
    case applyStatus(HaloCapsuleApplyStatusPayload)
}

public enum HaloCapsulePackageTransfer {
    public static let defaultChunkSize = 1 * 1024 * 1024

    public static func sendPackage(
        message: HaloCapsuleMeshMessage,
        packageDirectory: URL,
        chunkSize: Int = defaultChunkSize,
        sendFrame: (Data) throws -> Void
    ) throws {
        try message.validate()
        guard chunkSize > 0 else {
            throw HaloCapsuleTransportError.invalidByteCount(
                field: "chunk_size",
                value: chunkSize
            )
        }
        try validatePackageFiles(message: message, packageDirectory: packageDirectory)
        try sendFrame(makeOfferFrame(message))

        for file in message.capsule.artifact.files {
            let fileURL = try artifactFileURL(packageDirectory: packageDirectory, name: file.name)
            let handle = try FileHandle(forReadingFrom: fileURL)
            defer { try? handle.close() }

            var offset = 0
            while offset < file.byteCount {
                let readCount = min(chunkSize, file.byteCount - offset)
                let chunk = try handle.read(upToCount: readCount) ?? Data()
                guard !chunk.isEmpty else {
                    throw HaloCapsuleTransportError.packageIncomplete(
                        name: file.name,
                        expected: file.byteCount,
                        actual: offset
                    )
                }
                let header = HaloCapsuleChunkHeader(
                    transferID: message.transferID,
                    fileName: file.name,
                    offset: offset,
                    byteCount: chunk.count,
                    sha256: sha256Hex(chunk)
                )
                try sendFrame(makeChunkHeaderFrame(header))
                try sendFrame(chunk)
                offset += chunk.count
            }
        }

        try sendFrame(makeCompleteFrame(transferID: message.transferID))
    }

    public static func packageFrames(
        message: HaloCapsuleMeshMessage,
        packageDirectory: URL,
        chunkSize: Int = defaultChunkSize
    ) throws -> [Data] {
        var frames: [Data] = []
        try sendPackage(
            message: message,
            packageDirectory: packageDirectory,
            chunkSize: chunkSize
        ) { frame in
            frames.append(frame)
        }
        return frames
    }

    public static func validatePackageFiles(
        message: HaloCapsuleMeshMessage,
        packageDirectory: URL
    ) throws {
        try message.validate()
        var actualBytes = 0
        for file in message.capsule.artifact.files {
            let fileURL = try artifactFileURL(packageDirectory: packageDirectory, name: file.name)
            let resource = try fileURL.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard resource.isRegularFile == true else {
                throw HaloCapsuleTransportError.artifactFileReadFailed(
                    name: file.name,
                    reason: "not a regular file"
                )
            }
            let byteCount = resource.fileSize ?? 0
            guard byteCount == file.byteCount else {
                throw HaloCapsuleTransportError.invalidByteCount(
                    field: "artifact.files[\(file.name)].byte_count",
                    value: byteCount
                )
            }
            let actualHash = try sha256Hex(forFileAt: fileURL)
            guard actualHash == file.sha256 else {
                throw HaloCapsuleTransportError.fileHashMismatch(
                    name: file.name,
                    expected: file.sha256,
                    actual: actualHash
                )
            }
            actualBytes += byteCount
        }
        guard actualBytes == message.capsule.artifact.totalBytes else {
            throw HaloCapsuleTransportError.invalidArtifactByteCount(
                expected: message.capsule.artifact.totalBytes,
                actual: actualBytes
            )
        }
    }

    public static func makeOfferFrame(_ message: HaloCapsuleMeshMessage) throws -> Data {
        try JSONEncoder().encode(TransferEnvelope(op: HaloCapsuleTransferOps.offer, payload: message))
    }

    public static func makeOfferAckFrame(_ ack: HaloCapsuleTransferAck) throws -> Data {
        try JSONEncoder().encode(TransferEnvelope(op: HaloCapsuleTransferOps.offerAck, payload: ack))
    }

    public static func makeChunkHeaderFrame(_ header: HaloCapsuleChunkHeader) throws -> Data {
        try JSONEncoder().encode(TransferEnvelope(op: HaloCapsuleTransferOps.chunk, payload: header))
    }

    public static func makeCompleteFrame(transferID: String) throws -> Data {
        try JSONEncoder().encode(TransferEnvelope(
            op: HaloCapsuleTransferOps.complete,
            payload: HaloCapsuleTransferComplete(transferID: transferID)
        ))
    }

    public static func makeCompleteAckFrame(_ ack: HaloCapsuleTransferAck) throws -> Data {
        try JSONEncoder().encode(TransferEnvelope(op: HaloCapsuleTransferOps.completeAck, payload: ack))
    }

    public static func makeApplyStatusFrame(_ status: HaloCapsuleApplyStatusPayload) throws -> Data {
        try JSONEncoder().encode(HaloCapsuleApplyStatusMessage(payload: status))
    }

    public static func decodeFrame(_ data: Data) throws -> HaloCapsuleTransferFrame {
        guard let probe = try? JSONDecoder().decode(TransferEnvelopeProbe.self, from: data) else {
            return .binary(data)
        }
        switch probe.op {
        case HaloCapsuleTransferOps.offer:
            return .offer(try JSONDecoder().decode(
                TransferEnvelope<HaloCapsuleMeshMessage>.self,
                from: data
            ).payload)
        case HaloCapsuleTransferOps.offerAck:
            return .offerAck(try JSONDecoder().decode(
                TransferEnvelope<HaloCapsuleTransferAck>.self,
                from: data
            ).payload)
        case HaloCapsuleTransferOps.chunk:
            return .chunkHeader(try JSONDecoder().decode(
                TransferEnvelope<HaloCapsuleChunkHeader>.self,
                from: data
            ).payload)
        case HaloCapsuleTransferOps.complete:
            return .complete(try JSONDecoder().decode(
                TransferEnvelope<HaloCapsuleTransferComplete>.self,
                from: data
            ).payload)
        case HaloCapsuleTransferOps.completeAck:
            return .completeAck(try JSONDecoder().decode(
                TransferEnvelope<HaloCapsuleTransferAck>.self,
                from: data
            ).payload)
        case HaloCapsuleTransferOps.applyStatus:
            return .applyStatus(try JSONDecoder().decode(
                HaloCapsuleApplyStatusMessage.self,
                from: data
            ).payload)
        default:
            throw HaloCapsuleTransportError.unsupportedMessageKind(probe.op)
        }
    }

    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func sha256Hex(forFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1024 * 1024) ?? Data()
            guard !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func artifactFileURL(packageDirectory: URL, name: String) throws -> URL {
        guard isSafeArtifactFileName(name) else {
            throw HaloCapsuleTransportError.unsafeArtifactFileName(name)
        }
        return packageDirectory.appendingPathComponent(name, isDirectory: false)
    }

    private static func isSafeArtifactFileName(_ name: String) -> Bool {
        guard !name.isEmpty, name != ".", name != ".." else { return false }
        guard !name.contains("/"), !name.contains("\\") else { return false }
        return true
    }
}

public final class HaloCapsulePackageReceiver: @unchecked Sendable {
    public struct Configuration: Sendable {
        public var destinationDirectory: URL
        public var expectedBaseModelID: String?
        public var expectedModelFamily: String?
        public var expectedHiddenSize: Int?
        public var expectedLayerCount: Int?
        public var expectedToolSchemaSHA256: String?
        public var currentRuntimeVersion: String?

        public init(
            destinationDirectory: URL,
            expectedBaseModelID: String? = nil,
            expectedModelFamily: String? = nil,
            expectedHiddenSize: Int? = nil,
            expectedLayerCount: Int? = nil,
            expectedToolSchemaSHA256: String? = nil,
            currentRuntimeVersion: String? = nil
        ) {
            self.destinationDirectory = destinationDirectory
            self.expectedBaseModelID = expectedBaseModelID
            self.expectedModelFamily = expectedModelFamily
            self.expectedHiddenSize = expectedHiddenSize
            self.expectedLayerCount = expectedLayerCount
            self.expectedToolSchemaSHA256 = expectedToolSchemaSHA256
            self.currentRuntimeVersion = currentRuntimeVersion
        }
    }

    public enum Event: Equatable, Sendable {
        case offerAccepted(HaloCapsuleTransferAck)
        case offerRejected(HaloCapsuleTransferAck)
        case chunkHeaderAccepted(HaloCapsuleChunkHeader)
        case chunkWritten(fileName: String, offset: Int, byteCount: Int)
        case completed(HaloCapsuleMeshMessage, HaloCapsuleTransferAck)
    }

    private let configuration: Configuration
    private var message: HaloCapsuleMeshMessage?
    private var pendingChunk: HaloCapsuleChunkHeader?
    private var receivedBytesByFile: [String: Int] = [:]

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func receive(_ frame: Data) throws -> Event {
        switch try HaloCapsulePackageTransfer.decodeFrame(frame) {
        case .offer(let offer):
            return try receiveOffer(offer)
        case .chunkHeader(let header):
            return try receiveChunkHeader(header)
        case .binary(let data):
            return try receiveBinaryChunk(data)
        case .complete(let complete):
            return try receiveComplete(complete)
        case .offerAck, .completeAck, .applyStatus:
            throw HaloCapsuleTransportError.unexpectedTransferFrame
        }
    }

    public func offerAckFrame(_ ack: HaloCapsuleTransferAck) throws -> Data {
        try HaloCapsulePackageTransfer.makeOfferAckFrame(ack)
    }

    public func completeAckFrame(_ ack: HaloCapsuleTransferAck) throws -> Data {
        try HaloCapsulePackageTransfer.makeCompleteAckFrame(ack)
    }

    private func receiveOffer(_ offer: HaloCapsuleMeshMessage) throws -> Event {
        do {
            try offer.validate(
                expectedBaseModelID: configuration.expectedBaseModelID,
                expectedModelFamily: configuration.expectedModelFamily,
                expectedHiddenSize: configuration.expectedHiddenSize,
                expectedLayerCount: configuration.expectedLayerCount,
                expectedToolSchemaSHA256: configuration.expectedToolSchemaSHA256,
                currentRuntimeVersion: configuration.currentRuntimeVersion
            )
            try prepareDestination(for: offer)
            message = offer
            pendingChunk = nil
            receivedBytesByFile = [:]
            for file in offer.capsule.artifact.files {
                receivedBytesByFile[file.name] = 0
            }
            let ack = HaloCapsuleTransferAck(
                transferID: offer.transferID,
                accepted: true,
                canonicalSHA256: try offer.canonicalSHA256()
            )
            return .offerAccepted(ack)
        } catch {
            message = nil
            pendingChunk = nil
            receivedBytesByFile = [:]
            let ack = HaloCapsuleTransferAck(
                transferID: offer.transferID,
                accepted: false,
                reason: String(describing: error)
            )
            return .offerRejected(ack)
        }
    }

    private func receiveChunkHeader(_ header: HaloCapsuleChunkHeader) throws -> Event {
        let offer = try requireOffer()
        try requireTransferID(header.transferID, expected: offer.transferID)
        guard offer.capsule.artifact.files.contains(where: { $0.name == header.fileName }) else {
            throw HaloCapsuleTransportError.unknownArtifactFile(header.fileName)
        }
        guard header.byteCount > 0 else {
            throw HaloCapsuleTransportError.invalidByteCount(
                field: "chunk.byte_count",
                value: header.byteCount
            )
        }
        try requireSHA256(header.sha256, field: "chunk.sha256")
        let expectedOffset = receivedBytesByFile[header.fileName] ?? 0
        guard header.offset == expectedOffset else {
            throw HaloCapsuleTransportError.invalidChunkOrder(
                name: header.fileName,
                expectedOffset: expectedOffset,
                actualOffset: header.offset
            )
        }
        pendingChunk = header
        return .chunkHeaderAccepted(header)
    }

    private func receiveBinaryChunk(_ data: Data) throws -> Event {
        guard let header = pendingChunk else {
            throw HaloCapsuleTransportError.unexpectedBinaryFrame
        }
        guard data.count == header.byteCount else {
            throw HaloCapsuleTransportError.invalidByteCount(
                field: "chunk.data.count",
                value: data.count
            )
        }
        let actualHash = HaloCapsulePackageTransfer.sha256Hex(data)
        guard actualHash == header.sha256 else {
            throw HaloCapsuleTransportError.chunkHashMismatch(
                expected: header.sha256,
                actual: actualHash
            )
        }

        let fileURL = try HaloCapsulePackageTransfer.artifactFileURL(
            packageDirectory: configuration.destinationDirectory,
            name: header.fileName
        )
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(header.offset))
        try handle.write(contentsOf: data)

        pendingChunk = nil
        receivedBytesByFile[header.fileName] = header.offset + data.count
        return .chunkWritten(
            fileName: header.fileName,
            offset: header.offset,
            byteCount: data.count
        )
    }

    private func receiveComplete(_ complete: HaloCapsuleTransferComplete) throws -> Event {
        let offer = try requireOffer()
        try requireTransferID(complete.transferID, expected: offer.transferID)
        guard pendingChunk == nil else {
            throw HaloCapsuleTransportError.incompleteChunk(
                name: pendingChunk?.fileName ?? ""
            )
        }
        try HaloCapsulePackageTransfer.validatePackageFiles(
            message: offer,
            packageDirectory: configuration.destinationDirectory
        )
        let ack = HaloCapsuleTransferAck(
            transferID: offer.transferID,
            accepted: true,
            canonicalSHA256: try offer.canonicalSHA256()
        )
        return .completed(offer, ack)
    }

    private func prepareDestination(for offer: HaloCapsuleMeshMessage) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: configuration.destinationDirectory,
            withIntermediateDirectories: true
        )
        for file in offer.capsule.artifact.files {
            let url = try HaloCapsulePackageTransfer.artifactFileURL(
                packageDirectory: configuration.destinationDirectory,
                name: file.name
            )
            if fm.fileExists(atPath: url.path) {
                try fm.removeItem(at: url)
            }
        }
    }

    private func requireOffer() throws -> HaloCapsuleMeshMessage {
        guard let message else {
            throw HaloCapsuleTransportError.missingTransferOffer
        }
        return message
    }

    private func requireTransferID(_ actual: String, expected: String) throws {
        guard actual == expected else {
            throw HaloCapsuleTransportError.transferIDMismatch(
                expected: expected,
                actual: actual
            )
        }
    }
}

private struct TransferEnvelope<Payload: Codable>: Codable {
    var op: String
    var payload: Payload
}

private struct TransferEnvelopeProbe: Codable {
    var op: String
}

private func requireSHA256(_ value: String, field: String) throws {
    guard !value.isEmpty else {
        throw HaloCapsuleTransportError.emptySHA256(field: field)
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendHaloCapsuleApplyStatus(_ status: HaloCapsuleApplyStatusPayload) throws {
        try sendJSON(HaloCapsuleApplyStatusMessage(payload: status))
    }
}
