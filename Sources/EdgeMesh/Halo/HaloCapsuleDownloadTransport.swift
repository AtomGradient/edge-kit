// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct HaloCapsuleDownloadedFile: Sendable {
    public let temporaryURL: URL
    public let httpStatusCode: Int?

    public init(temporaryURL: URL, httpStatusCode: Int? = nil) {
        self.temporaryURL = temporaryURL
        self.httpStatusCode = httpStatusCode
    }
}

public struct HaloCapsuleDownloadTransport: Sendable {
    public typealias FileDownloader = @Sendable (URL) async throws -> HaloCapsuleDownloadedFile

    private let fileDownloader: FileDownloader

    public init() {
        self.fileDownloader = Self.defaultFileDownloader
    }

    public init(fileDownloader: @escaping FileDownloader) {
        self.fileDownloader = fileDownloader
    }

    public static func usesDownloadTransport(_ message: HaloCapsuleMeshMessage) -> Bool {
        !message.capsule.artifact.files.isEmpty
            && message.capsule.artifact.files.allSatisfy { $0.downloadURL != nil }
    }

    public func downloadPackage(
        message: HaloCapsuleMeshMessage,
        to destination: URL
    ) async throws -> HaloCapsuleInboundTransferSession.CompletedPackage {
        try await downloadFiles(message: message, to: destination)
        try HaloCapsulePackageTransfer.validatePackageFiles(
            message: message,
            packageDirectory: destination
        )
        let ack = HaloCapsuleTransferAck(
            transferID: message.transferID,
            accepted: true,
            canonicalSHA256: try message.canonicalSHA256()
        )
        return HaloCapsuleInboundTransferSession.CompletedPackage(
            message: message,
            packageDirectory: destination,
            completeAck: ack
        )
    }

    public func downloadFiles(
        message: HaloCapsuleMeshMessage,
        to destination: URL
    ) async throws {
        try message.validate()
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        for file in message.capsule.artifact.files {
            guard let downloadURL = file.downloadURL else {
                throw HaloCapsuleTransportError.missingDownloadURL(name: file.name)
            }

            let downloaded = try await fileDownloader(downloadURL)
            if let statusCode = downloaded.httpStatusCode,
               !(200..<300).contains(statusCode) {
                throw HaloCapsuleTransportError.downloadHTTPStatus(
                    name: file.name,
                    statusCode: statusCode
                )
            }

            let destinationURL = try HaloCapsulePackageTransfer.artifactFileURL(
                packageDirectory: destination,
                name: file.name
            )
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.moveItem(at: downloaded.temporaryURL, to: destinationURL)

            let resource = try destinationURL.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = resource.fileSize ?? 0
            guard byteCount == file.byteCount else {
                throw HaloCapsuleTransportError.invalidByteCount(
                    field: "artifact.files[\(file.name)].byte_count",
                    value: byteCount
                )
            }
            let actualHash = try HaloCapsulePackageTransfer.sha256Hex(forFileAt: destinationURL)
            guard actualHash == file.sha256 else {
                throw HaloCapsuleTransportError.fileHashMismatch(
                    name: file.name,
                    expected: file.sha256,
                    actual: actualHash
                )
            }
        }
    }

    private static let defaultFileDownloader: FileDownloader = { url in
        let (temporaryURL, response) = try await URLSession.shared.download(from: url)
        return HaloCapsuleDownloadedFile(
            temporaryURL: temporaryURL,
            httpStatusCode: (response as? HTTPURLResponse)?.statusCode
        )
    }
}
