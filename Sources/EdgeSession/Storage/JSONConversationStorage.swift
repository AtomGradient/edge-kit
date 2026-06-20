// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum JSONConversationStorageError: Error, Equatable {
    case unsupportedSchemaVersion(Int)
    case missingDocumentsDirectory
}

/// JSON file storage backend for conversations.
///
/// Persistence contract:
/// - Directory layout: `{baseURL}/conversations/{uuid}/metadata.json` and
///   `{baseURL}/conversations/{uuid}/messages.json`
/// - Incremental appends are staged in `{baseURL}/conversations/{uuid}/messages.jsonl`.
///   `saveMessages` remains the full replacement path and clears the append log.
/// - `baseURL` is injectable; the default is the app documents directory.
/// - Dates are encoded as ISO 8601 strings with fractional seconds.
/// - Writes are staged to a same-directory `.tmp` file and then replaced.
/// - JSON files use a top-level `schemaVersion` value.
public actor JSONConversationStorage: ConversationStorageBackend {
    public static let schemaVersion = 1

    public let baseURL: URL

    private let fileManager: FileManager

    public init(baseURL: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        if let baseURL {
            self.baseURL = baseURL
        } else if let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            self.baseURL = documents
        } else {
            self.baseURL = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            NSLog("[JSONConversationStorage] Documents directory unavailable; using temporary storage")
        }
    }

    public func loadConversationList() async throws -> [EdgeConversation] {
        try ensureBaseDirectory()
        guard fileManager.fileExists(atPath: conversationsDirectory.path) else {
            return []
        }

        let directories = try fileManager.contentsOfDirectory(
            at: conversationsDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        var conversations: [EdgeConversation] = []
        for directory in directories {
            let values = try directory.resourceValues(forKeys: [.isDirectoryKey])
            guard values.isDirectory == true else { continue }
            let metadataURL = directory.appendingPathComponent("metadata.json")
            guard fileManager.fileExists(atPath: metadataURL.path) else { continue }
            guard let conversation = try? loadMetadata(from: metadataURL) else {
                continue
            }
            conversations.append(conversation)
        }

        return conversations.sorted { $0.updatedAt > $1.updatedAt }
    }

    public func loadMessages(conversationID: UUID) async throws -> [EdgeConversationMessage] {
        let url = messagesURL(for: conversationID)
        var messages: [EdgeConversationMessage] = []
        if fileManager.fileExists(atPath: url.path) {
            messages = try loadSnapshotMessages(from: url)
        }

        let logURL = messagesLogURL(for: conversationID)
        if fileManager.fileExists(atPath: logURL.path) {
            messages.append(contentsOf: try loadLoggedMessages(from: logURL))
        }
        return messages
    }

    public func saveMessages(
        _ messages: [EdgeConversationMessage],
        for conversationID: UUID
    ) async throws {
        try ensureConversationDirectory(for: conversationID)
        let envelope = MessagesEnvelope(
            schemaVersion: Self.schemaVersion,
            messages: messages
        )
        try writeAtomically(try encoder().encode(envelope), to: messagesURL(for: conversationID))
        try removeMessagesLog(for: conversationID)
    }

    public func appendMessages(
        _ messages: [EdgeConversationMessage],
        for conversationID: UUID
    ) async throws {
        guard !messages.isEmpty else { return }
        try ensureConversationDirectory(for: conversationID)
        let url = messagesLogURL(for: conversationID)
        if !fileManager.fileExists(atPath: url.path) {
            fileManager.createFile(atPath: url.path, contents: nil)
        }

        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        for message in messages {
            let envelope = MessageLogEnvelope(
                schemaVersion: Self.schemaVersion,
                message: message
            )
            var data = try lineEncoder().encode(envelope)
            data.append(0x0A)
            handle.write(data)
        }
    }

    public func saveConversationMetadata(_ conversation: EdgeConversation) async throws {
        try saveMetadata(conversation)
    }

    public func deleteConversation(_ id: UUID) async throws {
        let directory = conversationDirectory(for: id)
        guard fileManager.fileExists(atPath: directory.path) else {
            return
        }
        try fileManager.removeItem(at: directory)
    }

    public func createConversation(
        title: String,
        modelCategory: String
    ) async throws -> EdgeConversation {
        let conversation = EdgeConversation(
            title: title,
            modelCategory: modelCategory
        )
        try saveMetadata(conversation)
        return conversation
    }

    private func loadSnapshotMessages(from url: URL) throws -> [EdgeConversationMessage] {
        let data = try Data(contentsOf: url)
        do {
            let envelope = try decoder().decode(
                MessagesEnvelope.self,
                from: data
            )
            try validateSchemaVersion(envelope.schemaVersion)
            return envelope.messages
        } catch let error as JSONConversationStorageError {
            throw error
        } catch {
            return try decoder().decode([LegacyMessage].self, from: data)
                .map(\.edgeMessage)
        }
    }

    private func loadLoggedMessages(from url: URL) throws -> [EdgeConversationMessage] {
        let data = try Data(contentsOf: url)
        guard let content = String(data: data, encoding: .utf8) else {
            return []
        }
        return try content
            .split(separator: "\n")
            .map { line in
                let data = Data(line.utf8)
                let envelope = try decoder().decode(MessageLogEnvelope.self, from: data)
                try validateSchemaVersion(envelope.schemaVersion)
                return envelope.message
            }
    }

    private var conversationsDirectory: URL {
        baseURL.appendingPathComponent("conversations", isDirectory: true)
    }

    private func conversationDirectory(for id: UUID) -> URL {
        conversationsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func metadataURL(for id: UUID) -> URL {
        conversationDirectory(for: id).appendingPathComponent("metadata.json")
    }

    private func messagesURL(for id: UUID) -> URL {
        conversationDirectory(for: id).appendingPathComponent("messages.json")
    }

    private func messagesLogURL(for id: UUID) -> URL {
        conversationDirectory(for: id).appendingPathComponent("messages.jsonl")
    }

    private func ensureBaseDirectory() throws {
        try fileManager.createDirectory(
            at: conversationsDirectory,
            withIntermediateDirectories: true
        )
    }

    private func ensureConversationDirectory(for id: UUID) throws {
        try ensureBaseDirectory()
        try fileManager.createDirectory(
            at: conversationDirectory(for: id),
            withIntermediateDirectories: true
        )
    }

    private func saveMetadata(_ conversation: EdgeConversation) throws {
        try ensureConversationDirectory(for: conversation.id)
        let envelope = MetadataEnvelope(
            schemaVersion: Self.schemaVersion,
            conversation: conversation
        )
        try writeAtomically(try encoder().encode(envelope), to: metadataURL(for: conversation.id))
    }

    private func removeMessagesLog(for id: UUID) throws {
        let url = messagesLogURL(for: id)
        if fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
    }

    private func loadMetadata(from url: URL) throws -> EdgeConversation {
        let data = try Data(contentsOf: url)
        do {
            let envelope = try decoder().decode(
                MetadataEnvelope.self,
                from: data
            )
            try validateSchemaVersion(envelope.schemaVersion)
            return envelope.conversation
        } catch let error as JSONConversationStorageError {
            throw error
        } catch {
            return try decoder().decode(EdgeConversation.self, from: data)
        }
    }

    private func validateSchemaVersion(_ version: Int) throws {
        guard version == Self.schemaVersion else {
            throw JSONConversationStorageError.unsupportedSchemaVersion(version)
        }
    }

    private func writeAtomically(_ data: Data, to destination: URL) throws {
        let directory = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(
            ".\(destination.lastPathComponent).\(UUID().uuidString).tmp"
        )

        try data.write(to: temporary)
        do {
            if fileManager.fileExists(atPath: destination.path) {
                _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
            } else {
                try fileManager.moveItem(at: temporary, to: destination)
            }
        } catch {
            if fileManager.fileExists(atPath: temporary.path) {
                try? fileManager.removeItem(at: temporary)
            }
            throw error
        }
    }

    private func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(Self.iso8601String(from: date))
        }
        return encoder
    }

    private func lineEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(Self.iso8601String(from: date))
        }
        return encoder
    }

    private func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(String.self)
            if let date = Self.iso8601Date(from: value) {
                return date
            }
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Invalid ISO 8601 date: \(value)"
            )
        }
        return decoder
    }

    private static func iso8601String(from date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.string(from: date)
    }

    private static func iso8601Date(from string: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: string) {
            return date
        }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return plain.date(from: string)
    }

    private struct MetadataEnvelope: Codable {
        let schemaVersion: Int
        let conversation: EdgeConversation
    }

    private struct MessagesEnvelope: Codable {
        let schemaVersion: Int
        let messages: [EdgeConversationMessage]
    }

    private struct MessageLogEnvelope: Codable {
        let schemaVersion: Int
        let message: EdgeConversationMessage
    }

    private struct LegacyMessage: Codable {
        let id: UUID
        let role: EdgeConversationRole
        let content: String
        let timestamp: Date
        let imageData: Data?
        let feedbackRating: String?

        var edgeMessage: EdgeConversationMessage {
            var metadata: [String: String] = [:]
            if let imageData {
                metadata["imageDataBase64"] = imageData.base64EncodedString()
            }
            if let feedbackRating {
                metadata["feedbackRating"] = feedbackRating
            }
            return EdgeConversationMessage(
                id: id,
                role: role,
                content: content,
                timestamp: timestamp,
                metadata: metadata
            )
        }
    }
}
