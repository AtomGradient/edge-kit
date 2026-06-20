// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import EdgeInference
import Foundation

/// SwiftUI-friendly conversation store backed by a pluggable storage backend.
@MainActor
public final class ConversationStore: ObservableObject {
    public static let shared = ConversationStore()

    @Published public private(set) var conversations: [EdgeConversation] = []

    private let backend: any ConversationStorageBackend
    private var savedMessageSnapshots: [UUID: [EdgeConversationMessage]] = [:]

    public init(
        backend: any ConversationStorageBackend = JSONConversationStorage(),
        autoload: Bool = true
    ) {
        self.backend = backend
        if autoload {
            Task { [weak self] in
                try? await self?.loadConversations()
            }
        }
    }

    public func loadConversations() async throws {
        conversations = try await backend.loadConversationList()
    }

    @discardableResult
    public func createConversation(
        title: String,
        modelCategory: String
    ) async throws -> EdgeConversation {
        let conversation = try await backend.createConversation(
            title: title,
            modelCategory: modelCategory
        )
        conversations.insert(conversation, at: 0)
        conversations.sort { $0.updatedAt > $1.updatedAt }
        savedMessageSnapshots[conversation.id] = []
        return conversation
    }

    public func deleteConversation(_ id: UUID) async throws {
        try await backend.deleteConversation(id)
        conversations.removeAll { $0.id == id }
        savedMessageSnapshots[id] = nil
    }

    public func loadMessages(for conversationID: UUID) async throws -> [EdgeConversationMessage] {
        let messages = try await backend.loadMessages(conversationID: conversationID)
        savedMessageSnapshots[conversationID] = messages
        return messages
    }

    public func saveMessages(
        _ messages: [EdgeConversationMessage],
        for conversationID: UUID
    ) async throws {
        if let previous = savedMessageSnapshots[conversationID],
           messages.count >= previous.count,
           Array(messages.prefix(previous.count)) == previous {
            let appended = Array(messages.dropFirst(previous.count))
            try await backend.appendMessages(appended, for: conversationID)
        } else {
            try await backend.saveMessages(messages, for: conversationID)
        }
        savedMessageSnapshots[conversationID] = messages

        guard let index = conversations.firstIndex(where: { $0.id == conversationID }) else {
            conversations = try await backend.loadConversationList()
            return
        }

        var conversation = conversations[index]
        conversation.updatedAt = Date()
        conversation.messageCount = messages.count

        if conversation.title.isEmpty || conversation.title == "New Chat" {
            if let firstUser = messages.first(where: { $0.role == .user }) {
                let title = String(firstUser.content.prefix(40))
                conversation.title = title.isEmpty ? "New Chat" : title
            }
        }

        try await backend.saveConversationMetadata(conversation)
        conversations[index] = conversation
        conversations.sort { $0.updatedAt > $1.updatedAt }
    }

    public func saveConversationMetadata(_ conversation: EdgeConversation) async throws {
        try await backend.saveConversationMetadata(conversation)
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        } else {
            conversations.append(conversation)
        }
        conversations.sort { $0.updatedAt > $1.updatedAt }
    }
}
