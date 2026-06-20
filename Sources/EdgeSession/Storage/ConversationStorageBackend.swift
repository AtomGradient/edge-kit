// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Storage backend abstraction for conversation metadata and messages.
///
/// Phase 0 ships a JSON backend. Future backends, such as GRDB, can conform to
/// this protocol without changing the public `ConversationStore` surface.
public protocol ConversationStorageBackend: Sendable {
    func loadConversationList() async throws -> [EdgeConversation]
    func loadMessages(conversationID: UUID) async throws -> [EdgeConversationMessage]
    func saveMessages(_ messages: [EdgeConversationMessage], for conversationID: UUID) async throws
    func appendMessages(_ messages: [EdgeConversationMessage], for conversationID: UUID) async throws
    func saveConversationMetadata(_ conversation: EdgeConversation) async throws
    func deleteConversation(_ id: UUID) async throws
    func createConversation(title: String, modelCategory: String) async throws -> EdgeConversation
}

public extension ConversationStorageBackend {
    func appendMessages(_ messages: [EdgeConversationMessage], for conversationID: UUID) async throws {
        guard !messages.isEmpty else { return }
        let existing = try await loadMessages(conversationID: conversationID)
        try await saveMessages(existing + messages, for: conversationID)
    }
}
