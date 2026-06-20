// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

/// SDK-owned persisted role enum for conversation storage.
///
/// This intentionally stays separate from `ChatMessage.Role` so the storage
/// schema is not tied to the inference message implementation.
public enum EdgeConversationRole: String, Codable, Sendable, Equatable {
    case system
    case user
    case assistant
    case tool

    public init(from chatRole: ChatMessage.Role) {
        switch chatRole {
        case .system:
            self = .system
        case .user:
            self = .user
        case .assistant:
            self = .assistant
        case .tool:
            self = .tool
        }
    }

    public var chatRole: ChatMessage.Role {
        switch self {
        case .system:
            return .system
        case .user:
            return .user
        case .assistant:
            return .assistant
        case .tool:
            return .tool
        }
    }
}

/// SDK-owned conversation message model without UI-specific payloads.
public struct EdgeConversationMessage: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let role: EdgeConversationRole
    public let content: String
    public let timestamp: Date
    public let metadata: [String: String]

    public init(
        id: UUID = UUID(),
        role: EdgeConversationRole,
        content: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.metadata = metadata
    }

    public init(
        id: UUID = UUID(),
        chatRole: ChatMessage.Role,
        content: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.init(
            id: id,
            role: EdgeConversationRole(from: chatRole),
            content: content,
            timestamp: timestamp,
            metadata: metadata
        )
    }

    public var chatMessage: ChatMessage {
        ChatMessage(role: role.chatRole, content: content)
    }
}

/// SDK-owned conversation metadata.
public struct EdgeConversation: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public var title: String
    public let modelCategory: String
    public let createdAt: Date
    public var updatedAt: Date
    public var messageCount: Int

    public init(
        id: UUID = UUID(),
        title: String,
        modelCategory: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        messageCount: Int = 0
    ) {
        self.id = id
        self.title = title
        self.modelCategory = modelCategory
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messageCount = messageCount
    }
}

