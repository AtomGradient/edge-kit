// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

@testable import EdgeSession
import XCTest

final class ConversationStorageTests: XCTestCase {
    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("EdgeSessionTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        if let temporaryDirectory,
           FileManager.default.fileExists(atPath: temporaryDirectory.path) {
            try FileManager.default.removeItem(at: temporaryDirectory)
        }
        temporaryDirectory = nil
    }

    func test_jsonStorageRoundTripCreateSaveLoadDelete() async throws {
        let storage = JSONConversationStorage(baseURL: temporaryDirectory)
        let conversation = try await storage.createConversation(
            title: "New Chat",
            modelCategory: "llm"
        )

        let messages = [
            EdgeConversationMessage(role: .user, content: "hello"),
            EdgeConversationMessage(role: .assistant, content: "world"),
        ]

        try await storage.saveMessages(messages, for: conversation.id)

        let loadedMessages = try await storage.loadMessages(conversationID: conversation.id)
        XCTAssertEqual(loadedMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(loadedMessages.map(\.content), ["hello", "world"])

        var updated = conversation
        updated.title = "Updated"
        updated.messageCount = loadedMessages.count
        try await storage.saveConversationMetadata(updated)

        let conversations = try await storage.loadConversationList()
        XCTAssertEqual(conversations.count, 1)
        XCTAssertEqual(conversations.first?.id, conversation.id)
        XCTAssertEqual(conversations.first?.title, "Updated")
        XCTAssertEqual(conversations.first?.messageCount, 2)

        try await storage.deleteConversation(conversation.id)
        let conversationsAfterDelete = try await storage.loadConversationList()
        let messagesAfterDelete = try await storage.loadMessages(conversationID: conversation.id)
        XCTAssertTrue(conversationsAfterDelete.isEmpty)
        XCTAssertTrue(messagesAfterDelete.isEmpty)
    }

    func test_jsonStorageAppendsMessagesWithoutRewritingSnapshot() async throws {
        let storage = JSONConversationStorage(baseURL: temporaryDirectory)
        let conversation = try await storage.createConversation(
            title: "Append Chat",
            modelCategory: "llm"
        )

        try await storage.appendMessages(
            [EdgeConversationMessage(role: .user, content: "first")],
            for: conversation.id
        )
        try await storage.appendMessages(
            [EdgeConversationMessage(role: .assistant, content: "second")],
            for: conversation.id
        )

        let loadedMessages = try await storage.loadMessages(conversationID: conversation.id)
        XCTAssertEqual(loadedMessages.map(\.content), ["first", "second"])

        let conversationDirectory = temporaryDirectory
            .appendingPathComponent("conversations", isDirectory: true)
            .appendingPathComponent(conversation.id.uuidString, isDirectory: true)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: conversationDirectory.appendingPathComponent("messages.jsonl").path
        ))

        try await storage.saveMessages(
            [EdgeConversationMessage(role: .user, content: "replacement")],
            for: conversation.id
        )

        let replacedMessages = try await storage.loadMessages(conversationID: conversation.id)
        XCTAssertEqual(replacedMessages.map(\.content), ["replacement"])
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: conversationDirectory.appendingPathComponent("messages.jsonl").path
        ))
    }

    func test_jsonStorageReadsLegacyUnversionedConversationFiles() async throws {
        let conversationID = UUID()
        let conversationDirectory = temporaryDirectory
            .appendingPathComponent("conversations", isDirectory: true)
            .appendingPathComponent(conversationID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: conversationDirectory,
            withIntermediateDirectories: true
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let conversation = EdgeConversation(
            id: conversationID,
            title: "Legacy Chat",
            modelCategory: "llm",
            messageCount: 2
        )
        try encoder.encode(conversation)
            .write(to: conversationDirectory.appendingPathComponent("metadata.json"))

        let imageData = Data([0x01, 0x02, 0x03])
        let messages = [
            LegacyDisplayMessage(
                id: UUID(),
                role: .user,
                content: "hello",
                timestamp: Date(),
                imageData: imageData,
                feedbackRating: nil
            ),
            LegacyDisplayMessage(
                id: UUID(),
                role: .assistant,
                content: "world",
                timestamp: Date(),
                imageData: nil,
                feedbackRating: "good"
            ),
        ]
        try encoder.encode(messages)
            .write(to: conversationDirectory.appendingPathComponent("messages.json"))

        let storage = JSONConversationStorage(baseURL: temporaryDirectory)
        let loadedConversations = try await storage.loadConversationList()
        XCTAssertEqual(loadedConversations.map(\.id), [conversationID])
        XCTAssertEqual(loadedConversations.first?.title, "Legacy Chat")

        let loadedMessages = try await storage.loadMessages(conversationID: conversationID)
        XCTAssertEqual(loadedMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(loadedMessages.map(\.content), ["hello", "world"])
        XCTAssertEqual(loadedMessages[0].metadata["imageDataBase64"], imageData.base64EncodedString())
        XCTAssertEqual(loadedMessages[1].metadata["feedbackRating"], "good")
    }

    @MainActor
    func test_conversationStoreUpdatesMetadataOnSaveMessages() async throws {
        let storage = JSONConversationStorage(baseURL: temporaryDirectory)
        let store = ConversationStore(backend: storage)
        let conversation = try await store.createConversation(
            title: "New Chat",
            modelCategory: "llm"
        )

        try await store.saveMessages(
            [
                EdgeConversationMessage(role: .user, content: "make this the title"),
                EdgeConversationMessage(role: .assistant, content: "ok"),
            ],
            for: conversation.id
        )

        XCTAssertEqual(store.conversations.count, 1)
        XCTAssertEqual(store.conversations[0].messageCount, 2)
        XCTAssertEqual(store.conversations[0].title, "make this the title")
    }

    @MainActor
    func test_conversationStoreUsesAppendForUnchangedPrefix() async throws {
        let backend = SpyConversationBackend()
        let store = ConversationStore(backend: backend, autoload: false)
        let conversation = try await store.createConversation(
            title: "New Chat",
            modelCategory: "llm"
        )

        let first = EdgeConversationMessage(role: .user, content: "first")
        let second = EdgeConversationMessage(role: .assistant, content: "second")
        try await store.saveMessages([first], for: conversation.id)
        try await store.saveMessages([first, second], for: conversation.id)

        let appendSizes = await backend.appendCallSizes()
        let saveMessagesCallCount = await backend.saveMessagesCallCount()
        let messages = await backend.messages(for: conversation.id)
        XCTAssertEqual(appendSizes, [1, 1])
        XCTAssertEqual(saveMessagesCallCount, 0)
        XCTAssertEqual(messages.map(\.content), ["first", "second"])
    }

    @MainActor
    func test_conversationStoreFallsBackToReplaceWhenExistingMessageChanges() async throws {
        let backend = SpyConversationBackend()
        let store = ConversationStore(backend: backend, autoload: false)
        let conversation = try await store.createConversation(
            title: "New Chat",
            modelCategory: "llm"
        )

        let first = EdgeConversationMessage(role: .user, content: "first")
        try await store.saveMessages([first], for: conversation.id)

        let corrected = EdgeConversationMessage(
            id: first.id,
            role: .user,
            content: "first corrected",
            timestamp: first.timestamp
        )
        try await store.saveMessages([corrected], for: conversation.id)

        let appendSizes = await backend.appendCallSizes()
        let saveMessagesCallCount = await backend.saveMessagesCallCount()
        let messages = await backend.messages(for: conversation.id)
        XCTAssertEqual(appendSizes, [1])
        XCTAssertEqual(saveMessagesCallCount, 1)
        XCTAssertEqual(messages.map(\.content), ["first corrected"])
    }
}

private struct LegacyDisplayMessage: Codable {
    let id: UUID
    let role: EdgeConversationRole
    let content: String
    let timestamp: Date
    let imageData: Data?
    let feedbackRating: String?
}

private actor SpyConversationBackend: ConversationStorageBackend {
    private var conversations: [UUID: EdgeConversation] = [:]
    private var messageStore: [UUID: [EdgeConversationMessage]] = [:]
    private var appendSizes: [Int] = []
    private var replaceCount = 0

    func loadConversationList() async throws -> [EdgeConversation] {
        conversations.values.sorted { $0.updatedAt > $1.updatedAt }
    }

    func loadMessages(conversationID: UUID) async throws -> [EdgeConversationMessage] {
        messageStore[conversationID] ?? []
    }

    func saveMessages(_ messages: [EdgeConversationMessage], for conversationID: UUID) async throws {
        replaceCount += 1
        messageStore[conversationID] = messages
    }

    func appendMessages(_ messages: [EdgeConversationMessage], for conversationID: UUID) async throws {
        appendSizes.append(messages.count)
        messageStore[conversationID, default: []].append(contentsOf: messages)
    }

    func saveConversationMetadata(_ conversation: EdgeConversation) async throws {
        conversations[conversation.id] = conversation
    }

    func deleteConversation(_ id: UUID) async throws {
        conversations[id] = nil
        messageStore[id] = nil
    }

    func createConversation(title: String, modelCategory: String) async throws -> EdgeConversation {
        let conversation = EdgeConversation(title: title, modelCategory: modelCategory)
        conversations[conversation.id] = conversation
        messageStore[conversation.id] = []
        return conversation
    }

    func appendCallSizes() -> [Int] {
        appendSizes
    }

    func saveMessagesCallCount() -> Int {
        replaceCount
    }

    func messages(for conversationID: UUID) -> [EdgeConversationMessage] {
        messageStore[conversationID] ?? []
    }
}
