// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
@testable import EdgeSession
import XCTest

final class ModelsTests: XCTestCase {
    func test_edgeConversationRoleMapsChatMessageRolesRoundTrip() {
        let roles: [ChatMessage.Role] = [.system, .user, .assistant, .tool]

        for role in roles {
            let persisted = EdgeConversationRole(from: role)
            XCTAssertEqual(persisted.chatRole, role)
        }
    }

    func test_edgeConversationMessageBuildsChatMessage() {
        let message = EdgeConversationMessage(
            chatRole: .assistant,
            content: "hello",
            metadata: ["source": "test"]
        )

        XCTAssertEqual(message.role, .assistant)
        XCTAssertEqual(message.chatMessage.role, .assistant)
        XCTAssertEqual(message.chatMessage.content, "hello")
        XCTAssertEqual(message.metadata["source"], "test")
    }

    func test_edgeConversationRoleCodableUsesStableStrings() throws {
        let data = try JSONEncoder().encode(EdgeConversationRole.tool)
        XCTAssertEqual(String(data: data, encoding: .utf8), "\"tool\"")

        let decoded = try JSONDecoder().decode(EdgeConversationRole.self, from: data)
        XCTAssertEqual(decoded, .tool)
    }
}

