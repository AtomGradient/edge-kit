// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

@testable import EdgeData
import XCTest

final class MinimalPromptBuilderTests: XCTestCase {
    func testBuildMessagesWithoutToolsExplicitlyForbidsToolCalls() {
        let messages = MinimalPromptBuilder().buildMessages(
            rawFact: RawFact(
                namespace: "test",
                rawPayload: ["description": "receipt"],
                candidateSchemas: ["test.transaction"],
                sensitivity: .localOnly
            ),
            candidateSchemas: ["test.transaction"],
            toolNames: []
        )

        let system = messages.first?["content"] ?? ""
        XCTAssertTrue(system.contains("本轮没有启用任何工具"))
        XCTAssertTrue(system.contains("不要输出 <tool_call>"))
        XCTAssertTrue(system.contains("完整输出必须是一个 JSON object"))
    }

    func testBuildMessagesWithToolsKeepsToolGuidance() {
        let messages = MinimalPromptBuilder().buildMessages(
            rawFact: RawFact(
                namespace: "test",
                rawPayload: ["description": "receipt"],
                candidateSchemas: ["test.transaction"],
                sensitivity: .localOnly
            ),
            candidateSchemas: ["test.transaction"],
            toolNames: ["query_history_facts"]
        )

        let system = messages.first?["content"] ?? ""
        XCTAssertTrue(system.contains("`query_history_facts`"))
        XCTAssertFalse(system.contains("本轮没有启用任何工具"))
    }
}
