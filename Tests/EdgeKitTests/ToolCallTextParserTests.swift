// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

@testable import EdgeInference
import XCTest

final class ToolCallTextParserTests: XCTestCase {
    func testParsesQwenXMLToolCall() {
        let text = """
        <tool_call>
        <function=query_expenses>
        <parameter=timeRange>
        month
        </parameter>
        <parameter=offset>
        -1
        </parameter>
        <parameter=limit>
        0
        </parameter>
        </function>
        </tool_call>
        """

        let calls = ToolCallTextParser.toolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].function.name, "query_expenses")
        XCTAssertEqual(calls[0].function.arguments["timeRange"] as? String, "month")
        XCTAssertEqual(calls[0].function.arguments["offset"] as? Int, -1)
        XCTAssertEqual(calls[0].function.arguments["limit"] as? Int, 0)
    }

    func testParsesJSONToolCall() {
        let text = """
        <tool_call>
        {"function":{"name":"query_user_profile","arguments":{"detailLevel":"summary"}}}
        </tool_call>
        """

        let calls = ToolCallTextParser.toolCalls(in: text)

        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].function.name, "query_user_profile")
        XCTAssertEqual(calls[0].function.arguments["detailLevel"] as? String, "summary")
    }

    func testIgnoresIncompleteToolCall() {
        let calls = ToolCallTextParser.toolCalls(in: "<tool_call><function=query_expenses>")

        XCTAssertTrue(calls.isEmpty)
    }
}
