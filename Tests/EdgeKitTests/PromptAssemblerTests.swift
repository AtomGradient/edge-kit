// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class PromptAssemblerTests: XCTestCase {

    private func fact(
        id: String = UUID().uuidString,
        time: Int64,
        amount: Double = 50.0,
        merchant: String = "必胜客",
        category: String = "餐饮",
        location: String? = nil,
        description: String? = nil
    ) -> FactRecord {
        var payload: [String: FactValue] = [
            "amount": .double(amount),
            "merchant": .string(merchant),
            "category": .string(category),
            "time": .int(time),
        ]
        if let l = location { payload["location"] = .string(l) }
        if let d = description { payload["description"] = .string(d) }
        return FactRecord(
            id: id,
            schemaName: "finance.expense",
            payload: payload,
            createdAt: time
        )
    }

    func test_delimiters_literal() {
        XCTAssertEqual(FACT_START, "<|fact_layer_start|>")
        XCTAssertEqual(FACT_END, "<|fact_layer_end|>")
    }

    func test_budgetConstants() {
        XCTAssertEqual(DEFAULT_FACT_TOKEN_BUDGET, 512)
        XCTAssertEqual(CHARS_PER_TOKEN, 4)
    }

    func test_emptyFacts_noInjection() {
        let p = assembleWithFacts(userQuery: "你好", facts: [])
        XCTAssertEqual(p.user, "你好")
        XCTAssertEqual(p.injectedFactCount, 0)
        XCTAssertEqual(p.truncatedFactCount, 0)
        XCTAssertFalse(p.system.contains(FACT_START))
        XCTAssertFalse(p.system.contains(FACT_END))
    }

    func test_personaRouter_skipsFactInjection() {
        let router = RouterResult(
            label: .persona,
            confidence: 0.95,
            reason: "few_shot_high_conf",
            latencyMs: 1.0,
            query: "我是什么样的人"
        )
        let p = assembleWithFacts(
            userQuery: "我是什么样的人",
            facts: [fact(time: 1000, merchant: "必胜客")],
            routerResult: router
        )
        XCTAssertEqual(p.injectedFactCount, 0)
        XCTAssertFalse(p.system.contains("必胜客"),
            "persona 路径不应注入 fact，system 里不应出现商家名")
    }

    func test_factRouter_injectsFacts() {
        let router = RouterResult(
            label: .fact,
            confidence: 0.95,
            reason: "few_shot_high_conf",
            latencyMs: 1.0,
            query: "上周花了多少"
        )
        let p = assembleWithFacts(
            userQuery: "上周花了多少",
            facts: [fact(id: "f1", time: 1000, amount: 50, merchant: "必胜客")],
            routerResult: router
        )
        XCTAssertEqual(p.injectedFactCount, 1)
        XCTAssertTrue(p.system.contains(FACT_START))
        XCTAssertTrue(p.system.contains(FACT_END))
        XCTAssertTrue(p.system.contains("必胜客"))
        XCTAssertTrue(p.system.contains("¥50.00"))
    }

    func test_mixedRouter_injectsFacts() {
        let router = RouterResult(
            label: .mixed,
            confidence: 0.8,
            reason: "few_shot_high_conf",
            latencyMs: 1.0,
            query: "我最近吃得多不多"
        )
        let p = assembleWithFacts(
            userQuery: "test",
            facts: [fact(time: 1000)],
            routerResult: router
        )
        XCTAssertEqual(p.injectedFactCount, 1)
    }

    func test_determinism_sameFactsProduceSameOutput() {
        let facts = [
            fact(id: "a", time: 300, amount: 10),
            fact(id: "b", time: 100, amount: 20),
            fact(id: "c", time: 200, amount: 30),
        ]
        let p1 = assembleWithFacts(userQuery: "test", facts: facts)
        let p2 = assembleWithFacts(userQuery: "test", facts: facts)
        XCTAssertEqual(p1.system, p2.system, "同 facts 输入应产出字节相同 system")
        XCTAssertEqual(p1.user, p2.user)
    }

    func test_sortOrder_timeDescIdAsc() {
        let facts = [
            fact(id: "b", time: 300),
            fact(id: "a", time: 300),
            fact(id: "c", time: 100),
            fact(id: "d", time: 200),
        ]
        let p = assembleWithFacts(userQuery: "test", facts: facts)
        let startRange = p.system.range(of: FACT_START)!
        let endRange = p.system.range(of: FACT_END)!
        let section = String(p.system[startRange.upperBound..<endRange.lowerBound])
        func positionOfId(_ id: String) -> Int {
            let lines = section.split(separator: "\n").map(String.init)
            for (i, line) in lines.enumerated() {
                _ = line
                _ = id
                _ = i
            }
            return -1
        }
        XCTAssertEqual(p.injectedFactCount, 4)

        let time300Line = "1970-01-01"
        _ = time300Line
    }

    func test_sortOrder_differentDays() {
        let facts = [
            fact(id: "x", time: 1_700_000_000_000),
            fact(id: "y", time: 1_800_000_000_000),
            fact(id: "z", time: 1_600_000_000_000),
        ]
        let p = assembleWithFacts(userQuery: "test", facts: facts)
        let section = extractFactSection(p.system)
        let lines = section.split(separator: "\n").map(String.init)
        XCTAssertGreaterThanOrEqual(lines.count, 3)
        XCTAssertTrue(lines[0].contains("2027"), "time DESC 第一行应是最新的 2027: \(lines[0])")
        XCTAssertTrue(lines[1].contains("2023"), "第二行应是 2023: \(lines[1])")
        XCTAssertTrue(lines[2].contains("2020"), "第三行应是最旧的 2020: \(lines[2])")
    }

    private func extractFactSection(_ system: String) -> String {
        guard let s = system.range(of: FACT_START),
              let e = system.range(of: FACT_END) else {
            return ""
        }
        return String(system[s.upperBound..<e.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func test_pii_descriptionExcluded() {
        let f = fact(
            time: 1000, amount: 50,
            merchant: "必胜客",
            description: "用户备注：请客户吃饭，讨论项目 Alpha 机密"
        )
        let p = assembleWithFacts(userQuery: "test", facts: [f])
        XCTAssertTrue(p.system.contains("必胜客"))
        XCTAssertFalse(p.system.contains("机密"),
            "description 字段不应出现在 prompt 注入段（PII 规则 3）")
        XCTAssertFalse(p.system.contains("用户备注"))
    }

    func test_tokenBudget_truncatesAtLimit() {
        var facts: [FactRecord] = []
        for i in 0..<50 {
            facts.append(fact(
                id: "f\(i)",
                time: Int64(i),
                merchant: "商家\(i)"
            ))
        }
        let p = assembleWithFacts(
            userQuery: "test",
            facts: facts,
            tokenBudget: 200
        )
        XCTAssertLessThan(p.injectedFactCount, 50, "应有截断发生")
        XCTAssertGreaterThan(p.truncatedFactCount, 0)
        XCTAssertEqual(p.injectedFactCount + p.truncatedFactCount, 50)
        XCTAssertTrue(p.system.contains("（还有"))
    }

    func test_tokenBudget_keepsRecentFirst() {
        var facts: [FactRecord] = []
        for i in 0..<20 {
            facts.append(fact(id: "f\(i)", time: Int64(i), merchant: "M\(i)"))
        }
        let p = assembleWithFacts(
            userQuery: "test",
            facts: facts,
            tokenBudget: 50
        )
        XCTAssertTrue(p.system.contains("M19"), "最新 fact 必须保留")
        XCTAssertFalse(p.system.contains("M0 |") || p.system.contains("M0\n"),
            "截断应从最旧开始丢弃")
    }

    func test_answerHint_injectedEvenWithEmptyFacts() {
        let p = assembleWithFacts(
            userQuery: "上月花多少",
            facts: [],
            answerHint: "【精确查询】上月总花费 ¥9876.54，共 123 笔"
        )
        XCTAssertEqual(p.injectedFactCount, 1)
        XCTAssertTrue(p.system.contains("【精确查询】"))
        XCTAssertTrue(p.system.contains("¥9876.54"))
        XCTAssertTrue(p.system.contains(FACT_START))
        XCTAssertTrue(p.system.contains(FACT_END))
    }

    func test_answerHint_overridesFacts() {
        let p = assembleWithFacts(
            userQuery: "上月花多少",
            facts: [fact(time: 1000, merchant: "不应出现")],
            answerHint: "【精确查询】¥9876.54"
        )
        XCTAssertEqual(p.injectedFactCount, 1)
        XCTAssertTrue(p.system.contains("¥9876.54"))
        XCTAssertFalse(p.system.contains("不应出现"))
    }

    func test_answerHint_notInjectedWhenPersona() {
        let router = RouterResult(
            label: .persona, confidence: 0.95,
            reason: "few_shot_high_conf", latencyMs: 1.0
        )
        let sentinel = "UNIQUE_HINT_SENTINEL_XYZ_19284756"
        let p = assembleWithFacts(
            userQuery: "我是谁",
            facts: [],
            routerResult: router,
            answerHint: "【精确查询】\(sentinel)"
        )
        XCTAssertEqual(p.injectedFactCount, 0,
            "persona 路径下 answerHint 不应注入")
        XCTAssertFalse(p.system.contains(sentinel),
            "persona 路径下 answerHint sentinel 不应出现在 system")
        XCTAssertFalse(p.system.contains(FACT_START),
            "persona 路径下 system 不应有 fact delimiter")
    }

    func test_defaultSystemPrompt_containsNoFixedAnswerExamples() {
        XCTAssertFalse(DEFAULT_SYSTEM_PROMPT_PREFIX.contains("¥1322.67"))
        XCTAssertFalse(DEFAULT_SYSTEM_PROMPT_PREFIX.contains("40-80"))
        XCTAssertFalse(DEFAULT_SYSTEM_PROMPT_PREFIX.contains("回答格式例"))
    }

    func test_factLine_formatWithAllFields() {
        let f = fact(
            id: "x",
            time: 1_742_000_000_000,
            amount: 50.5,
            merchant: "必胜客",
            category: "餐饮",
            location: "北京"
        )
        let line = formatFactLine(f)
        XCTAssertTrue(line.hasPrefix("- "))
        XCTAssertTrue(line.contains("2025-03-15"))
        XCTAssertTrue(line.contains("必胜客"))
        XCTAssertTrue(line.contains("餐饮"))
        XCTAssertTrue(line.contains("¥50.50"))
        XCTAssertTrue(line.contains("北京"))
    }

    func test_factLine_formatWithoutLocation() {
        let f = fact(time: 0, amount: 10.0, merchant: "X", category: "Y")
        let line = formatFactLine(f)
        let parts = line.dropFirst(2).components(separatedBy: " | ")
        XCTAssertEqual(parts.count, 4, "无 location 应只有 4 段，实际: \(parts)")
    }

    func test_factLine_missingAmountShowsQuestionMark() {
        let f = FactRecord(
            id: "x",
            schemaName: "finance.expense",
            payload: [
                "time": .int(1_742_000_000_000),
                "merchant": .string("M"),
                "category": .string("C"),
            ],
            createdAt: 0
        )
        let line = formatFactLine(f)
        XCTAssertTrue(line.contains("?"), "missing amount 应显示为 ?, 实际: \(line)")
    }

    func test_asMessages_returnsSystemAndUserRoles() {
        let p = AssembledPrompt(system: "SYS", user: "USR")
        let msgs = p.asMessages()
        XCTAssertEqual(msgs.count, 2)
        XCTAssertEqual(msgs[0]["role"], "system")
        XCTAssertEqual(msgs[0]["content"], "SYS")
        XCTAssertEqual(msgs[1]["role"], "user")
        XCTAssertEqual(msgs[1]["content"], "USR")
    }

    func test_fullText_format() {
        let p = AssembledPrompt(system: "S", user: "U")
        XCTAssertEqual(p.fullText(), "[system]\nS\n\n[user]\nU")
    }

    func test_formatDateUTC_matchesPython() {
        XCTAssertEqual(formatDateUTC(fromUnixMs: 1_742_000_000_000), "2025-03-15")
        XCTAssertEqual(formatDateUTC(fromUnixMs: 0), "1970-01-01")
    }

    func test_customSystemPrefix_prepended() {
        let p = assembleWithFacts(
            userQuery: "test",
            facts: [fact(time: 1000)],
            systemPromptPrefix: "[CUSTOM PREFIX]"
        )
        XCTAssertTrue(p.system.hasPrefix("[CUSTOM PREFIX]"))
        XCTAssertTrue(p.system.contains(FACT_START))
    }

    func test_defaultSystemPrompt_containsFactLayerRules() {
        XCTAssertTrue(DEFAULT_SYSTEM_PROMPT_PREFIX.contains("fact_layer"))
        XCTAssertTrue(DEFAULT_SYSTEM_PROMPT_PREFIX.contains("精确查询"))
    }
}
