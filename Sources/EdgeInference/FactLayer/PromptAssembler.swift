// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Start delimiter for the fact injection section.
public let FACT_START: String = "<|fact_layer_start|>"

/// End delimiter for the fact injection section.
public let FACT_END: String = "<|fact_layer_end|>"

/// Default token budget for the fact injection section.
public let DEFAULT_FACT_TOKEN_BUDGET: Int = 512

/// Approximate character-to-token ratio used for mixed-language prompts.
public let CHARS_PER_TOKEN: Int = 4

/// Default system prompt prefix used before optional fact-layer context.
public let DEFAULT_SYSTEM_PROMPT_PREFIX: String =
    "你是用户的个人 AI 助手。基于你对用户的了解来回答，用第一人称自然口吻。\n\n" +
    "**fact_layer 处理规则（绝对优先级）：**\n" +
    "1. 如果 fact_layer 段出现【精确查询】前缀：这是从用户记账数据库查出的" +
    "权威答案。你必须**直接引用其中的具体数字**（金额、次数、日期），" +
    "不要改写、补全或编造 fact_layer 中没有出现的数字。\n" +
    "2. 如果 fact_layer 段是事实列表：用其中的商家/金额/时间支撑你的回答。\n" +
    "3. 如果没有 fact_layer：基于画像直觉回答，只能给非精确、带不确定性的概括，不要输出具体账单数字。\n\n" +
    "不要编造没有依据的精确数字；不要忽略 fact_layer 里的权威查询结果。"

/// Fully assembled chat prompt.
public struct AssembledPrompt: Sendable, Equatable {
    public let system: String
    public let user: String
    /// Number of facts injected after budget enforcement.
    public let injectedFactCount: Int
    /// Number of facts omitted because of the token budget.
    public let truncatedFactCount: Int

    public init(
        system: String,
        user: String,
        injectedFactCount: Int = 0,
        truncatedFactCount: Int = 0
    ) {
        self.system = system
        self.user = user
        self.injectedFactCount = injectedFactCount
        self.truncatedFactCount = truncatedFactCount
    }

    /// Returns messages in a standard chat-completion shape.
    public func asMessages() -> [[String: String]] {
        [
            ["role": "system", "content": system],
            ["role": "user", "content": user],
        ]
    }

    /// Returns a single string representation for tests and diagnostics.
    public func fullText() -> String {
        "[system]\n\(system)\n\n[user]\n\(user)"
    }
}

/// Fact injection mode.
public enum FactInjectionMode: Sendable, Equatable {
    /// Appends `<|fact_layer_start|>...<|fact_layer_end|>` to the system prompt.
    case delimitered
    /// Adds the fact hint to the user message while leaving the system prompt unchanged.
    case inlineUser
}

/// Assembles a user query with optional fact-layer context.
///
/// - Parameters:
///   - userQuery: Original user query.
///   - facts: Retrieved fact records.
///   - routerResult: Optional router result. Persona-only routing skips fact injection.
///   - systemPromptPrefix: System prompt prefix supplied by the caller.
///   - tokenBudget: Token budget for the fact section.
///   - answerHint: Structured query exact answer, preferred over facts injection.
///   - mode: Fact injection mode.
public func assembleWithFacts(
    userQuery: String,
    facts: [FactRecord],
    routerResult: RouterResult? = nil,
    systemPromptPrefix: String = DEFAULT_SYSTEM_PROMPT_PREFIX,
    tokenBudget: Int = DEFAULT_FACT_TOKEN_BUDGET,
    answerHint: String? = nil,
    mode: FactInjectionMode = .delimitered
) -> AssembledPrompt {
    var shouldInject = true
    if let label = routerResult?.label, label == .persona {
        shouldInject = false
    }

    if let hint = answerHint, shouldInject {
        switch mode {
        case .delimitered:
            let hintSection = "\n\n\(FACT_START)\n\(hint)\n\(FACT_END)"
            return AssembledPrompt(
                system: systemPromptPrefix + hintSection,
                user: userQuery,
                injectedFactCount: 1,
                truncatedFactCount: 0
            )
        case .inlineUser:
            let augmentedUser = "\(userQuery)\n\n（参考数据：\(hint)）"
            return AssembledPrompt(
                system: systemPromptPrefix,
                user: augmentedUser,
                injectedFactCount: 1,
                truncatedFactCount: 0
            )
        }
    }

    if !shouldInject || facts.isEmpty {
        return AssembledPrompt(
            system: systemPromptPrefix,
            user: userQuery,
            injectedFactCount: 0,
            truncatedFactCount: 0
        )
    }

    let sortedFacts = facts.sorted { a, b in
        let ta = factTime(a), tb = factTime(b)
        if ta != tb { return ta > tb }
        return a.id < b.id
    }

    var lines: [String] = []
    var totalChars = 0
    let budgetChars = tokenBudget * CHARS_PER_TOKEN
    var injected = 0

    for fact in sortedFacts {
        let line = formatFactLine(fact)
        if totalChars + line.count + 1 > budgetChars {
            break
        }
        lines.append(line)
        totalChars += line.count + 1
        injected += 1
    }

    let truncated = sortedFacts.count - injected

    var factSection = "\n\n\(FACT_START)\n" + lines.joined(separator: "\n") + "\n\(FACT_END)"
    if truncated > 0 {
        factSection += "\n（还有 \(truncated) 条因 token 预算未展示，按时间倒序优先）"
    }

    return AssembledPrompt(
        system: systemPromptPrefix + factSection,
        user: userQuery,
        injectedFactCount: injected,
        truncatedFactCount: truncated
    )
}

func factTime(_ fact: FactRecord) -> Int64 {
    fact.payload["time"]?.intValue ?? 0
}

func formatFactLine(_ fact: FactRecord) -> String {
    let payload = fact.payload

    let dateStr: String
    if let tMs = payload["time"]?.intValue {
        dateStr = formatDateUTC(fromUnixMs: tMs)
    } else {
        dateStr = "?"
    }

    let merchant = payload["merchant"]?.stringValue ?? "?"
    let category = payload["category"]?.stringValue ?? "?"
    let amountStr: String
    if let amount = payload["amount"]?.doubleValue {
        amountStr = String(format: "¥%.2f", amount)
    } else {
        amountStr = "?"
    }
    let location = payload["location"]?.stringValue ?? ""

    var parts: [String] = [dateStr, merchant, category, amountStr]
    if !location.isEmpty {
        parts.append(location)
    }
    return "- " + parts.joined(separator: " | ")
}

private let _yyyyMMddPA: DateFormatter = {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(identifier: "UTC")!
    let f = DateFormatter()
    f.calendar = cal
    f.timeZone = TimeZone(identifier: "UTC")!
    f.locale = Locale(identifier: "en_US_POSIX")
    f.dateFormat = "yyyy-MM-dd"
    return f
}()

func formatDateUTC(fromUnixMs ms: Int64) -> String {
    let date = Date(timeIntervalSince1970: Double(ms) / 1000.0)
    return _yyyyMMddPA.string(from: date)
}
