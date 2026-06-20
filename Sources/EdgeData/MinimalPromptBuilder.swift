// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Default schema-driven classifier prompt builder.
public struct MinimalPromptBuilder: PromptBuilderProvider {

    public init() {}

    public func buildMessages(
        rawFact: RawFact,
        candidateSchemas: [String],
        toolNames: [String]
    ) -> [[String: String]] {
        let augmentedSystemPrompt = toolNames.isEmpty
            ? Self.systemPrompt + "\n\n" + Self.noToolGuidance
            : Self.systemPrompt + "\n\n" + Self.buildToolGuidance(toolNames: toolNames)
        return [
            ["role": "system", "content": augmentedSystemPrompt],
            ["role": "user", "content": Self.buildUserPrompt(rawFact: rawFact, candidateSchemas: candidateSchemas)]
        ]
    }

    public func parse(
        llmOutput: String,
        candidateSchemas: [String]
    ) throws -> ClassificationResult {
        return try ClassificationParser.parse(
            llmOutput: llmOutput,
            candidateSchemas: candidateSchemas
        )
    }

    static let systemPrompt = """
    你是 Edge 数据分类器. 任务: 把用户的 raw 输入归类到给定的 candidate schema,
    并填充 schema 定义的归一化字段.

    行为规则:
    1. 严格按 schema 字段类型/枚举值输出 (numeric → 浮点数, categorical → 枚举值之一)
    2. raw 字段可能含矛盾, 优先采用语义最明确的字段值
    3. "null" / "n/a" / "" 字符串视为 NULL, 不要保留
    4. 时间字段统一转 int64 ms timestamp (UTC)
    5. 金额字段保留 2 位小数 (decimal(10,2))
    6. 输出 JSON, 不输出 markdown / 解释 / thinking
    7. confidence 给出主观置信度 (0.0-1.0):
       - 1.0 = 字段语义明确无歧义
       - 0.5 = 字段歧义但能给出合理判断
       - 0.0 = 完全无法判断 (返回 schema="__failed__")
    8. ⚠️ schema 字段必须从下方"候选 schema"列表中精确选择 (通常只有一个 candidate).
       严禁创造列表外的 schema 名 — 即使你觉得某个简短名更合适, 它也不是合法 schema.
       合法 schema 名带 namespace 前缀 (业务侧 Edge.registerSchema 时定义).
       注意区分: schema 名 (带 namespace) vs payload 内字段值 (业务定义) — 它们在
       不同层, 不要混淆.
    """

    static let noToolGuidance = """
    本轮没有启用任何工具. 不要调用工具, 不要输出 <tool_call> 或 XML/函数调用标签.
    你的完整输出必须是一个 JSON object, 第一字符必须是 `{`, 最后字符必须是 `}`.
    """

    static func buildUserPrompt(rawFact: RawFact,
                                 candidateSchemas: [String]) -> String {
        let schemaSpec = renderCandidateSchemas(candidateSchemas)
        let rawSpec = renderRawPayload(rawFact.rawPayload)

        let allowedSchemas = candidateSchemas.joined(separator: " | ")
        let primarySchema = candidateSchemas.first ?? "?"
        return """
        候选 schema (你必须选下方之一, 或返回 __failed__):
        \(schemaSpec)

        Raw 输入:
        \(rawSpec)

        请输出 JSON (按以下结构, 不要 markdown / 不要解释 / 不要 thinking):
        {
          "schema": "\(primarySchema)",
          "payload": { /* 按上方 schema 字段定义填 */ },
          "confidence": <0.0-1.0>,
          "reasoning": "<简短解释字段填法依据>"
        }

        ⚠️ 关键约束:
        1. schema 字段必须填 "\(primarySchema)" 这一个值 (合法 schema 名带 namespace 前缀).
           严禁填不带 namespace 的简短名 — 简短名通常是 payload 内某字段的值, 不是 schema 名.
        2. payload 必须按上方 schema 字段定义填; 严格遵守 categorical 字段的枚举范围.
        3. 列表外的 schema 值会被系统拒绝, 浪费推断算力.
        允许 schema 集合: [\(allowedSchemas)] (或 __failed__)
        """
    }

    static func buildToolGuidance(toolNames: [String]) -> String {
        let toolList = toolNames.map { "`\($0)`" }.joined(separator: ", ")
        return """
        你可以调用以下工具查询用户历史记忆: \(toolList).

        分类前**强烈建议**先用工具看一眼用户对相似数据的历史校正/归类:
        - 如果历史有同类对象的归类, 优先尊重用户历史选择
        - 如果历史无相关记录, 走通用推断
        - 工具返回结构化 JSON, 你应该解析其中字段做参考

        这让分类**随用户成长** — 用户校正过一次后, 下次相同情况自动遵从.
        """
    }

    static func renderCandidateSchemas(_ names: [String]) -> String {
        var lines: [String] = []
        for name in names {
            guard let schema = Edge.schema(name) else {
                lines.append("- \(name): (schema 未在 SchemaRegistry 注册, classifier 应跳过)")
                continue
            }
            lines.append("- \(name):")
            if let primaryEntity = schema.semanticLabels.primaryEntity {
                lines.append("    primary entity: \(primaryEntity)")
            }
            if let primaryValue = schema.semanticLabels.primaryValue {
                lines.append("    primary value: \(primaryValue)")
            }
            if let primaryTimestamp = schema.semanticLabels.primaryTimestamp {
                lines.append("    primary timestamp: \(primaryTimestamp)")
            }
            lines.append("    payload 字段:")
            for field in schema.fields {
                let typeStr = renderFieldType(field.type)
                let required = field.required ? "required" : "optional"
                lines.append("      \(field.name) (\(typeStr), \(required))")
            }
        }
        return lines.joined(separator: "\n")
    }

    static func renderFieldType(_ type: FieldType) -> String {
        switch type {
        case .numeric: return "numeric"
        case .text: return "text"
        case .entity: return "entity (string)"
        case .timestamp: return "timestamp (int64 ms UTC)"
        case .geohash: return "geohash (string len=8)"
        case .categorical(let values):
            let preview = values.prefix(10).joined(separator: " | ")
            let suffix = values.count > 10 ? " | ... (共 \(values.count) 项)" : ""
            return "categorical[\(preview)\(suffix)]"
        }
    }

    static func renderRawPayload(_ payload: [String: Any]) -> String {
        let sortedKeys = payload.keys.sorted()
        var lines: [String] = []
        for key in sortedKeys {
            let value = payload[key] ?? NSNull()
            let valueStr = renderValue(value)
            lines.append("  \(key): \(valueStr)")
        }
        return lines.joined(separator: "\n")
    }

    static func renderValue(_ v: Any) -> String {
        if v is NSNull { return "null" }
        if let s = v as? String {
            return "\"\(s.replacingOccurrences(of: "\"", with: "\\\""))\""
        }
        if let n = v as? NSNumber {
            if String(cString: n.objCType) == "c" {
                return n.boolValue ? "true" : "false"
            }
            return n.stringValue
        }
        if let i = v as? Int { return "\(i)" }
        if let d = v as? Double { return "\(d)" }
        return "\(v)"
    }
}
