// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Historical Fact Layer prompt pipeline.
///
/// Rule-based natural-language planning has been removed. The pipeline now
/// only preserves the prompt assembly boundary for callers that still construct
/// this type; it never performs keyword routing, natural-language planning, or
/// fact lookup.
public final class FactLayerPipeline: @unchecked Sendable {

    public struct Config: Sendable {
        /// Retained for source compatibility with existing test harnesses.
        public var referenceDate: Date?

        /// Retained for source compatibility; no list lookup is performed.
        public var listLimit: Int

        /// Prompt token budget. Passed through to PromptAssembler.
        public var tokenBudget: Int

        /// System prompt prefix. Passed through to PromptAssembler.
        public var systemPromptPrefix: String

        /// Fact injection mode. Passed through to PromptAssembler, although no
        /// facts or answer hints are produced by this pipeline.
        public var injectionMode: FactInjectionMode

        public init(
            referenceDate: Date? = nil,
            listLimit: Int = 20,
            tokenBudget: Int = DEFAULT_FACT_TOKEN_BUDGET,
            systemPromptPrefix: String = DEFAULT_SYSTEM_PROMPT_PREFIX,
            injectionMode: FactInjectionMode = .delimitered
        ) {
            self.referenceDate = referenceDate
            self.listLimit = listLimit
            self.tokenBudget = tokenBudget
            self.systemPromptPrefix = systemPromptPrefix
            self.injectionMode = injectionMode
        }
    }

    public struct Result: Sendable {
        public let routerResult: RouterResult
        public let prompt: AssembledPrompt
        public let answerHint: String?
        public let factsUsed: [FactRecord]
    }

    private let router: QueryRouter
    private let config: Config

    public init(
        store: FactStore,
        router: QueryRouter = QueryRouter(),
        config: Config = Config()
    ) {
        _ = store
        self.router = router
        self.config = config
    }

    public func chat(query: String) throws -> Result {
        let routerResult = router.classify(query)
        let prompt = assembleWithFacts(
            userQuery: query,
            facts: [],
            routerResult: routerResult,
            systemPromptPrefix: config.systemPromptPrefix,
            tokenBudget: config.tokenBudget,
            mode: config.injectionMode
        )

        return Result(
            routerResult: routerResult,
            prompt: prompt,
            answerHint: nil,
            factsUsed: []
        )
    }
}
