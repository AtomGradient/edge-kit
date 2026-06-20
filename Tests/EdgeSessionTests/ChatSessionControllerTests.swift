// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
@testable import EdgeSession
import XCTest

@MainActor
final class ChatSessionControllerTests: XCTestCase {
    func test_runTurnAppendsUserAndAssistantMessages() async throws {
        let client = FakeGenerationClient(chunks: ["hello", " world"])
        let controller = ChatSessionController(client: client)
        var streamed: [String] = []

        let reply = try await controller.runTurn(
            userText: "Hi",
            systemPrompt: "System",
            onChunk: { streamed.append($0) }
        )

        XCTAssertEqual(reply, "hello world")
        XCTAssertEqual(streamed, ["hello", " world"])
        XCTAssertEqual(controller.history.map(\.role), [.system, .user, .assistant])
        XCTAssertEqual(controller.history.map(\.content), ["System", "Hi", "hello world"])
        XCTAssertEqual(client.resetReasons, ["session_start"])
        XCTAssertFalse(controller.isGenerating)
    }

    func test_replaceHistoryCompactsLongHistory() {
        let client = FakeGenerationClient(chunks: [])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 2,
            historyCharacterBudget: 40
        )

        controller.replaceHistory([
            .system("System"),
            .user("old"),
            .assistant("old answer"),
            .user("middle"),
            .assistant("middle answer"),
            .user("new"),
            .assistant("new answer"),
        ])

        XCTAssertEqual(controller.history.first?.role, .system)
        XCTAssertTrue(controller.history.first?.content.contains("older messages were compacted") == true)
        XCTAssertEqual(controller.history.suffix(4).map(\.content), ["middle", "middle answer", "new", "new answer"])
    }

    func test_generatePreparedUsesMemoryPolicyCompactionWhenProvided() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 100,
            historyCharacterBudget: 10_000
        )
        let messages = longConversation()
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .batteryFriendly,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 2_048
        ))

        _ = try await controller.generatePrepared(
            messages: messages,
            mode: .plain,
            memoryPolicy: ChatSessionMemoryPolicy(plan: plan),
            onChunk: { _ in }
        )

        let sentMessages = try XCTUnwrap(client.generatedMessages.first)
        XCTAssertTrue(sentMessages.first?.content.contains("older messages were compacted") == true)
        XCTAssertLessThan(sentMessages.count, messages.count)
        let audit = try XCTUnwrap(controller.lastEvent?.memoryPolicyCompaction)
        XCTAssertEqual(audit.intent, .batteryFriendly)
        XCTAssertEqual(audit.summaryDensity, MemoryPolicyPlanner.SummaryDensity.compact.rawValue)
        XCTAssertEqual(audit.sourceCharacterCount, characterCount(messages))
        XCTAssertEqual(audit.preparedCharacterCount, characterCount(sentMessages))
        XCTAssertEqual(audit.estimatedContextTokens, estimatedTokens(forCharacters: characterCount(messages)))
        XCTAssertEqual(audit.estimatedPreparedContextTokens, estimatedTokens(forCharacters: characterCount(sentMessages)))
        XCTAssertEqual(audit.estimatedRecentWindowTokens, audit.recentWindowTokens)
        XCTAssertEqual(audit.estimatedSummaryBudgetTokens, audit.summaryBudgetTokens)
        XCTAssertEqual(audit.recentWindowCharacterBudget, audit.recentWindowTokens * audit.estimatedCharactersPerToken)
        XCTAssertEqual(audit.summaryCharacterBudget, audit.summaryBudgetTokens * audit.estimatedCharactersPerToken)
        XCTAssertTrue(audit.tightenedCharacterBudget)
        XCTAssertTrue(audit.compactedHistory)
    }

    func test_generatePreparedWithoutMemoryPolicyKeepsExistingCompactionBehavior() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 100,
            historyCharacterBudget: 10_000
        )
        let messages = longConversation()

        _ = try await controller.generatePrepared(
            messages: messages,
            mode: .plain,
            onChunk: { _ in }
        )

        let sentMessages = try XCTUnwrap(client.generatedMessages.first)
        XCTAssertEqual(sentMessages.map(\.role), messages.map(\.role))
        XCTAssertEqual(sentMessages.map(\.content), messages.map(\.content))
        XCTAssertNil(controller.lastEvent?.memoryPolicyCompaction)
    }

    func test_exactRecallBlockedRecallRemainsAuditOnlyForSessionCompaction() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(client: client)
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .exactRecall,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 2_048,
            hasAuditableFactRequirement: true,
            factStoreAvailable: false,
            toolRecallAvailable: false
        ))

        XCTAssertEqual(plan.recall.status, .blocked)
        _ = try await controller.generatePrepared(
            messages: [.system("System"), .user("Question")],
            mode: .plain,
            memoryPolicy: ChatSessionMemoryPolicy(plan: plan),
            onChunk: { _ in }
        )

        XCTAssertNil(client.generatedTools.first ?? nil)
        let audit = try XCTUnwrap(controller.lastEvent?.memoryPolicyCompaction)
        XCTAssertEqual(audit.intent, .exactRecall)
        XCTAssertFalse(audit.compactedHistory)
    }

    func test_generatePreparedRecordsMemoryPolicyQualitySignalsWhenEnabled() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 100,
            historyCharacterBudget: 10_000
        )
        let messages = longConversation()
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .batteryFriendly,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 2_048,
            qualitySignalsEnabled: true
        ))

        _ = try await controller.generatePrepared(
            messages: messages,
            mode: .plain,
            memoryPolicy: ChatSessionMemoryPolicy(plan: plan),
            onChunk: { _ in }
        )

        let quality = try XCTUnwrap(controller.lastEvent?.memoryPolicyQuality)
        XCTAssertEqual(quality.status, .recorded)
        XCTAssertTrue(quality.shouldRecordReceipt)
        XCTAssertTrue(quality.requiresProbe)
        XCTAssertEqual(quality.audit.memoryIntent, EdgeMemoryIntent.batteryFriendly.rawValue)
        XCTAssertTrue(quality.audit.observationRecorded)
        XCTAssertTrue(quality.audit.compactionPlanned)
        XCTAssertTrue(quality.audit.compactionApplied)
        XCTAssertTrue(quality.audit.plannedRequiresPostCompactionProbe)
        XCTAssertFalse(quality.audit.qualityImprovementClaimAllowed)
        XCTAssertFalse(quality.audit.toolsExecuted)
        XCTAssertFalse(quality.audit.toolCallsCreated)
        XCTAssertFalse(quality.audit.factStoreQueried)
        XCTAssertFalse(quality.audit.meshRoutingExecuted)
        XCTAssertFalse(quality.audit.usesExpectedAnswerOrExpectedTool)
    }

    func test_generatePreparedQualityLoopDisabledFailsClosed() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 100,
            historyCharacterBudget: 10_000
        )
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .batteryFriendly,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 2_048,
            qualitySignalsEnabled: false
        ))

        _ = try await controller.generatePrepared(
            messages: longConversation(),
            mode: .plain,
            memoryPolicy: ChatSessionMemoryPolicy(plan: plan),
            onChunk: { _ in }
        )

        let quality = try XCTUnwrap(controller.lastEvent?.memoryPolicyQuality)
        XCTAssertEqual(quality.status, .disabled)
        XCTAssertFalse(quality.shouldRecordReceipt)
        XCTAssertFalse(quality.requiresProbe)
        XCTAssertFalse(quality.audit.qualityLoopEnabled)
        XCTAssertFalse(quality.audit.observationRecorded)
        XCTAssertFalse(quality.audit.compactionApplied)
        XCTAssertFalse(quality.audit.fallbackOccurred)
        XCTAssertFalse(quality.audit.userCorrectionObserved)
        XCTAssertFalse(quality.audit.qualityImprovementClaimAllowed)
    }

    func test_generatePreparedMergesCallerQualityObservationWithRuntimeCompaction() async throws {
        let client = FakeGenerationClient(chunks: ["ok"])
        let controller = ChatSessionController(
            client: client,
            maxHistoryMessages: 100,
            historyCharacterBudget: 10_000
        )
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 2_048,
            qualitySignalsEnabled: true
        ))

        _ = try await controller.generatePrepared(
            messages: longConversation(),
            mode: .plain,
            memoryPolicy: ChatSessionMemoryPolicy(plan: plan),
            memoryQualityObservation: .init(
                sessionFingerprint: " session-fp ",
                turnFingerprint: " turn-fp ",
                compactionApplied: false,
                fallbackOccurred: true,
                postCompactionProbeCompleted: true
            ),
            onChunk: { _ in }
        )

        let quality = try XCTUnwrap(controller.lastEvent?.memoryPolicyQuality)
        XCTAssertEqual(quality.status, .recorded)
        XCTAssertEqual(quality.audit.sessionFingerprint, "session-fp")
        XCTAssertEqual(quality.audit.turnFingerprint, "turn-fp")
        XCTAssertTrue(quality.audit.compactionApplied)
        XCTAssertTrue(quality.audit.fallbackOccurred)
        XCTAssertTrue(quality.audit.postCompactionProbeCompleted)
        XCTAssertTrue(quality.requiresProbe)
        XCTAssertFalse(quality.audit.usesRegexOrKeywordFactDetection)
        XCTAssertFalse(quality.audit.usesExpectedAnswerOrExpectedTool)
    }

    func test_memoryPolicyBridgeNeverRelaxesFallbackCompactionBudget() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .longSession,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 4_096
        ))
        let policy = ChatSessionMemoryPolicy(plan: plan)
        let base = HistoryCompactor.Config(
            maxMessages: 10,
            characterBudget: 1_000,
            preserveLastNTurns: 1
        )

        let effective = policy.compactorConfig(base: base)

        XCTAssertEqual(effective.characterBudget, base.characterBudget)
        XCTAssertEqual(effective.maxMessages, base.maxMessages)
        XCTAssertEqual(effective.preserveLastNTurns, base.preserveLastNTurns)
    }

    func test_memoryPolicyCompactionAuditDoesNotChangeBudgetCalculation() {
        let plan = MemoryPolicyPlanner.plan(signals: .init(
            intent: .balanced,
            contextTokenCount: 5_000,
            modelContextLimitTokens: 4_096
        ))
        let policy = ChatSessionMemoryPolicy(
            plan: plan,
            estimatedCharactersPerToken: 5,
            minimumCharacterBudget: 600
        )
        let base = HistoryCompactor.Config(
            maxMessages: 10,
            characterBudget: 10_000,
            preserveLastNTurns: 1
        )
        let effective = policy.compactorConfig(base: base)

        let audit = policy.compactionAudit(
            base: base,
            effective: effective,
            compactedHistory: true,
            sourceCharacterCount: 1_249,
            preparedCharacterCount: 901
        )

        XCTAssertEqual(audit.estimatedCharactersPerToken, 5)
        XCTAssertEqual(audit.estimatedContextTokens, 250)
        XCTAssertEqual(audit.estimatedPreparedContextTokens, 181)
        XCTAssertEqual(audit.estimatedRecentWindowTokens, plan.compaction.recentWindowTokens)
        XCTAssertEqual(audit.estimatedSummaryBudgetTokens, plan.compaction.summaryBudgetTokens)
        XCTAssertEqual(audit.recentWindowCharacterBudget, plan.compaction.recentWindowTokens * 5)
        XCTAssertEqual(audit.summaryCharacterBudget, plan.compaction.summaryBudgetTokens * 5)
        XCTAssertEqual(audit.minimumCharacterBudget, 600)
        XCTAssertEqual(audit.plannedCharacterBudget, max(600, plan.compaction.recentWindowTokens * 5))
        XCTAssertEqual(audit.effectiveCharacterBudget, min(base.characterBudget, audit.plannedCharacterBudget))
    }

    private func longConversation() -> [ChatMessage] {
        [
            .system("System"),
            .user(String(repeating: "a", count: 1_800)),
            .assistant(String(repeating: "b", count: 1_800)),
            .user(String(repeating: "c", count: 1_000)),
            .assistant(String(repeating: "d", count: 1_000)),
            .user("new question"),
        ]
    }

    private func characterCount(_ messages: [ChatMessage]) -> Int {
        messages.reduce(0) { $0 + $1.content.count }
    }

    private func estimatedTokens(forCharacters characters: Int, charactersPerToken: Int = 4) -> Int {
        guard characters > 0 else { return 0 }
        return (characters + charactersPerToken - 1) / charactersPerToken
    }
}

@MainActor
private final class FakeGenerationClient: EdgeGenerationClient {
    var currentInferenceMetrics: InferenceMetrics?
    var resetReasons: [String] = []
    var generatedMessages: [[ChatMessage]] = []
    var generatedTools: [[EdgeSessionToolSpec]?] = []
    private let chunks: [String]

    init(chunks: [String]) {
        self.chunks = chunks
    }

    func generate(
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [EdgeSessionToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters: EdgeGenerateParameters?,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String {
        generatedMessages.append(messages)
        _ = ciImages
        generatedTools.append(tools)
        _ = onToolCall
        _ = parameters
        var output = ""
        for chunk in chunks {
            output += chunk
            onChunk(chunk)
        }
        return output
    }

    func resetRuntime(reason: String) async {
        resetReasons.append(reason)
    }
}
