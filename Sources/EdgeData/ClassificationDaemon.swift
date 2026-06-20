// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// App-provided LLM bridge used by the classification daemon.
public protocol EdgeClassificationLLMClient: Sendable {
    /// Indicates whether the app LLM is currently busy with a foreground task.
    @MainActor var isBusy: Bool { get }

    /// Model/runtime label used for classification audit fields.
    /// SDK cannot assume Qwen model size or quantization; apps should return
    /// their currently loaded model identifier.
    @MainActor var classificationModelVersion: String { get }

    /// Generates a classifier response from OpenAI-style chat messages.
    @MainActor func generate(messages: [[String: String]]) async -> String

    /// Generates a classifier response with app-owned tool names available.
    @MainActor func generate(
        messages: [[String: String]],
        toolNames: [String]
    ) async -> String
}

extension EdgeClassificationLLMClient {
    @MainActor public var classificationModelVersion: String {
        "unknown-classification-runtime"
    }

    @MainActor public func generate(
        messages: [[String: String]],
        toolNames: [String]
    ) async -> String {
        await generate(messages: messages)
    }
}

public actor ClassificationDaemon {

    public static let shared = ClassificationDaemon()

    private let batchSize = 5
    private let busyRetryNs: UInt64 = 5_000_000_000
    private let idleRetryNs: UInt64 = 30_000_000_000
    private let errorRetryNs: UInt64 = 30_000_000_000

    private var isRunning = false
    private var totalProcessed = 0
    private var totalSucceeded = 0
    private var totalFailed = 0

    private var llmClient: (any EdgeClassificationLLMClient)?
    private var toolNames: [String] = []
    private var promptBuilder: any PromptBuilderProvider = MinimalPromptBuilder()
    private var sleepTask: Task<Void, Never>?

    /// Starts the daemon. Repeated calls are ignored while it is already running.
    ///
    /// - Parameters:
    ///   - namespace: Namespace whose raw facts should be classified.
    ///   - candidateSchemas: Candidate schema names the daemon may choose from.
    ///   - llmClient: App-owned LLM bridge.
    ///   - toolNames: Optional app-owned tool names exposed to the classifier.
    ///   - promptBuilder: Prompt and parser implementation used for classification.
    public func start(
        namespace: String,
        candidateSchemas: [String],
        llmClient: any EdgeClassificationLLMClient,
        toolNames: [String] = [],
        promptBuilder: any PromptBuilderProvider = MinimalPromptBuilder()
    ) async {
        guard !isRunning else {
            NSLog("[ClassificationDaemon] start() called but already running, ignoring")
            return
        }
        self.llmClient = llmClient
        self.toolNames = toolNames
        self.promptBuilder = promptBuilder
        isRunning = true
        NSLog("[ClassificationDaemon] STARTED, namespace=\(namespace), candidates=\(candidateSchemas), batch=\(batchSize), tools=\(toolNames), promptBuilder=\(type(of: promptBuilder))")
        await runLoop(namespace: namespace, candidateSchemas: candidateSchemas)
    }

    public func stop() {
        isRunning = false
        sleepTask?.cancel()
        sleepTask = nil
        llmClient = nil
        toolNames = []
        promptBuilder = MinimalPromptBuilder()
        NSLog("[ClassificationDaemon] stop(): processed=\(totalProcessed), succeeded=\(totalSucceeded), failed=\(totalFailed)")
    }

    /// Interrupts the current daemon sleep and asks the daemon to poll again.
    public nonisolated func wake() {
        Task { await self.performWake() }
    }

    private func performWake() {
        guard let task = sleepTask else {
            return
        }
        task.cancel()
        sleepTask = nil
        NSLog("[ClassificationDaemon] wake() — interrupted sleep, will re-poll raw_unclassified")
    }

    private func wakeableSleep(nanoseconds: UInt64) async {
        let task = Task<Void, Never> {
            try? await Task.sleep(nanoseconds: nanoseconds)
        }
        sleepTask = task
        await task.value
        sleepTask = nil
    }

    private func runLoop(namespace: String, candidateSchemas: [String]) async {
        while isRunning {
            let facts: [Fact]
            do {
                facts = try Edge.queryRetryEligibleFacts(
                    namespace: namespace,
                    limit: batchSize
                )
            } catch {
                NSLog("[ClassificationDaemon] ❌ Query batch failed: \(error.localizedDescription), sleeping 30s (wakeable)")
                await wakeableSleep(nanoseconds: errorRetryNs)
                continue
            }

            if facts.isEmpty {
                NSLog("[ClassificationDaemon] No retry-eligible facts, idle 30s wakeable (processed=\(totalProcessed), succeeded=\(totalSucceeded), failed=\(totalFailed))")
                await wakeableSleep(nanoseconds: idleRetryNs)
                continue
            }

            guard let client = llmClient else {
                NSLog("[ClassificationDaemon] ❌ no LLM client configured, exiting runLoop")
                isRunning = false
                break
            }
            let busy = await client.isBusy
            if busy {
                NSLog("[ClassificationDaemon] LLM client busy (user chat), sleeping 5s (wakeable)")
                await wakeableSleep(nanoseconds: busyRetryNs)
                continue
            }

            NSLog("[ClassificationDaemon] Processing batch of \(facts.count) facts (total raw_unclassified to process)")
            let snapshotToolNames = toolNames
            let snapshotBuilder = promptBuilder
            for fact in facts {
                guard isRunning else { break }
                await processFact(
                    fact,
                    candidateSchemas: candidateSchemas,
                    client: client,
                    toolNames: snapshotToolNames,
                    promptBuilder: snapshotBuilder
                )
            }

            NSLog("[ClassificationDaemon] Batch done. Cumulative: processed=\(totalProcessed), succeeded=\(totalSucceeded), failed=\(totalFailed)")
        }
        NSLog("[ClassificationDaemon] EXITED runLoop")
    }

    private func processFact(
        _ fact: Fact,
        candidateSchemas: [String],
        client: any EdgeClassificationLLMClient,
        toolNames: [String],
        promptBuilder: any PromptBuilderProvider
    ) async {
        let factIdShort = String(fact.id.prefix(12))
        let descPreview: String = {
            if let s = fact.payload["description"] as? String {
                return String(s.prefix(60))
            }
            for key in fact.payload.keys.sorted() {
                if let s = fact.payload[key] as? String, !s.isEmpty {
                    return "[\(key)=\(s.prefix(60))]"
                }
            }
            return "?"
        }()

        NSLog("[ClassificationDaemon] → fact.id=\(factIdShort) preview=\"\(descPreview)\"")

        let rawFact = RawFact(
            namespace: fact.namespace,
            rawPayload: fact.payload,
            candidateSchemas: candidateSchemas,
            sensitivity: fact.sensitivity
        )

        let messages = promptBuilder.buildMessages(
            rawFact: rawFact,
            candidateSchemas: candidateSchemas,
            toolNames: toolNames
        )

        let startTime = Date()
        let modelVersion = await client.classificationModelVersion
        let llmOutput = toolNames.isEmpty
            ? await client.generate(messages: messages)
            : await client.generate(messages: messages, toolNames: toolNames)
        let elapsed = Date().timeIntervalSince(startTime)
        let outputPreview = String(llmOutput.prefix(200)).replacingOccurrences(of: "\n", with: "\\n")
        NSLog("[ClassificationDaemon]   LLM (\(String(format: "%.1f", elapsed))s, tools=\(toolNames.count)) output preview: \(outputPreview)...")

        do {
            let result = try promptBuilder.parse(
                llmOutput: llmOutput,
                candidateSchemas: candidateSchemas
            )
            try Edge.applyClassification(
                factId: fact.id,
                schema: result.schema,
                payload: result.payload,
                confidence: result.confidence,
                modelVer: modelVersion,
                reasoning: result.reasoning
            )
            totalProcessed += 1
            totalSucceeded += 1
            NSLog("[ClassificationDaemon] ✅ fact.id=\(factIdShort) → schema=\(result.schema), confidence=\(result.confidence), reasoning=\"\(result.reasoning?.prefix(80) ?? "")\"")
            Edge.emitClassified(factId: fact.id)
        } catch let parseError as ClassificationParseError {
            do {
                let currentRetry = (try? Edge.currentRetryCount(factId: fact.id)) ?? 0
                let newRetryCount = currentRetry + 1
                let nowMs = Int64(Date().timeIntervalSince1970 * 1000)

                if newRetryCount >= 3 {
                    totalProcessed += 1
                    totalFailed += 1
                    NSLog("[ClassificationDaemon] ❌ fact.id=\(factIdShort) parse failed (retry exhausted, count=\(newRetryCount)): \(parseError.localizedDescription)")
                    try Edge.markClassificationFailed(
                        factId: fact.id,
                        modelVer: modelVersion,
                        error: parseError
                    )
                    Edge.emitClassificationFailed(factId: fact.id, error: parseError)
                } else {
                    totalProcessed += 1
                    NSLog("[ClassificationDaemon] ⚠️ fact.id=\(factIdShort) parse failed (retry \(newRetryCount)/3, will backoff): \(parseError.localizedDescription)")
                    try Edge.markRetry(
                        factId: fact.id,
                        modelVer: modelVersion,
                        newRetryCount: newRetryCount,
                        lastAttemptAtMs: nowMs,
                        parseError: parseError
                    )
                }
            } catch {
                NSLog("[ClassificationDaemon] ❌ markRetry/markFailed also failed: \(error.localizedDescription)")
            }
        } catch {
            totalProcessed += 1
            totalFailed += 1
            NSLog("[ClassificationDaemon] ❌ fact.id=\(factIdShort) unknown error: \(error.localizedDescription)")
        }
    }
}
