// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CoreImage
import EdgeInference
import Foundation

/// Liveness-aware generation runner extracted from the device test
/// orchestrator. It treats total turn time and "no token progress" as
/// separate failure modes so slow-but-active device generations are not cut
/// off by a short wall-clock timeout.
@MainActor
public enum EdgeGenerationWatchdog {
    public enum EndReason: String, Sendable {
        case completed
        case generationError = "generation_error"
        case turnTimeout = "decode_hang_timeout"
        case livenessTimeout = "decode_hang_liveness"
    }

    public struct Configuration: Sendable {
        public var enabled: Bool
        public var turnTimeoutSeconds: TimeInterval
        public var livenessTimeoutSeconds: TimeInterval
        public var heartbeatIntervalSeconds: TimeInterval

        public init(
            enabled: Bool = true,
            turnTimeoutSeconds: TimeInterval = 300,
            livenessTimeoutSeconds: TimeInterval = 120,
            heartbeatIntervalSeconds: TimeInterval = 10
        ) {
            self.enabled = enabled
            self.turnTimeoutSeconds = turnTimeoutSeconds
            self.livenessTimeoutSeconds = livenessTimeoutSeconds
            self.heartbeatIntervalSeconds = heartbeatIntervalSeconds
        }
    }

    public struct Progress: Sendable {
        public let finished: Bool
        public let generatedChunkCount: Int
        public let totalElapsedSeconds: TimeInterval
        public let silenceSeconds: TimeInterval
    }

    public struct Result: Sendable {
        public let text: String
        public let generatedChunkCount: Int
        public let firstChunkMs: Double
        public let metrics: InferenceMetrics?
        public let endReason: EndReason
        public let errorDescription: String?

        public var timedOut: Bool {
            endReason == .turnTimeout || endReason == .livenessTimeout
        }
    }

    public struct TimeoutError: LocalizedError {
        public let result: Result
        public let configuration: Configuration

        public var errorDescription: String? {
            switch result.endReason {
            case .turnTimeout:
                return "Generation timed out after \(Int(configuration.turnTimeoutSeconds))s"
            case .livenessTimeout:
                return "Generation stalled after \(Int(configuration.livenessTimeoutSeconds))s without token progress"
            case .generationError:
                return result.errorDescription ?? "Generation failed"
            case .completed:
                return nil
            }
        }
    }

    public typealias DiagnosticSink = @Sendable (_ message: String) -> Void

    public static func run(
        client: any EdgeGenerationClient,
        messages: [ChatMessage],
        ciImages: [CIImage] = [],
        tools: [EdgeSessionToolSpec]? = nil,
        onToolCall: (@Sendable (ToolCall) async throws -> String)? = nil,
        parameters: EdgeGenerateParameters? = nil,
        configuration: Configuration = .init(),
        scenarioID: String = "generation",
        turnIndex: Int = 1,
        diagnosticSink: DiagnosticSink? = nil,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> Result {
        let state = TurnState(startTime: CFAbsoluteTimeGetCurrent())
        diagnosticSink?(
            "generation_watchdog_begin scenario=\(scenarioID) turn=\(turnIndex) enabled=\(configuration.enabled) watchdog=\(Int(configuration.turnTimeoutSeconds))s liveness=\(Int(configuration.livenessTimeoutSeconds))s"
        )

        if !configuration.enabled {
            await runGenerationTask(
                client: client,
                messages: messages,
                ciImages: ciImages,
                tools: tools,
                onToolCall: onToolCall,
                parameters: parameters,
                state: state,
                onChunk: onChunk
            )
            return makeResult(
                state: state,
                metrics: client.currentInferenceMetrics,
                fallbackEndReason: nil
            )
        }

        let endReason = await runWithBackgroundWatchdog(
            client: client,
            messages: messages,
            ciImages: ciImages,
            tools: tools,
            onToolCall: onToolCall,
            parameters: parameters,
            state: state,
            onChunk: onChunk,
            configuration: configuration,
            scenarioID: scenarioID,
            turnIndex: turnIndex,
            diagnosticSink: diagnosticSink
        )
        if endReason == .turnTimeout || endReason == .livenessTimeout {
            await client.resetRuntime(reason: "generation_watchdog_timeout")
        }
        let result = makeResult(
            state: state,
            metrics: client.currentInferenceMetrics,
            fallbackEndReason: endReason
        )

        if result.timedOut {
            throw TimeoutError(result: result, configuration: configuration)
        }
        if result.endReason == .generationError {
            return result
        }
        return result
    }

    private static func runGenerationTask(
        client: any EdgeGenerationClient,
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [EdgeSessionToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters: EdgeGenerateParameters?,
        state: TurnState,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async {
        do {
            _ = try await client.generate(
                messages: messages,
                ciImages: ciImages,
                tools: tools,
                onToolCall: onToolCall,
                parameters: parameters
            ) { chunk in
                guard !Task.isCancelled else { return }
                state.recordChunk(chunk)
                onChunk(chunk)
            }
            state.markFinished(error: nil)
        } catch {
            state.markFinished(error: error.localizedDescription)
        }
    }

    private static func runWithBackgroundWatchdog(
        client: any EdgeGenerationClient,
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [EdgeSessionToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters: EdgeGenerateParameters?,
        state: TurnState,
        onChunk: @escaping @MainActor @Sendable (String) -> Void,
        configuration: Configuration,
        scenarioID: String,
        turnIndex: Int,
        diagnosticSink: DiagnosticSink?
    ) async -> EndReason? {
        await withCheckedContinuation { continuation in
            let completion = Completion()
            let generationTask = Task { @MainActor in
                await runGenerationTask(
                    client: client,
                    messages: messages,
                    ciImages: ciImages,
                    tools: tools,
                    onToolCall: onToolCall,
                    parameters: parameters,
                    state: state,
                    onChunk: onChunk
                )
                if completion.tryComplete() {
                    continuation.resume(returning: nil)
                }
            }

            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
            timer.schedule(deadline: .now() + 1, repeating: 1)
            timer.setEventHandler {
                if completion.isFinished {
                    timer.cancel()
                    return
                }

                let progress = state.progress()
                if shouldEmitHeartbeat(
                    elapsedSeconds: progress.totalElapsedSeconds,
                    intervalSeconds: configuration.heartbeatIntervalSeconds,
                    state: state
                ) {
                    diagnosticSink?(
                        "generation_watchdog_heartbeat scenario=\(scenarioID) turn=\(turnIndex) generated=\(progress.generatedChunkCount) elapsed=\(String(format: "%.1f", progress.totalElapsedSeconds))s silence=\(String(format: "%.1f", progress.silenceSeconds))s"
                    )
                }

                let reason: EndReason?
                if progress.totalElapsedSeconds >= configuration.turnTimeoutSeconds {
                    reason = .turnTimeout
                    diagnosticSink?(
                        "generation_watchdog_hang scenario=\(scenarioID) turn=\(turnIndex) type=turn_timeout generated=\(progress.generatedChunkCount) elapsed=\(String(format: "%.1f", progress.totalElapsedSeconds))s silence=\(String(format: "%.1f", progress.silenceSeconds))s"
                    )
                } else if progress.silenceSeconds >= configuration.livenessTimeoutSeconds {
                    reason = .livenessTimeout
                    diagnosticSink?(
                        "generation_watchdog_hang scenario=\(scenarioID) turn=\(turnIndex) type=decode_stall generated=\(progress.generatedChunkCount) elapsed=\(String(format: "%.1f", progress.totalElapsedSeconds))s silence=\(String(format: "%.1f", progress.silenceSeconds))s"
                    )
                } else {
                    reason = nil
                }

                guard let reason else { return }
                generationTask.cancel()
                if completion.tryComplete() {
                    timer.cancel()
                    continuation.resume(returning: reason)
                }
            }
            timer.resume()
        }
    }

    private static func makeResult(
        state: TurnState,
        metrics: InferenceMetrics?,
        fallbackEndReason: EndReason?
    ) -> Result {
        let snapshot = state.snapshot()
        let endReason: EndReason
        if let fallbackEndReason {
            endReason = fallbackEndReason
        } else if snapshot.error != nil {
            endReason = .generationError
        } else {
            endReason = .completed
        }
        return Result(
            text: snapshot.text,
            generatedChunkCount: snapshot.generatedChunkCount,
            firstChunkMs: snapshot.firstChunkMs,
            metrics: metrics,
            endReason: endReason,
            errorDescription: snapshot.error
        )
    }

    private static func shouldEmitHeartbeat(
        elapsedSeconds: TimeInterval,
        intervalSeconds: TimeInterval,
        state: TurnState
    ) -> Bool {
        guard intervalSeconds > 0 else { return false }
        let bucket = Int(elapsedSeconds / intervalSeconds)
        guard bucket > 0 else { return false }
        return state.markHeartbeatBucketIfNeeded(bucket)
    }
}

private final class TurnState: @unchecked Sendable {
    struct Snapshot: Sendable {
        let text: String
        let generatedChunkCount: Int
        let firstChunkMs: Double
        let error: String?
    }

    private let lock = NSLock()
    private let startTime: CFAbsoluteTime
    private var lastChunkTime: CFAbsoluteTime?
    private var text = ""
    private var generatedChunkCount = 0
    private var firstChunkMs: Double = 0
    private var finished = false
    private var error: String?
    private var lastHeartbeatBucket = -1

    init(startTime: CFAbsoluteTime) {
        self.startTime = startTime
    }

    func recordChunk(_ chunk: String, now: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()) {
        lock.lock()
        defer { lock.unlock() }
        if generatedChunkCount == 0 {
            firstChunkMs = (now - startTime) * 1000
        }
        generatedChunkCount += 1
        text += chunk
        lastChunkTime = now
    }

    func markFinished(error: String?) {
        lock.lock()
        self.error = error
        self.finished = true
        lock.unlock()
    }

    func progress(now: CFAbsoluteTime = CFAbsoluteTimeGetCurrent()) -> EdgeGenerationWatchdog.Progress {
        lock.lock()
        defer { lock.unlock() }
        let lastProgressTime = lastChunkTime ?? startTime
        return EdgeGenerationWatchdog.Progress(
            finished: finished,
            generatedChunkCount: generatedChunkCount,
            totalElapsedSeconds: now - startTime,
            silenceSeconds: now - lastProgressTime
        )
    }

    func markHeartbeatBucketIfNeeded(_ bucket: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard bucket != lastHeartbeatBucket else { return false }
        lastHeartbeatBucket = bucket
        return true
    }

    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            text: text,
            generatedChunkCount: generatedChunkCount,
            firstChunkMs: firstChunkMs,
            error: error
        )
    }
}

private final class Completion: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false

    var isFinished: Bool {
        lock.lock()
        defer { lock.unlock() }
        return finished
    }

    func tryComplete() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if finished {
            return false
        }
        finished = true
        return true
    }
}
