// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CoreImage
import EdgeInference
import Foundation

/// Protocol boundary between EdgeSession orchestration and an app's inference owner.
@MainActor
public protocol EdgeGenerationClient: AnyObject {
    var currentInferenceMetrics: InferenceMetrics? { get }

    func generate(
        messages: [ChatMessage],
        ciImages: [CIImage],
        tools: [EdgeSessionToolSpec]?,
        onToolCall: (@Sendable (ToolCall) async throws -> String)?,
        parameters: EdgeGenerateParameters?,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String

    func resetRuntime(reason: String) async
}

public extension EdgeGenerationClient {
    var currentInferenceMetrics: InferenceMetrics? { nil }

    func generate(
        messages: [ChatMessage],
        parameters: EdgeGenerateParameters? = nil,
        onChunk: @escaping @MainActor @Sendable (String) -> Void
    ) async throws -> String {
        try await generate(
            messages: messages,
            ciImages: [],
            tools: nil,
            onToolCall: nil,
            parameters: parameters,
            onChunk: onChunk
        )
    }
}
