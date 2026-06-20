// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public final class VLMRouteMatrixEmbeddingEncoder: RouteMatrixEmbeddingEncoder, @unchecked Sendable {
    public let supportedEncoder = RouteMatrixShadowProbeSupportedEncoder(
        kinds: ["base_model_last_hidden"],
        poolings: ["mean_excluding_special"],
        layerIndices: nil
    )

    private let engine: VLMEngine

    public init(engine: VLMEngine) {
        self.engine = engine
    }

    public func encodeRouteMatrixEmbedding(
        text: String,
        encoder: RouteRouterEncoderSpec
    ) async throws -> [Float] {
        try await engine.captureRouteMatrixEmbedding(
            text: text,
            encoder: encoder
        )
    }
}

