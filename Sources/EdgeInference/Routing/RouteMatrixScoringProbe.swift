// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public protocol RouteMatrixEmbeddingEncoder: Sendable {
    var supportedEncoder: RouteMatrixShadowProbeSupportedEncoder { get }

    func encodeRouteMatrixEmbedding(
        text: String,
        encoder: RouteRouterEncoderSpec
    ) async throws -> [Float]
}

public final class RouteMatrixScoringProbe: RouteMatrixShadowProbe, @unchecked Sendable {
    public var supportedEncoder: RouteMatrixShadowProbeSupportedEncoder {
        embeddingEncoder.supportedEncoder
    }

    private let adapterDirectory: URL
    private let embeddingEncoder: any RouteMatrixEmbeddingEncoder
    private let scorerCache: RouteMatrixIntentScorerCache

    public init(
        adapterDirectory: URL,
        embeddingEncoder: any RouteMatrixEmbeddingEncoder,
        scorerCache: RouteMatrixIntentScorerCache = RouteMatrixIntentScorerCache()
    ) {
        self.adapterDirectory = adapterDirectory
        self.embeddingEncoder = embeddingEncoder
        self.scorerCache = scorerCache
    }

    public func probe(
        input: UserInputContext,
        manifest: RouteRouterManifest,
        calibration: RouteRouterCalibration
    ) async throws -> RouteMatrixShadowProbeResult {
        let start = Date.timeIntervalSinceReferenceDate
        let embedding = try await embeddingEncoder.encodeRouteMatrixEmbedding(
            text: input.text,
            encoder: manifest.encoder
        )
        let scorer = try await scorerCache.scorer(
            adapterDirectory: adapterDirectory,
            manifest: manifest
        )
        let prediction = try scorer.predict(
            embedding: embedding,
            calibration: calibration
        )
        return RouteMatrixShadowProbeResult(
            prediction: prediction,
            latencyMs: (Date.timeIntervalSinceReferenceDate - start) * 1000
        )
    }
}

public actor RouteMatrixIntentScorerCache {
    private var scorers: [String: RouteMatrixIntentScorer] = [:]

    public init() {}

    public func scorer(
        adapterDirectory: URL,
        manifest: RouteRouterManifest
    ) throws -> RouteMatrixIntentScorer {
        let key = [
            adapterDirectory.standardizedFileURL.path,
            manifest.manifestSHA256,
            manifest.trainingRunID,
        ].joined(separator: "\u{0}")
        if let scorer = scorers[key] {
            return scorer
        }
        let scorer = try RouteMatrixIntentScorer.load(
            adapterDirectory: adapterDirectory,
            manifest: manifest
        )
        scorers[key] = scorer
        return scorer
    }
}

