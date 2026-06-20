// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class RouteMatrixScoringProbeTests: XCTestCase {
    func test_routeMatrixCaptureLayerRejectsFinalHiddenSentinelUntilSupported() throws {
        XCTAssertThrowsError(try routeMatrixResolvedCaptureLayerIndex(-1)) { error in
            XCTAssertTrue(String(describing: error).contains("final hidden capture"))
        }
    }

    func test_routeMatrixCaptureLayerAcceptsExplicitLayerIndex() throws {
        XCTAssertEqual(try routeMatrixResolvedCaptureLayerIndex(23), 23)
    }

    func test_routeMatrixMeanPoolingAcceptsTwoDimensionalHidden() throws {
        let hidden = RouteMatrixHiddenState(shape: [3, 2], values: [
            Float(1.0), Float(2.0),
            Float(3.0), Float(4.0),
            Float(5.0), Float(6.0),
        ])

        let pooled = try routeMatrixMeanPooledEmbedding(
            from: hidden,
            selectedTokenIndices: [0, 2],
            expectedHiddenSize: 2
        )

        XCTAssertEqual(pooled[0], 3.0, accuracy: 0.0001)
        XCTAssertEqual(pooled[1], 4.0, accuracy: 0.0001)
    }

    func test_routeMatrixMeanPoolingAcceptsThreeDimensionalHidden() throws {
        let hidden = RouteMatrixHiddenState(shape: [1, 3, 2], values: [
            Float(1.0), Float(2.0),
            Float(3.0), Float(4.0),
            Float(5.0), Float(6.0),
        ])

        let pooled = try routeMatrixMeanPooledEmbedding(
            from: hidden,
            selectedTokenIndices: [1, 2],
            expectedHiddenSize: 2
        )

        XCTAssertEqual(pooled[0], 4.0, accuracy: 0.0001)
        XCTAssertEqual(pooled[1], 5.0, accuracy: 0.0001)
    }

    func test_routeMatrixMeanPoolingRejectsUnexpectedHiddenSize() throws {
        let hidden = RouteMatrixHiddenState(shape: [2, 2], values: [
            Float(1.0), Float(2.0),
            Float(3.0), Float(4.0),
        ])

        XCTAssertThrowsError(
            try routeMatrixMeanPooledEmbedding(
                from: hidden,
                selectedTokenIndices: [0],
                expectedHiddenSize: 3
            )
        )
    }

    func test_routeMatrixScoringProbeScoresWithInjectedEmbeddingEncoder() async throws {
        let directory = try Self.writeMatrixFixture()
        defer { try? FileManager.default.removeItem(at: directory) }

        let probe = RouteMatrixScoringProbe(
            adapterDirectory: directory,
            embeddingEncoder: StaticEmbeddingEncoder(embedding: [1.0, 0.0])
        )
        let result = try await probe.probe(
            input: UserInputContext(text: "show my latest purchases"),
            manifest: Self.manifest(),
            calibration: Self.calibration()
        )

        XCTAssertEqual(result.prediction.label, "base_chat")
        XCTAssertTrue(result.prediction.thresholdPassed)
        XCTAssertNotNil(result.latencyMs)
    }

    func test_routeMatrixScoringProbeUsesScorerCacheAcrossCalls() async throws {
        let directory = try Self.writeMatrixFixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = RouteMatrixIntentScorerCache()
        let probe = RouteMatrixScoringProbe(
            adapterDirectory: directory,
            embeddingEncoder: StaticEmbeddingEncoder(embedding: [0.0, 1.0]),
            scorerCache: cache
        )

        let first = try await probe.probe(
            input: UserInputContext(text: "first"),
            manifest: Self.manifest(),
            calibration: Self.calibration()
        )
        try FileManager.default.removeItem(
            at: directory.appendingPathComponent("route_intent_matrix.safetensors")
        )
        let second = try await probe.probe(
            input: UserInputContext(text: "second"),
            manifest: Self.manifest(),
            calibration: Self.calibration()
        )

        XCTAssertEqual(first.prediction.label, "exact_fact")
        XCTAssertEqual(second.prediction.label, "exact_fact")
    }

    func test_routeMatrixScoringProbeExposesEmbeddingEncoderCapability() throws {
        let encoder = StaticEmbeddingEncoder(
            embedding: [1.0, 0.0],
            supportedEncoder: RouteMatrixShadowProbeSupportedEncoder(
                kinds: ["base_model_last_hidden"],
                poolings: ["mean_excluding_special"],
                layerIndices: [23]
            )
        )
        let probe = RouteMatrixScoringProbe(
            adapterDirectory: FileManager.default.temporaryDirectory,
            embeddingEncoder: encoder
        )

        XCTAssertEqual(probe.supportedEncoder, encoder.supportedEncoder)
    }

    private static func writeMatrixFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("route-matrix-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try SafeTensorsTestWriter.writeF32(
            tensors: [
                "intent_weights": .init(shape: [2, 2], values: [
                    Float(2.0), Float(-2.0),
                    Float(0.0), Float(2.0),
                ]),
                "intent_bias": .init(shape: [2], values: [Float(0.0), Float(0.0)]),
            ],
            to: directory.appendingPathComponent("route_intent_matrix.safetensors")
        )
        return directory
    }

    private static func manifest() -> RouteRouterManifest {
        RouteRouterManifest(
            encoder: RouteRouterEncoderSpec(
                kind: "base_model_last_hidden",
                hiddenSize: 2,
                layerIndex: -1,
                pooling: "mean_excluding_special",
                baseModelID: "qwen3.5-4b-base",
                tokenizerSHA256: String(repeating: "a", count: 64)
            ),
            intentVocab: ["base_chat", "exact_fact"],
            matrices: [
                "intent": RouteRouterMatrixSpec(
                    file: "route_intent_matrix.safetensors",
                    tensor: "intent_weights",
                    biasTensor: "intent_bias",
                    shape: [2, 2],
                    dtype: "float16"
                )
            ],
            calibrationFile: "route_calibration.json",
            minRuntimeVersion: "0.9.0",
            trainingRunID: "rrr-probe-test",
            manifestSHA256: String(repeating: "c", count: 64),
            fallbackChain: ["matrix", "evidence_matcher", "base_router"]
        )
    }

    private static func calibration() -> RouteRouterCalibration {
        RouteRouterCalibration(
            intentTemperature: 1.0,
            intentThresholds: [
                "base_chat": 0.55,
                "exact_fact": 0.60,
            ],
            toolThresholdDefault: 0.55,
            calibrationSetSize: 20,
            calibrationECE: 0.04
        )
    }
}

private struct StaticEmbeddingEncoder: RouteMatrixEmbeddingEncoder {
    var embedding: [Float]
    var supportedEncoder = RouteMatrixShadowProbeSupportedEncoder(
        kinds: ["base_model_last_hidden"],
        poolings: ["mean_excluding_special"],
        layerIndices: nil
    )

    func encodeRouteMatrixEmbedding(
        text: String,
        encoder: RouteRouterEncoderSpec
    ) async throws -> [Float] {
        embedding
    }
}
