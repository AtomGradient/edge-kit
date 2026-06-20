// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class RouteMatrixIntentScorerTests: XCTestCase {
    func test_intentScorerPredictsTopIntentWithoutChangingRouting() throws {
        let scorer = try RouteMatrixIntentScorer(
            hiddenSize: 2,
            intentVocab: ["base_chat", "exact_fact"],
            weights: [
                2.0, -2.0,
                0.0, 1.0,
            ],
            bias: [0.0, 0.0]
        )

        let prediction = try scorer.predict(
            embedding: [3.0, 0.0],
            calibration: Self.calibration()
        )

        XCTAssertEqual(prediction.label, "base_chat")
        XCTAssertEqual(prediction.intentTag, .baseChat)
        XCTAssertTrue(prediction.thresholdPassed)
        XCTAssertGreaterThan(prediction.probability, 0.9)
        XCTAssertEqual(prediction.threshold, 0.55)
    }

    func test_intentScorerFailsClosedOnBadEmbeddingShape() throws {
        let scorer = try RouteMatrixIntentScorer(
            hiddenSize: 2,
            intentVocab: ["base_chat", "exact_fact"],
            weights: [1, 0, 0, 1],
            bias: [0, 0]
        )

        XCTAssertThrowsError(try scorer.predict(
            embedding: [1.0],
            calibration: Self.calibration()
        )) { error in
            XCTAssertEqual(
                error as? RouteMatrixIntentScorerError,
                .embeddingSizeMismatch(expected: 2, actual: 1)
            )
        }
    }

    func test_intentScorerFailsClosedOnZeroEmbedding() throws {
        let scorer = try RouteMatrixIntentScorer(
            hiddenSize: 2,
            intentVocab: ["base_chat", "exact_fact"],
            weights: [1, 0, 0, 1],
            bias: [0, 0]
        )

        XCTAssertThrowsError(try scorer.predict(
            embedding: [0.0, 0.0],
            calibration: Self.calibration()
        )) { error in
            XCTAssertEqual(error as? RouteMatrixIntentScorerError, .zeroEmbedding)
        }
    }

    func test_intentScorerLoadsManifestSpecifiedSafetensors() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("route-matrix-scorer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let matrixURL = directory.appendingPathComponent("route_intent_matrix.safetensors")
        try SafeTensorsTestWriter.writeF32(
            tensors: [
                "intent_weights": .init(shape: [2, 2], values: [
                    Float(2.0), Float(-2.0),
                    Float(0.0), Float(1.0),
                ]),
                "intent_bias": .init(shape: [2], values: [Float(0.0), Float(0.0)]),
            ],
            to: matrixURL
        )

        let scorer = try RouteMatrixIntentScorer.load(
            adapterDirectory: directory,
            manifest: Self.manifest()
        )
        let prediction = try scorer.predict(
            embedding: [0.0, 2.0],
            calibration: Self.calibration()
        )

        XCTAssertEqual(prediction.label, "exact_fact")
        XCTAssertGreaterThan(prediction.probability, 0.7)
    }

    func test_intentScorerSafetensorsUseRowMajorMatrixContract() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("route-matrix-scorer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let matrixURL = directory.appendingPathComponent("route_intent_matrix.safetensors")
        try SafeTensorsTestWriter.writeF32(
            tensors: [
                "intent_weights": .init(shape: [2, 2], values: [
                    Float(1.0), Float(0.0),
                    Float(0.0), Float(1.0),
                ]),
                "intent_bias": .init(shape: [2], values: [Float(0.0), Float(0.0)]),
            ],
            to: matrixURL
        )

        let scorer = try RouteMatrixIntentScorer.load(
            adapterDirectory: directory,
            manifest: Self.manifest()
        )
        let prediction = try scorer.predict(
            embedding: [1.0, 0.0],
            calibration: Self.calibration(),
            normalizeEmbedding: false
        )

        XCTAssertEqual(prediction.label, "base_chat")
        let baseChatProbability = try XCTUnwrap(prediction.probabilitiesByIntent["base_chat"])
        XCTAssertEqual(
            baseChatProbability,
            1.0 / (1.0 + exp(-1.0)),
            accuracy: 1e-6
        )
    }

    func test_intentScorerRejectsMissingBiasTensor() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("route-matrix-scorer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let matrixURL = directory.appendingPathComponent("route_intent_matrix.safetensors")
        try SafeTensorsTestWriter.writeF32(
            tensors: [
                "intent_weights": .init(shape: [2, 2], values: [
                    Float(1.0), Float(0.0),
                    Float(0.0), Float(1.0),
                ]),
            ],
            to: matrixURL
        )

        XCTAssertThrowsError(try RouteMatrixIntentScorer.load(
            adapterDirectory: directory,
            manifest: Self.manifest()
        )) { error in
            XCTAssertEqual(
                error as? RouteMatrixIntentScorerError,
                .biasTensorMissing("intent_bias", available: ["intent_weights"])
            )
        }
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
            trainingRunID: "rrr-scorer-test",
            manifestSHA256: String(repeating: "b", count: 64),
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
