// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class RouteRouterManifestTests: XCTestCase {
    func test_routeRouterManifest_decodesPythonFixture() throws {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "route_router_manifest_v0",
            withExtension: "json"
        ))

        let manifest = try RouteRouterManifest.load(from: url)

        XCTAssertEqual(manifest.schemaVersion, "edgestudio.route_router_manifest.v0")
        XCTAssertEqual(manifest.routerType, "matrix_v0")
        XCTAssertEqual(manifest.encoder.kind, "base_model_last_hidden")
        XCTAssertEqual(manifest.encoder.hiddenSize, 2048)
        XCTAssertEqual(manifest.encoder.baseModelID, "qwen3.5-4b-base")
        XCTAssertEqual(manifest.encoder.tokenizerSHA256, String(repeating: "a", count: 64))
        XCTAssertEqual(manifest.intentVocab, [
            "base_chat",
            "exact_fact",
            "aggregate_fact",
            "app_action",
            "user_profile",
            "mixed",
        ])
        XCTAssertEqual(
            manifest.matrices["intent"],
            RouteRouterMatrixSpec(
                file: "route_intent_matrix.safetensors",
                tensor: "intent_weights",
                biasTensor: "intent_bias",
                shape: [2048, 6],
                dtype: "float16"
            )
        )
        XCTAssertEqual(manifest.calibrationFile, "route_calibration.json")
        XCTAssertEqual(manifest.fallbackChain, ["matrix", "evidence_matcher", "base_router"])
        XCTAssertEqual(
            manifest.manifestSHA256,
            "6087a153b0808c48dceda445df8d00bb600075b32b88e00a6122005ea843d4da"
        )
    }

    func test_routeRouterManifest_rejectsBaseModelMismatch() throws {
        let manifest = Self.manifest()

        XCTAssertThrowsError(try manifest.validate(expectedBaseModelID: "base-b")) { error in
            XCTAssertEqual(
                error as? RouteRouterManifestError,
                .baseModelMismatch(expected: "base-b", actual: "base-a")
            )
        }
    }

    func test_routeRouterManifest_rejectsTokenizerMismatch() throws {
        let manifest = Self.manifest()

        XCTAssertThrowsError(try manifest.validate(
            expectedBaseModelID: "base-a",
            expectedTokenizerSHA256: String(repeating: "c", count: 64)
        )) { error in
            XCTAssertEqual(
                error as? RouteRouterManifestError,
                .tokenizerMismatch(
                    expected: String(repeating: "c", count: 64),
                    actual: String(repeating: "a", count: 64)
                )
            )
        }
    }

    func test_routeRouterManifest_rejectsUnsupportedRuntimeVersion() throws {
        let manifest = Self.manifest(minRuntimeVersion: "0.9.2")

        XCTAssertThrowsError(try manifest.validate(currentRuntimeVersion: "0.9.1")) { error in
            XCTAssertEqual(
                error as? RouteRouterManifestError,
                .runtimeVersionUnsupported(minimum: "0.9.2", actual: "0.9.1")
            )
        }
    }

    func test_routeRouterManifest_rejectsIntentShapeMismatch() throws {
        let manifest = RouteRouterManifest(
            encoder: RouteRouterEncoderSpec(
                kind: "base_model_last_hidden",
                hiddenSize: 4,
                layerIndex: -1,
                pooling: "mean_excluding_special",
                baseModelID: "base-a",
                tokenizerSHA256: String(repeating: "a", count: 64)
            ),
            intentVocab: ["base_chat", "exact_fact"],
            matrices: [
                "intent": RouteRouterMatrixSpec(
                    file: "route_intent_matrix.safetensors",
                    tensor: "intent_weights",
                    biasTensor: "intent_bias",
                    shape: [4, 3],
                    dtype: "float16"
                )
            ],
            calibrationFile: "route_calibration.json",
            minRuntimeVersion: "0.9.0",
            trainingRunID: "rrr-test",
            manifestSHA256: String(repeating: "b", count: 64),
            fallbackChain: ["matrix", "evidence_matcher", "base_router"]
        )

        XCTAssertThrowsError(try manifest.validate()) { error in
            XCTAssertEqual(
                error as? RouteRouterManifestError,
                .invalidIntentMatrixShape([4, 3])
            )
        }
    }

    func test_routeRouterCalibration_rejectsInvalidThresholdRange() throws {
        let calibration = RouteRouterCalibration(
            intentTemperature: 0.8,
            intentThresholds: ["base_chat": 1.2],
            toolThresholdDefault: 0.55,
            calibrationSetSize: 20,
            calibrationECE: 0.04
        )

        XCTAssertThrowsError(try calibration.validate(intentVocab: ["base_chat"])) { error in
            XCTAssertEqual(
                error as? RouteRouterCalibrationError,
                .invalidIntentThreshold(label: "base_chat", value: 1.2)
            )
        }
    }

    func test_routeRouterCalibration_rejectsInvalidToolThreshold() throws {
        let calibration = RouteRouterCalibration(
            intentTemperature: 0.8,
            intentThresholds: ["base_chat": 0.55],
            toolThresholdDefault: -0.1,
            calibrationSetSize: 20,
            calibrationECE: 0.04
        )

        XCTAssertThrowsError(try calibration.validate(intentVocab: ["base_chat"])) { error in
            XCTAssertEqual(
                error as? RouteRouterCalibrationError,
                .invalidToolThresholdDefault(-0.1)
            )
        }
    }

    private static func manifest(minRuntimeVersion: String = "0.9.0") -> RouteRouterManifest {
        RouteRouterManifest(
            encoder: RouteRouterEncoderSpec(
                kind: "base_model_last_hidden",
                hiddenSize: 4,
                layerIndex: -1,
                pooling: "mean_excluding_special",
                baseModelID: "base-a",
                tokenizerSHA256: String(repeating: "a", count: 64)
            ),
            intentVocab: ["base_chat", "exact_fact"],
            matrices: [
                "intent": RouteRouterMatrixSpec(
                    file: "route_intent_matrix.safetensors",
                    tensor: "intent_weights",
                    biasTensor: "intent_bias",
                    shape: [4, 2],
                    dtype: "float16"
                )
            ],
            calibrationFile: "route_calibration.json",
            minRuntimeVersion: minRuntimeVersion,
            trainingRunID: "rrr-test",
            manifestSHA256: String(repeating: "b", count: 64),
            fallbackChain: ["matrix", "evidence_matcher", "base_router"]
        )
    }
}
