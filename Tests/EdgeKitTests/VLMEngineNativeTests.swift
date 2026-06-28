// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CoreImage
import XCTest
@testable import EdgeInference

final class VLMEngineNativeTests: XCTestCase {
    private static let smokeGate = "EDGE_RUN_VLM_IMAGE_SMOKE"
    private static let textSmokeGate = "EDGE_RUN_VLM_TEXT_SMOKE"
    private static let smokeModelPath = "EDGEKIT_VLM_SMOKE_MODEL_PATH"
    private static let smokeImagePath = "EDGEKIT_VLM_SMOKE_IMAGE_PATH"

    @MainActor
    func testImageGenerateRequiresLoadedNativeRuntime() async {
        let engine = VLMEngine()
        let stream = engine.generate(
            messages: [.user("Describe this image.")],
            images: [URL(fileURLWithPath: "/does/not/exist.png")]
        )
        var iterator = stream.makeAsyncIterator()

        do {
            _ = try await iterator.next()
            XCTFail("image VLM path should require a loaded native runtime")
        } catch EdgeRuntimeError.loadFailed(let reason) {
            XCTAssertTrue(reason.contains("No VLM model loaded"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    @MainActor
    func testTokenizeRequiresLoadedNativeTokenizer() async {
        let engine = VLMEngine()

        do {
            _ = try await engine.tokenize("hello")
            XCTFail("tokenize should require a loaded native tokenizer")
        } catch EdgeRuntimeError.loadFailed(let reason) {
            XCTAssertTrue(reason.contains("tokenizer"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    /// Non-gated regression guard for the 0.5 tok/s VLM text bug: temperature > 0
    /// (the default 0.7) must remain eligible for the fast Metal CMLX text path
    /// and not fall back to the slow Swift decoder. No model required.
    func testNativeVLMCmlxTextEligibilityAcceptsSamplingTemperature() {
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0.7),
                hasTools: false
            ),
            .eligible
        )
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0, topP: 1),
                hasTools: false
            ),
            .eligible
        )
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0.7),
                hasTools: true
            ),
            .ineligible(reason: "tools_present")
        )
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0.7, topK: 0),
                hasTools: false
            ),
            .ineligible(reason: "invalid_top_k")
        )
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0.7, minP: -0.1),
                hasTools: false
            ),
            .ineligible(reason: "invalid_min_p")
        )
        XCTAssertEqual(
            VLMEngine.nativeVLMCmlxTextEligibility(
                parameters: EdgeGenerateParameters(temperature: 0.7, repetitionPenalty: 0),
                hasTools: false
            ),
            .ineligible(reason: "invalid_repetition_penalty")
        )
    }

    func testNativeVLMImageFeaturePrunerBuildsUniformRowMajorPlan() {
        let plan = VLMEngine.NativeVLMImageFeaturePruner.makePlan(
            imageTokenCounts: [8, 3],
            maxTokensPerImage: 4
        )

        XCTAssertEqual(plan?.originalImageTokenCounts, [8, 3])
        XCTAssertEqual(plan?.effectiveImageTokenCounts, [4, 3])
        XCTAssertEqual(plan?.selectedRowIndices, [0, 2, 5, 7, 8, 9, 10])
        XCTAssertEqual(plan?.isPruned, true)
    }

    func testNativeVLMImageFeaturePrunerAppliesSelectedRows() throws {
        let plan = VLMEngine.NativeVLMImageFeaturePruner.makePlan(
            imageTokenCounts: [6],
            maxTokensPerImage: 3
        )
        let values = (0..<24).map(Float.init)

        let pruned = try VLMEngine.NativeVLMImageFeaturePruner.apply(
            plan: plan,
            values: values,
            shape: [6, 4]
        )

        XCTAssertEqual(pruned.shape, [3, 4])
        XCTAssertEqual(pruned.values, [
            0, 1, 2, 3,
            12, 13, 14, 15,
            20, 21, 22, 23,
        ])
    }

    func testVLMImagePolicySettingsUseBalancedByDefault() {
        let settings = VLMEngine.nativeVLMImagePolicySettings(
            for: .balanced,
            environment: [:]
        )

        XCTAssertEqual(settings.maxImageTokens, 256)
        XCTAssertNil(settings.pruneTokens)
        XCTAssertFalse(settings.maxImageTokensOverriddenByEnvironment)
        XCTAssertFalse(settings.pruneTokensOverriddenByEnvironment)
    }

    func testVLMImagePolicySettingsUseFastResizeAndPrune() {
        let settings = VLMEngine.nativeVLMImagePolicySettings(
            for: .fast,
            environment: [:]
        )

        XCTAssertEqual(settings.maxImageTokens, 130)
        XCTAssertEqual(settings.pruneTokens, 64)
    }

    func testVLMImagePolicySettingsHonorEnvironmentOverrides() {
        let settings = VLMEngine.nativeVLMImagePolicySettings(
            for: .balanced,
            environment: [
                "EDGE_VLM_IMAGE_TOKEN_BUDGET": "130",
                "EDGE_VLM_IMAGE_FEATURE_PRUNE_TOKENS": "64",
            ]
        )

        XCTAssertEqual(settings.maxImageTokens, 130)
        XCTAssertEqual(settings.pruneTokens, 64)
        XCTAssertTrue(settings.maxImageTokensOverriddenByEnvironment)
        XCTAssertTrue(settings.pruneTokensOverriddenByEnvironment)
    }

    func testVLMImagePolicySettingsCanDisablePolicyWithZeroEnvironmentOverrides() {
        let settings = VLMEngine.nativeVLMImagePolicySettings(
            for: .fast,
            environment: [
                "EDGE_VLM_IMAGE_TOKEN_BUDGET": "0",
                "EDGE_VLM_IMAGE_FEATURE_PRUNE_TOKENS": "0",
            ]
        )

        XCTAssertNil(settings.maxImageTokens)
        XCTAssertNil(settings.pruneTokens)
        XCTAssertTrue(settings.maxImageTokensOverriddenByEnvironment)
        XCTAssertTrue(settings.pruneTokensOverriddenByEnvironment)
    }

    @MainActor
    func testNativeImageGeneratePizzaSmokeWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment[Self.smokeGate] == "1" else {
            throw XCTSkip("Set \(Self.smokeGate)=1 to run the native VLM image smoke.")
        }

        let (modelURL, imageURL) = try realVLMSmokeFixture()

        let engine = VLMEngine()
        try await engine.loadLocal(directory: modelURL)
        let stream = engine.generate(
            messages: [.user("Describe this image in one short sentence.")],
            images: [imageURL],
            parameters: EdgeGenerateParameters(
                temperature: 0,
                topP: 1,
                maxTokens: 48
            )
        )

        var output = ""
        for try await chunk in stream {
            output += chunk.text
        }
        let lowercased = output.lowercased()
        XCTAssertTrue(
            ["pizza", "food", "flatbread", "cheese"].contains { lowercased.contains($0) },
            "unexpected VLM output: \(output)"
        )
    }

    @MainActor
    func testNativeCIImageGeneratePizzaSmokeWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment[Self.smokeGate] == "1" else {
            throw XCTSkip("Set \(Self.smokeGate)=1 to run the native VLM CIImage smoke.")
        }
        let (modelURL, imageURL) = try realVLMSmokeFixture()
        guard let image = CIImage(contentsOf: imageURL) else {
            XCTFail("unable to load CIImage fixture")
            return
        }

        let engine = VLMEngine()
        try await engine.loadLocal(directory: modelURL)
        let stream = engine.generate(
            messages: [.user("Describe this image in one short sentence.")],
            ciImages: [image],
            parameters: EdgeGenerateParameters(
                temperature: 0,
                topP: 1,
                maxTokens: 48
            )
        )

        var output = ""
        for try await chunk in stream {
            output += chunk.text
        }
        let lowercased = output.lowercased()
        XCTAssertTrue(
            ["pizza", "food", "flatbread", "cheese"].contains { lowercased.contains($0) },
            "unexpected VLM CIImage output: \(output)"
        )
    }

    @MainActor
    func testNativeMultiImageGenerateAcceptsTwoImagesWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment[Self.smokeGate] == "1" else {
            throw XCTSkip("Set \(Self.smokeGate)=1 to run the native VLM multi-image smoke.")
        }
        let (modelURL, imageURL) = try realVLMSmokeFixture()

        let engine = VLMEngine()
        try await engine.loadLocal(directory: modelURL)
        let stream = engine.generate(
            messages: [.user("Describe these two images briefly.")],
            images: [imageURL, imageURL],
            parameters: EdgeGenerateParameters(
                temperature: 0,
                topP: 1,
                maxTokens: 24
            )
        )

        var output = ""
        for try await chunk in stream {
            output += chunk.text
        }
        XCTAssertNotNil(engine.lastMetrics)
        XCTAssertFalse(output.isEmpty, "multi-image VLM output should not be empty")
    }

    /// Text-only VLM generation with default sampling (temperature 0.7) must
    /// take the fast Metal cmlx decode path, not the slow Swift greedy fallback
    /// that previously caused ~0.5 tok/s on iPhone Air. The policy reasoning tag
    /// is the regression guard.
    @MainActor
    func testNativeTextGenerateSampledUsesFastCmlxPathWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment[Self.textSmokeGate] == "1" else {
            throw XCTSkip("Set \(Self.textSmokeGate)=1 to run the native VLM text-only sampled smoke.")
        }
        let modelURL = try realVLMModelFixture()
        let engine = VLMEngine()
        try await engine.loadLocal(directory: modelURL)
        let stream = engine.generate(
            messages: [.user("Reply with a short greeting.")],
            parameters: EdgeGenerateParameters(temperature: 0.7, maxTokens: 48)
        )
        var output = ""
        for try await chunk in stream {
            output += chunk.text
        }
        XCTAssertFalse(output.isEmpty, "sampled VLM text output should not be empty")
        let reasoning = engine.lastMetrics?.policyReasoning ?? ""
        XCTAssertTrue(
            reasoning.contains("nativeVLMTextCmlx=sampled"),
            "sampled VLM text must use the fast cmlx path, got reasoning: \(reasoning)"
        )
    }

    @MainActor
    func testNativeTextGenerateGreedyUsesFastCmlxPathWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment[Self.textSmokeGate] == "1" else {
            throw XCTSkip("Set \(Self.textSmokeGate)=1 to run the native VLM text-only greedy smoke.")
        }
        let modelURL = try realVLMModelFixture()
        let engine = VLMEngine()
        try await engine.loadLocal(directory: modelURL)
        let stream = engine.generate(
            messages: [.user("Reply with a short greeting.")],
            parameters: EdgeGenerateParameters(temperature: 0, topP: 1, maxTokens: 48)
        )
        var output = ""
        for try await chunk in stream {
            output += chunk.text
        }
        XCTAssertFalse(output.isEmpty, "greedy VLM text output should not be empty")
        let reasoning = engine.lastMetrics?.policyReasoning ?? ""
        XCTAssertTrue(
            reasoning.contains("nativeVLMTextCmlx=greedy"),
            "greedy VLM text must use the fast cmlx path, got reasoning: \(reasoning)"
        )
    }

    private func realVLMModelFixture() throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment[Self.smokeModelPath],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("Set \(Self.smokeModelPath) to a local VLM model bundle.")
        }
        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(
            atPath: modelURL.appendingPathComponent("config.json").path
        ) else {
            throw XCTSkip("Real Qwen3.5 VLM model fixture is unavailable")
        }
        return modelURL
    }

    private func realVLMSmokeFixture() throws -> (modelURL: URL, imageURL: URL) {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment[Self.smokeModelPath],
              !modelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("Set \(Self.smokeModelPath) to a local VLM model bundle.")
        }
        guard let imagePath = environment[Self.smokeImagePath],
              !imagePath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("Set \(Self.smokeImagePath) to a local VLM image fixture.")
        }

        let modelURL = URL(fileURLWithPath: modelPath)
        let imageURL = URL(fileURLWithPath: imagePath)
        guard FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("config.json").path),
              FileManager.default.fileExists(atPath: imageURL.path)
        else {
            throw XCTSkip("Real Qwen3.5 VLM model/image fixture is unavailable")
        }
        return (modelURL, imageURL)
    }
}
