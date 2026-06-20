// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CoreImage
import XCTest
@testable import EdgeInference

final class VLMEngineNativeTests: XCTestCase {
    private static let smokeGate = "EDGE_RUN_VLM_IMAGE_SMOKE"
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
