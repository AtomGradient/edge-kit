// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class GenerationPolicyRegistryTests: XCTestCase {
    func testReturnsDefaultWhenNoPolicyIsRegistered() {
        let registry = GenerationPolicyRegistry()

        let params = registry.parameters(forModelID: "qwen3", useCase: .chat)

        XCTAssertEqual(params.temperature, EdgeGenerateParameters.default.temperature)
        XCTAssertEqual(params.topK, EdgeGenerateParameters.default.topK)
        XCTAssertEqual(params.topP, EdgeGenerateParameters.default.topP)
        XCTAssertEqual(params.maxTokens, EdgeGenerateParameters.default.maxTokens)
    }

    func testGlobalUseCasePolicyAppliesAcrossModels() {
        let registry = GenerationPolicyRegistry()
        registry.register(useCase: .summarization) { params in
            params.temperature = 0.2
            params.topP = 0.8
            params.maxTokens = 384
        }

        let params = registry.parameters(forModelID: "unknown-model", useCase: .summarization)

        XCTAssertEqual(params.temperature, 0.2)
        XCTAssertEqual(params.topP, 0.8)
        XCTAssertEqual(params.maxTokens, 384)
    }

    func testModelSpecificPolicyOverridesGlobalPolicy() {
        let registry = GenerationPolicyRegistry()
        registry.register(EdgeGenerateParameters(temperature: 0.7, maxTokens: 1024), useCase: .chat)
        registry.register(
            EdgeGenerateParameters(temperature: 0.1, topP: 1.0, maxTokens: 128),
            forModelID: "qwen3.5-4b",
            useCase: .chat
        )

        let params = registry.parameters(forModelID: "qwen3.5-4b", useCase: .chat)

        XCTAssertEqual(params.temperature, 0.1)
        XCTAssertEqual(params.topP, 1.0)
        XCTAssertEqual(params.maxTokens, 128)
    }

    func testCustomUseCaseKeyIsSupported() {
        let registry = GenerationPolicyRegistry()
        let classification = GenerationUseCase("classification")
        registry.register(EdgeGenerateParameters(temperature: 0, topP: 1, maxTokens: 192), useCase: classification)

        let params = registry.parameters(useCase: classification)

        XCTAssertEqual(params.temperature, 0)
        XCTAssertEqual(params.topP, 1)
        XCTAssertEqual(params.maxTokens, 192)
    }
}
