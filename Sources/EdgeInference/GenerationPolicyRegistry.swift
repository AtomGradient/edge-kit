// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Generation use-case key. Apps may use the built-in cases or create custom cases.
public struct GenerationUseCase: RawRepresentable, Hashable, Sendable, ExpressibleByStringLiteral {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    public init(stringLiteral value: StringLiteralType) {
        self.rawValue = value
    }

    public static let chat = GenerationUseCase("chat")
    public static let toolCall = GenerationUseCase("tool-call")
    public static let summarization = GenerationUseCase("summarization")
}

public struct GenerationPolicyKey: Hashable, Sendable {
    public let modelID: String?
    public let useCase: GenerationUseCase

    public init(modelID: String?, useCase: GenerationUseCase) {
        let trimmed = modelID?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.modelID = trimmed?.isEmpty == true ? nil : trimmed
        self.useCase = useCase
    }
}

/// Thread-safe registry for model/use-case generation parameters.
public final class GenerationPolicyRegistry: @unchecked Sendable {
    public static let shared = GenerationPolicyRegistry()

    private let lock = NSLock()
    private let defaultParameters: EdgeGenerateParameters
    private var policies: [GenerationPolicyKey: EdgeGenerateParameters]

    public init(defaultParameters: EdgeGenerateParameters = .default) {
        self.defaultParameters = defaultParameters
        self.policies = [:]
    }

    public func register(
        _ parameters: EdgeGenerateParameters,
        forModelID modelID: String? = nil,
        useCase: GenerationUseCase
    ) {
        lock.lock()
        defer { lock.unlock() }
        policies[GenerationPolicyKey(modelID: modelID, useCase: useCase)] = parameters
    }

    public func register(
        forModelID modelID: String? = nil,
        useCase: GenerationUseCase,
        configure: (inout EdgeGenerateParameters) -> Void
    ) {
        var parameters = self.parameters(forModelID: modelID, useCase: useCase)
        configure(&parameters)
        register(parameters, forModelID: modelID, useCase: useCase)
    }

    public func unregister(
        forModelID modelID: String? = nil,
        useCase: GenerationUseCase
    ) {
        lock.lock()
        defer { lock.unlock() }
        policies.removeValue(forKey: GenerationPolicyKey(modelID: modelID, useCase: useCase))
    }

    public func parameters(
        forModelID modelID: String? = nil,
        useCase: GenerationUseCase
    ) -> EdgeGenerateParameters {
        lock.lock()
        defer { lock.unlock() }

        let exactKey = GenerationPolicyKey(modelID: modelID, useCase: useCase)
        if let parameters = policies[exactKey] {
            return parameters
        }

        let globalKey = GenerationPolicyKey(modelID: nil, useCase: useCase)
        if let parameters = policies[globalKey] {
            return parameters
        }

        return defaultParameters
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        policies.removeAll()
    }
}
