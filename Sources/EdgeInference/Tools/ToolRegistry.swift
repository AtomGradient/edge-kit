// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum ToolRegistryError: Error, Equatable {
    case toolNotFound(String)
    case outputEncodingFailed
}

public final class ToolRegistry: @unchecked Sendable {
    /// Process-wide registry. iOS app processes are isolated, so the shared
    /// instance is scoped to one bundle. For tests or mesh peer scoping,
    /// instantiate a new `ToolRegistry`.
    public static let shared = ToolRegistry()

    private let lock = NSLock()
    private var tools: [String: RegisteredTool] = [:]

    public init() {}

    public func register<T: EdgeTool>(_ tool: T) {
        register(RegisteredTool(tool))
    }

    public func register(_ tool: RegisteredTool) {
        lock.lock()
        defer { lock.unlock() }
        tools[tool.metadata.name] = tool
    }

    public func unregister(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        tools.removeValue(forKey: name)
    }

    public func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        tools.removeAll()
    }

    public func tool(named name: String) -> RegisteredTool? {
        lock.lock()
        defer { lock.unlock() }
        return tools[name]
    }

    public func metadata(named name: String) -> ToolMetadata? {
        tool(named: name)?.metadata
    }

    public func allSchemas() -> [ToolMetadata] {
        lock.lock()
        defer { lock.unlock() }
        return tools.values
            .map(\.metadata)
            .sorted { $0.name < $1.name }
    }

    public func schemas(forNames names: [String]) -> [ToolMetadata] {
        lock.lock()
        defer { lock.unlock() }
        return names.compactMap { tools[$0]?.metadata }
    }

    public func schemas(forIntentTag tag: PersonalIntentTag) -> [ToolMetadata] {
        schemas(forIntentTags: [tag])
    }

    public func schemas(forIntent intent: PersonalIntent) -> [ToolMetadata] {
        schemas(forIntentTags: Self.intentTags(from: intent))
    }

    public func execute(_ plan: ToolCallPlan) async throws -> String {
        guard let tool = tool(named: plan.toolName) else {
            throw ToolRegistryError.toolNotFound(plan.toolName)
        }
        let arguments = ToolArgumentNormalizer.normalized(
            plan.arguments,
            for: tool.metadata.argumentsSchema
        )
        return try await tool.execute(arguments: arguments)
    }

    private func schemas(forIntentTags tags: Set<PersonalIntentTag>) -> [ToolMetadata] {
        let executableTags = tags.subtracting([.baseChat, .mixed])
        guard !executableTags.isEmpty else {
            return []
        }

        lock.lock()
        defer { lock.unlock() }
        return tools.values
            .map(\.metadata)
            .filter { metadata in
                !Set(metadata.intentTags).isDisjoint(with: executableTags)
            }
            .sorted { $0.name < $1.name }
    }

    private static func intentTags(from intent: PersonalIntent) -> Set<PersonalIntentTag> {
        switch intent {
        case .mixed(let candidates):
            return Set(candidates.flatMap { intentTags(from: $0) })
        default:
            return [intent.tag]
        }
    }
}
