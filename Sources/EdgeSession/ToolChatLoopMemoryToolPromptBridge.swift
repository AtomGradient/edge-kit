// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import Foundation

public extension ToolChatLoop.Request {
    enum MemoryToolPromptMergePolicy: Sendable, Equatable {
        case append
        case replace
    }

    /// Apply caller-provided A8 memory recall tool prompt material to a tool
    /// chat request. This helper only transforms request metadata; it does not
    /// register tools, create tool calls, execute tools, or inspect user text.
    func applying(
        memoryToolPrompt prompt: MemoryPolicyRecallToolPlanner.ToolPrompt,
        mergePolicy: MemoryToolPromptMergePolicy = .append
    ) -> Self {
        var updated = self
        switch mergePolicy {
        case .append:
            guard !prompt.toolSpecs.isEmpty || !prompt.allowedToolNames.isEmpty else {
                return updated
            }
            updated.tools = Self.appendingToolSpecs(
                existing: updated.tools,
                additions: prompt.toolSpecs
            )
            updated.allowedToolNames = Self.appendingUnique(
                existing: updated.allowedToolNames,
                additions: prompt.allowedToolNames
            )
        case .replace:
            updated.tools = Self.deduplicatedToolSpecs(prompt.toolSpecs)
            updated.allowedToolNames = Self.unique(prompt.allowedToolNames)
        }
        return updated
    }

    private static func appendingToolSpecs(
        existing: [EdgeSessionToolSpec],
        additions: [EdgeSessionToolSpec]
    ) -> [EdgeSessionToolSpec] {
        var merged = existing
        var seenNames = Set(existing.compactMap(toolName))
        for spec in additions {
            if let name = toolName(spec), !seenNames.insert(name).inserted {
                continue
            }
            merged.append(spec)
        }
        return merged
    }

    private static func deduplicatedToolSpecs(_ specs: [EdgeSessionToolSpec]) -> [EdgeSessionToolSpec] {
        var deduplicated: [EdgeSessionToolSpec] = []
        var seenNames = Set<String>()
        for spec in specs {
            if let name = toolName(spec), !seenNames.insert(name).inserted {
                continue
            }
            deduplicated.append(spec)
        }
        return deduplicated
    }

    private static func appendingUnique(existing: [String], additions: [String]) -> [String] {
        var merged = existing
        var seen = Set(existing)
        for value in additions where seen.insert(value).inserted {
            merged.append(value)
        }
        return merged
    }

    private static func unique(_ values: [String]) -> [String] {
        var result: [String] = []
        var seen = Set<String>()
        for value in values where seen.insert(value).inserted {
            result.append(value)
        }
        return result
    }

    private static func toolName(_ spec: EdgeSessionToolSpec) -> String? {
        guard let function = spec["function"] as? [String: any Sendable],
              let name = function["name"] as? String,
              !name.isEmpty else {
            return nil
        }
        return name
    }
}
