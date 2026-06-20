// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import Foundation

public struct ToolSchemaExport: Sendable, Codable, Equatable {
    public let schemaVersion: String
    public let tools: [ToolMetadata]

    public init(
        schemaVersion: String = "edgestudio.tool_schema_export.v1",
        tools: [ToolMetadata]
    ) {
        self.schemaVersion = schemaVersion
        self.tools = tools
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case tools
    }
}

public struct ToolSchemaSnapshot: Sendable, Equatable {
    public let export: ToolSchemaExport
    public let jsonData: Data
    public let sha256: String

    public init(export: ToolSchemaExport, jsonData: Data, sha256: String) {
        self.export = export
        self.jsonData = jsonData
        self.sha256 = sha256
    }

    public var jsonString: String? {
        String(data: jsonData, encoding: .utf8)
    }
}

public extension ToolRegistry {
    func toolSchemaExport() -> ToolSchemaExport {
        ToolSchemaExport(tools: allSchemas())
    }

    func toolSchemaExport(forNames names: [String]) -> ToolSchemaExport {
        ToolSchemaExport(tools: schemas(forNames: names))
    }

    func toolSchemaSnapshot() throws -> ToolSchemaSnapshot {
        try ToolSchemaSnapshot.make(export: toolSchemaExport())
    }

    func toolSchemaSnapshot(forNames names: [String]) throws -> ToolSchemaSnapshot {
        try ToolSchemaSnapshot.make(export: toolSchemaExport(forNames: names))
    }
}

private extension ToolSchemaSnapshot {
    static func make(export: ToolSchemaExport) throws -> ToolSchemaSnapshot {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(export)
        return ToolSchemaSnapshot(
            export: export,
            jsonData: data,
            sha256: sha256Hex(data)
        )
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
