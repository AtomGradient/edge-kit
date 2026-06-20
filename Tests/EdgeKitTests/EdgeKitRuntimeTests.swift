// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeInference

final class EdgeKitRuntimeTests: XCTestCase {
    func test_edgeKitRuntimeVersionMatchesDependencyContract() throws {
        XCTAssertEqual(
            EdgeKitRuntime.version,
            try Self.dependencyVersion(named: "edge_kit")
        )
        XCTAssertEqual(
            EdgeKitRuntime.nativeRuntimeVersion,
            try Self.dependencyVersion(named: "edge_engine")
        )
    }

    private static func dependencyVersion(named key: String) throws -> String {
        let root = try packageRoot()
        let text = try String(contentsOf: root.appendingPathComponent(".dependency_versions"))
        for line in text.split(separator: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { continue }
            let parts = trimmed.split(separator: "=", maxSplits: 1)
            guard parts.count == 2, parts[0] == key else { continue }
            return String(parts[1])
        }
        throw XCTSkip("missing dependency version for \(key)")
    }

    private static func packageRoot() throws -> URL {
        var url = URL(fileURLWithPath: #filePath)
        while url.path != "/" {
            let candidate = url.appendingPathComponent(".dependency_versions")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return url
            }
            url.deleteLastPathComponent()
        }
        throw XCTSkip("could not locate edge-kit package root")
    }
}
