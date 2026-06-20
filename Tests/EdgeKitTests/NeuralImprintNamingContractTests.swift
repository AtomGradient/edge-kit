// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest

final class NeuralImprintNamingContractTests: XCTestCase {
    func testSourceKeepsOldNameOnlyInExplicitLegacyBoundaries() throws {
        let root = Self.packageRoot()
        let sourceRoot = root.appendingPathComponent("Sources", isDirectory: true)
        let files = try Self.swiftFiles(under: sourceRoot)
        let oldSnake = "persona" + "_kv"
        let needles = [
            "persona" + "kv",
            "persona" + "KV",
            "Persona" + "KV",
            "Persona " + "KV",
            "persona " + "kv",
            "persona" + "-kv",
            oldSnake,
            "EDGE_" + "PERSONA" + "_KV",
        ]

        var unexpected: [String] = []
        for file in files {
            let relativePath = file.path.replacingOccurrences(of: root.path + "/", with: "")
            let contents = try String(contentsOf: file, encoding: .utf8)
            for (lineNumber, line) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let text = String(line)
                guard needles.contains(where: { text.range(of: $0, options: [.caseInsensitive]) != nil }) else {
                    continue
                }
                if !Self.isExplicitLegacyBoundary(relativePath: relativePath, line: text) {
                    unexpected.append("\(relativePath):\(lineNumber + 1): \(text)")
                }
            }
        }

        XCTAssertTrue(
            unexpected.isEmpty,
            "Old Neural Imprint names must not remain in live source:\n\(unexpected.joined(separator: "\n"))"
        )
    }

    func testPublicSurfaceUsesNeuralImprintPrimaryNames() throws {
        let root = Self.packageRoot()
        let llmEngine = try Self.read(
            "Sources/EdgeInference/NativeDefault/LLMEngine.swift",
            root: root
        )
        XCTAssertTrue(llmEngine.contains("public struct NeuralImprintArtifactCaptureRequest"))
        XCTAssertTrue(llmEngine.contains("public struct NeuralImprintPrefixRender"))
        XCTAssertTrue(llmEngine.contains("public private(set) var activeNeuralImprintCache"))
        XCTAssertTrue(llmEngine.contains("public func restoreNeuralImprintCache"))
        XCTAssertTrue(llmEngine.contains("public func captureNeuralImprintArtifact"))

        let transport = try Self.read(
            "Sources/EdgeMesh/Inference/JointInferenceTransport.swift",
            root: root
        )
        XCTAssertTrue(transport.contains("public var useNeuralImprint: Bool"))

        let snapshot = try Self.read(
            "Sources/EdgeMesh/Lifecycle/DeviceLearningSnapshot.swift",
            root: root
        )
        XCTAssertTrue(snapshot.contains("public let neuralImprint: Artifact"))

        let builder = try Self.read(
            "Sources/EdgeMesh/Lifecycle/DeviceLearningSnapshotBuilder.swift",
            root: root
        )
        XCTAssertTrue(builder.contains("activeNeuralImprintPrefixTokenCount"))
        XCTAssertTrue(builder.contains("hasActiveNeuralImprintCache"))
    }

    private static func isExplicitLegacyBoundary(relativePath: String, line: String) -> Bool {
        let lower = line.lowercased()
        switch relativePath {
        case "Sources/EdgeInference/NativeDefault/LLMEngine.swift":
            return lower.contains("legacy")
        case "Sources/EdgeMesh/Inference/JointInferenceTransport.swift",
             "Sources/EdgeMesh/Lifecycle/DeviceLearningSnapshot.swift":
            return lower.contains("legacy") || lower.contains("decodedlegacy")
        default:
            return false
        }
    }

    private static func packageRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func swiftFiles(under directory: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        var files: [URL] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true, url.pathExtension == "swift" {
                files.append(url)
            }
        }
        return files.sorted { $0.path < $1.path }
    }

    private static func read(_ path: String, root: URL) throws -> String {
        try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }
}
