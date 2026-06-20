// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
import EdgeInference
import EdgeMesh

final class HaloCapsuleAcceptPolicyTests: XCTestCase {
    func testSnapshotMirrorsToolSchemaAndRuntimePolicy() {
        let snapshot = HaloCapsuleAcceptPolicy.snapshot(
            schemaVersion: "app.policy.v1",
            baseModelID: "Qwen3.5-9B-4bit",
            modelDisplayName: "Qwen 9B",
            currentRuntimeVersion: "1.0.0-rc59",
            toolSchemaSnapshot: toolSchemaSnapshot(sha256: hash("tools"), count: 3),
            defaultEnableThinking: false
        )

        XCTAssertEqual(snapshot.schemaVersion, "app.policy.v1")
        XCTAssertEqual(snapshot.baseModelID, "Qwen3.5-9B-4bit")
        XCTAssertEqual(snapshot.currentRuntimeVersion, "1.0.0-rc59")
        XCTAssertEqual(snapshot.supportedMessageSchemaVersion, HaloCapsuleMeshMessage.supportedSchemaVersion)
        XCTAssertEqual(snapshot.supportedMessageKind, HaloCapsuleMeshMessage.offerKind)
        XCTAssertEqual(snapshot.toolSchemaSHA256, hash("tools"))
        XCTAssertEqual(snapshot.registeredToolCount, 3)
        XCTAssertNotNil(snapshot.jsonString)
    }

    func testValidateOfferReturnsCanonicalReceipt() throws {
        let message = try makeMessage()
        let policy = HaloCapsuleAcceptPolicy.Snapshot(
            baseModelID: "Qwen3.5-9B-4bit",
            modelDisplayName: "Qwen 9B",
            currentRuntimeVersion: "1.0.0",
            toolSchemaSHA256: hash("tools"),
            registeredToolCount: 0,
            defaultEnableThinking: false
        )

        let receipt = try HaloCapsuleAcceptPolicy.validateOffer(message, policy: policy)

        XCTAssertEqual(receipt.transferID, "transfer-a")
        XCTAssertEqual(receipt.capsuleID, "capsule-a")
        XCTAssertEqual(receipt.artifactID, "artifact-a")
        XCTAssertEqual(receipt.artifactBytes, 100)
        XCTAssertEqual(receipt.canonicalSHA256, try message.canonicalSHA256())
    }

    func testValidateOfferRejectsWrongBaseModel() throws {
        let message = try makeMessage()
        let policy = HaloCapsuleAcceptPolicy.Snapshot(
            baseModelID: "Qwen3.5-4B-6bit",
            modelDisplayName: "Qwen 4B",
            currentRuntimeVersion: "1.0.0",
            toolSchemaSHA256: hash("tools"),
            registeredToolCount: 0,
            defaultEnableThinking: false
        )

        XCTAssertThrowsError(try HaloCapsuleAcceptPolicy.validateOffer(message, policy: policy))
    }

    private func toolSchemaSnapshot(sha256: String, count: Int) -> ToolSchemaSnapshot {
        let tools = (0..<count).map { index in
            ToolMetadata(
                name: "tool_\(index)",
                description: "Tool \(index)",
                argumentsSchema: .jsonSchema(#"{"type":"object","properties":{}}"#)
            )
        }
        return ToolSchemaSnapshot(
            export: ToolSchemaExport(tools: tools),
            jsonData: Data(),
            sha256: sha256
        )
    }

    private func makeMessage() throws -> HaloCapsuleMeshMessage {
        let requirements = HaloCapsuleRequirementsDescriptor(
            modelConfigSHA256: hash("config"),
            modelWeightsSHA256: hash("weights"),
            tokenizerJSONSHA256: hash("tokenizer-json"),
            tokenizerConfigSHA256: hash("tokenizer-config"),
            chatTemplateSHA256: hash("chat-template"),
            systemPromptSHA256: hash("system-prompt"),
            renderedPrefixSHA256: hash("rendered-prefix"),
            prefixTokenCount: 128,
            toolSchemaSHA256: hash("tools"),
            profileBodySHA256: hash("profile"),
            enableThinking: false,
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc132",
            cacheTopologySHA256: hash("topology"),
            modelFamily: "qwen3_5",
            hiddenSize: 4_096,
            layerCount: 32
        )
        let cacheSnapshot = HaloCacheSnapshotDescriptor(
            snapshotID: "snapshot-a",
            createdAt: Date(timeIntervalSince1970: 1_779_000_000),
            tokenCount: 128,
            tokenIDsSHA256: hash("tokens"),
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc132",
            tensors: [
                HaloCacheTensorDescriptor(
                    name: "k0",
                    shape: [1, 2],
                    dtype: "float16",
                    byteCount: 4,
                    sha256: hash("k0")
                )
            ]
        )
        let artifact = HaloCapsuleArtifactDescriptor(
            artifactID: "artifact-a",
            totalBytes: 100,
            sha256: hash("artifact"),
            files: [
                HaloCapsuleArtifactFile(
                    name: "neural_imprint.safetensors",
                    byteCount: 100,
                    sha256: hash("artifact-file")
                )
            ]
        )
        let descriptor = try HaloCapsuleDescriptor.make(
            capsuleID: "capsule-a",
            createdAt: Date(timeIntervalSince1970: 1_779_000_000),
            baseModelID: "Qwen3.5-9B-4bit",
            minRuntimeVersion: "1.0.0",
            requirements: requirements,
            cacheSnapshot: cacheSnapshot,
            artifact: artifact
        )
        return HaloCapsuleMeshMessage(
            transferID: "transfer-a",
            capsule: descriptor
        )
    }

    private func hash(_ value: String) -> String {
        String(repeating: value, count: 64 / max(value.count, 1) + 1).prefix(64).description
    }
}
