// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class DeviceLearningSnapshotTests: XCTestCase {
    func testDeviceStateSnapshotMessageUsesStableWireShape() throws {
        let snapshot = Self.sampleSnapshot()
        let message = DeviceStateSnapshotMessage(payload: snapshot)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "device_state_snapshot")

        let payload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(payload["schema_version"] as? String, DeviceLearningSnapshot.schemaVersion)
        XCTAssertEqual(payload["captured_at_unix_seconds"] as? Double, 1_779_600_000)

        let identity = try XCTUnwrap(payload["identity"] as? [String: Any])
        XCTAssertEqual(identity["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(identity["edge_kit_version"] as? String, "1.0.0-rc49")

        let learning = try XCTUnwrap(payload["learning"] as? [String: Any])
        XCTAssertEqual(learning["target_layer"] as? Int, 31)
        let neuralImprint = try XCTUnwrap(learning["neural_imprint"] as? [String: Any])
        XCTAssertEqual(neuralImprint["status"] as? String, "active")
        XCTAssertEqual(neuralImprint["prefix_token_count"] as? Int, 2423)
        XCTAssertNil(learning["persona_kv"])

        let decoded = try JSONDecoder().decode(DeviceStateSnapshotMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func testLearningSummaryDecodesLegacyPersonaKVField() throws {
        let json = """
        {
          "tools_only": {"status": "unknown"},
          "rpp": {"status": "active"},
          "persona_kv": {"status": "active", "prefix_token_count": 1999}
        }
        """.data(using: .utf8)!

        let learning = try JSONDecoder().decode(DeviceLearningSnapshot.LearningSummary.self, from: json)

        XCTAssertEqual(learning.neuralImprint.status, .active)
        XCTAssertEqual(learning.neuralImprint.prefixTokenCount, 1999)
    }

    func testLearningStatusProviderCanProduceSnapshot() async throws {
        let provider = FakeProvider(snapshot: Self.sampleSnapshot())
        let snapshot = try await provider.makeDeviceLearningSnapshot()
        XCTAssertEqual(snapshot.identity.appID, "com.atomgradient.dogfood")
        XCTAssertEqual(snapshot.learning.neuralImprint.status, .active)
    }

    func testAppIdentityUsesEmbeddedBuildCommitWhenExplicitCommitIsAbsent() throws {
        XCTAssertEqual(
            DeviceLearningSnapshotBuilder.AppIdentity.embeddedBuildGitCommit(
                from: ["EdgeBuildCommit": "  abcdef1234  "]
            ),
            "abcdef1234"
        )
    }

    func testAppIdentityKeepsDeviceTestCommitAheadOfEmbeddedBuildCommit() throws {
        let identity = DeviceLearningSnapshotBuilder.AppIdentity(
            peerID: "ios-peer",
            displayName: "iPhone",
            gitCommit: "device-test-commit"
        )

        XCTAssertEqual(identity.gitCommit, "device-test-commit")
    }

    func testAppIdentityIgnoresUnknownEmbeddedBuildCommit() throws {
        XCTAssertNil(
            DeviceLearningSnapshotBuilder.AppIdentity.embeddedBuildGitCommit(
                from: ["EdgeBuildCommit": "unknown"]
            )
        )
        XCTAssertNil(
            DeviceLearningSnapshotBuilder.AppIdentity.embeddedBuildGitCommit(
                from: ["EdgeBuildCommit": "   "]
            )
        )
    }

    private static func sampleSnapshot() -> DeviceLearningSnapshot {
        DeviceLearningSnapshot(
            capturedAtUnixSeconds: 1_779_600_000,
            identity: .init(
                peerID: "ios-peer",
                displayName: "iPhone 17e",
                appID: "com.atomgradient.dogfood",
                bundleIdentifier: "com.atomgradient.dogfood",
                bundleVersion: "1.0",
                buildNumber: "49",
                gitCommit: "abcdef1",
                edgeKitVersion: "1.0.0-rc49",
                edgeHaloVersion: "1.0.0-rc12",
                edgeEngineVersion: "1.0.0-rc132"
            ),
            device: .init(
                platform: "iOS",
                modelIdentifier: "iPhone18,5",
                osVersion: "26.4.2",
                physicalMemoryBytes: 8_000_000_000
            ),
            model: .init(
                selectedModelID: "Qwen3.5-9B-4bit",
                loadedModelID: "Qwen3.5-9B-4bit",
                loadState: "loaded",
                installedModels: [
                    .init(
                        modelID: "Qwen3.5-9B-4bit",
                        displayName: "Qwen3.5 9B 4-bit",
                        family: "qwen3.5",
                        quantization: "4bit",
                        isSelected: true
                    )
                ]
            ),
            data: .init(
                eventStoreTotal: 355,
                factsTotal: 706,
                factsClassified: 355,
                factsRawUnclassified: 351,
                readiness: "enough"
            ),
            learning: .init(
                toolsOnly: .init(status: .active, prefixTokenCount: 900),
                rpp: .init(status: .active, runID: "rpp-1"),
                neuralImprint: .init(
                    status: .active,
                    prefixTokenCount: 2423,
                    artifactSHA256: String(repeating: "a", count: 64)
                ),
                activeArtifactKind: "neural_imprint",
                targetLayer: 31,
                aLibraryID: "qwen35-9b-layer31",
                aLibrarySHA256: String(repeating: "b", count: 64),
                toolSchemaSHA256: String(repeating: "c", count: 64)
            ),
            corrections: .init(totalCount: 2, pendingCount: 1, needsRegen: true, countsByType: ["classification": 2]),
            eval: .init(latestStatus: "passed", latestRunID: "eval-1", latestScore: 0.91),
            sync: .init(lastSnapshotAtUnixSeconds: 1_779_600_010)
        )
    }

    private struct FakeProvider: LearningStatusProvider {
        let snapshot: DeviceLearningSnapshot

        func makeDeviceLearningSnapshot() async throws -> DeviceLearningSnapshot {
            snapshot
        }
    }
}
