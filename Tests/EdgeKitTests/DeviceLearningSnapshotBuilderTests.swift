// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
import EdgeMesh

final class DeviceLearningSnapshotBuilderTests: XCTestCase {
    func testMakeSnapshotBuildsCommonDeviceLearningState() {
        let snapshot = DeviceLearningSnapshotBuilder.makeSnapshot(
            identity: .init(
                peerID: "ios-peer",
                displayName: "iPhone"
            ),
            modelConfig: .init(
                selectedModelID: "Qwen3.5-9B-4bit",
                displayName: "Qwen 9B",
                family: "qwen3.5-9b",
                quantization: "4bit",
                documentsDirectory: nil
            ),
            modelState: .init(
                isLoaded: true,
                loadedModelID: "Qwen3.5-9B-4bit",
                activeNeuralImprintPrefixTokenCount: 2_438,
                activeNeuralImprintArtifactSHA256: hash("neural-imprint")
            ),
            dataCounts: .init(
                eventStoreTotal: 12,
                factsTotal: 12,
                factsClassified: 11,
                factsRawUnclassified: 1
            ),
            rppState: .init(
                runID: "rpp-run",
                targetLayer: 11,
                aLibraryID: "directions_a_layer_11_qwen35_9b",
                aLibrarySHA256: hash("alib")
            ),
            neuralImprintDirectoryExists: true,
            toolSchemaSHA256: hash("tools"),
            capturedAtUnixSeconds: 1_779_000_000
        )

        XCTAssertEqual(snapshot.identity.peerID, "ios-peer")
        XCTAssertEqual(snapshot.model.loadState, "loaded")
        XCTAssertEqual(snapshot.data.readiness, "enough")
        XCTAssertEqual(snapshot.learning.neuralImprint.status, .active)
        XCTAssertEqual(snapshot.learning.neuralImprint.prefixTokenCount, 2_438)
        XCTAssertEqual(snapshot.learning.rpp.runID, "rpp-run")
        XCTAssertEqual(snapshot.learning.activeArtifactKind, "combined_kv")
        XCTAssertEqual(snapshot.learning.targetLayer, 11)
        XCTAssertEqual(snapshot.learning.toolSchemaSHA256, hash("tools"))
    }

    func testMakeSnapshotMarksPresentInactiveNeuralImprintWhenNotLoaded() {
        let snapshot = DeviceLearningSnapshotBuilder.makeSnapshot(
            identity: .init(peerID: "ios-peer", displayName: "iPhone"),
            modelConfig: .init(selectedModelID: "Qwen3.5-4B-6bit", documentsDirectory: nil),
            modelState: .init(isLoaded: false),
            dataCounts: .init(eventStoreTotal: 2, factsClassified: 2),
            rppState: .init(runID: "rpp-run", targetLayer: 23),
            neuralImprintDirectoryExists: true,
            toolSchemaSHA256: nil
        )

        XCTAssertEqual(snapshot.model.loadState, "not_loaded")
        XCTAssertEqual(snapshot.data.readiness, "insufficient")
        XCTAssertEqual(snapshot.learning.neuralImprint.status, .presentInactive)
        XCTAssertEqual(snapshot.learning.toolsOnly.status, .unknown)
        XCTAssertNil(snapshot.learning.activeArtifactKind)
    }

    func testQuantizationLabelReadsCommonModelIDs() {
        XCTAssertEqual(DeviceLearningSnapshotBuilder.quantizationLabel(for: "Qwen3.5-9B-4bit"), "4bit")
        XCTAssertEqual(DeviceLearningSnapshotBuilder.quantizationLabel(for: "Qwen3.5-4B-6bit"), "6bit")
        XCTAssertEqual(DeviceLearningSnapshotBuilder.quantizationLabel(for: "Qwen3.5-9B-8bit"), "8bit")
        XCTAssertNil(DeviceLearningSnapshotBuilder.quantizationLabel(for: "Qwen3.5-9B"))
    }

    private func hash(_ value: String) -> String {
        String(repeating: value, count: 64 / max(value.count, 1) + 1).prefix(64).description
    }
}
