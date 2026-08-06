// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

@testable import EdgeInference
import XCTest

final class NeuralImprintGreedyEquivalenceDiagnosticTests: XCTestCase {
    func testPathReceiptExportsHashesWithoutRawGeneratedTokens() throws {
        let receipt = NeuralImprintGreedyPathReceipt(
            mode: "visible_profile_chat",
            inputTokenCount: 42,
            generatedTokenCount: 3,
            tokenIDsSHA256: "token-hash",
            textSHA256: "text-hash"
        )

        let data = try JSONEncoder().encode(receipt)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(object["tokenIDsSHA256"] as? String, "token-hash")
        XCTAssertEqual(object["textSHA256"] as? String, "text-hash")
        XCTAssertNil(object["generatedTokenIDs"])
        XCTAssertNil(object["sampleDiagnostics"])
    }

    func testFirstDifferenceDistinguishesMismatchFromLengthMismatch() {
        XCTAssertNil(
            NeuralImprintGreedyEquivalenceSupport.firstDifference([1, 2], [1, 2])
        )
        XCTAssertEqual(
            NeuralImprintGreedyEquivalenceSupport.firstDifference([1, 2], [1, 3]),
            1
        )
        XCTAssertEqual(
            NeuralImprintGreedyEquivalenceSupport.firstDifference([1, 2], [1, 2, 3]),
            2
        )
    }

    func testComparisonSeparatesVisibleSplitFromSerializationRestore() {
        let comparison = NeuralImprintGreedyEquivalenceSupport.compare(
            visible: [10, 20, 30],
            live: [10, 21, 31],
            restored: [10, 21, 31]
        )

        XCTAssertFalse(comparison.visibleLiveTokenIDsEqual)
        XCTAssertTrue(comparison.liveRestoredTokenIDsEqual)
        XCTAssertFalse(comparison.allThreeTokenIDsEqual)
        XCTAssertEqual(comparison.visibleLiveFirstTokenDifference, 1)
        XCTAssertNil(comparison.liveRestoredFirstTokenDifference)
    }

    func testForcedTokenComparisonUsesOnlyPostInterventionTokens() {
        let comparison = NeuralImprintGreedyEquivalenceSupport.compareCrossoverTokens(
            comparisonKind: "same_state",
            leftMode: "visible_state_visible_token",
            left: [10, 20, 30, 40, 50],
            rightMode: "visible_state_live_cache_token",
            right: [10, 20, 31, 40, 60],
            interventionIndex: 2
        )

        XCTAssertFalse(comparison.postInterventionTokenIDsEqual)
        XCTAssertEqual(comparison.postInterventionFirstTokenDifference, 1)
        XCTAssertEqual(comparison.postInterventionEditDistance, 1)
        XCTAssertEqual(comparison.postInterventionNormalizedEditDistance, 0.5)
    }

    func testForcedTokenEditDistanceSupportsInsertDeleteAndSubstitute() {
        XCTAssertEqual(
            NeuralImprintGreedyEquivalenceSupport.editDistance(
                [1, 2, 3],
                [1, 4, 3, 5]
            ),
            2
        )
        XCTAssertEqual(
            NeuralImprintGreedyEquivalenceSupport.editDistance([], [1, 2]),
            2
        )
    }

    func testForcedTokenReceiptDoesNotExportRawGeneratedContent() throws {
        let receipt = NeuralImprintForcedTokenPathReceipt(
            mode: "visible_state_live_cache_token",
            stateSource: "visible_profile",
            interventionTokenSource: "live_cache",
            interventionIndex: 12,
            interventionTokenID: 100_937,
            generatedTokenCount: 220,
            postInterventionTokenCount: 207,
            tokenIDsSHA256: "token-hash",
            postInterventionTokenIDsSHA256: "post-token-hash",
            textSHA256: "text-hash"
        )

        let data = try JSONEncoder().encode(receipt)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertNil(object["generatedTokenIDs"])
        XCTAssertNil(object["text"])
        XCTAssertEqual(object["interventionTokenID"] as? Int, 100_937)
    }
}
