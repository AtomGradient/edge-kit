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
}
