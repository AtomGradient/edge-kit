// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
import EdgeKit

@MainActor
final class CorrectionMemoryGuardCoordinatorTests: XCTestCase {
    func testPrepareForCorrection_UnloadsIdleLoadedModel() {
        var state = CorrectionMemoryGuardModelState(isLoaded: true, isBusy: false)
        var unloadCount = 0
        let coordinator = CorrectionMemoryGuardCoordinator(
            readState: { state },
            unloadModel: {
                unloadCount += 1
                state.isLoaded = false
            }
        )

        let receipt = coordinator.prepareForCorrection(surface: "review")

        XCTAssertEqual(receipt.action, .unloadModelBeforeCorrection)
        XCTAssertTrue(receipt.modelLoadedBefore)
        XCTAssertFalse(receipt.modelBusyBefore)
        XCTAssertFalse(receipt.modelLoadedAfter)
        XCTAssertEqual(unloadCount, 1)
    }

    func testPrepareForCorrection_SkipsBusyModel() {
        var unloadCount = 0
        let coordinator = CorrectionMemoryGuardCoordinator(
            readState: { CorrectionMemoryGuardModelState(isLoaded: true, isBusy: true) },
            unloadModel: { unloadCount += 1 }
        )

        let receipt = coordinator.prepareForCorrection(surface: "review")

        XCTAssertEqual(receipt.action, .skipUnloadModelBusy)
        XCTAssertTrue(receipt.modelLoadedBefore)
        XCTAssertTrue(receipt.modelBusyBefore)
        XCTAssertTrue(receipt.modelLoadedAfter)
        XCTAssertEqual(unloadCount, 0)
    }

    func testPrepareForCorrection_NoopsWhenNoModelLoaded() {
        var unloadCount = 0
        let coordinator = CorrectionMemoryGuardCoordinator(
            readState: { CorrectionMemoryGuardModelState(isLoaded: false, isBusy: false) },
            unloadModel: { unloadCount += 1 }
        )

        let receipt = coordinator.prepareForCorrection(surface: "review")

        XCTAssertEqual(receipt.action, .noModelLoaded)
        XCTAssertFalse(receipt.modelLoadedBefore)
        XCTAssertFalse(receipt.modelBusyBefore)
        XCTAssertFalse(receipt.modelLoadedAfter)
        XCTAssertEqual(unloadCount, 0)
    }
}
