// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeInference
import XCTest

final class NativeRuntimeBridgeReexportTests: XCTestCase {
    func testEdgeInferenceReexportsNativeRuntimeBridge() {
        XCTAssertFalse(NativeRuntimeBridge.runtimeVersion.isEmpty)

        let configuration = NativeRuntimeBridge.defaultMetalConfiguration(contextLengthHint: 12_288)

        XCTAssertEqual(configuration.maxOpsPerCommandBuffer, 20)
        XCTAssertEqual(configuration.effectiveMaxOpsPerCommandBuffer, 5)
        XCTAssertEqual(configuration.maxMBPerCommandBuffer, 40)
    }
}
