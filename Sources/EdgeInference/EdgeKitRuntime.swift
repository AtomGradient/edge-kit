// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum EdgeKitRuntime {
    public static let version = "1.0.0-rc102"

    public static var nativeRuntimeVersion: String {
        NativeRuntimeBridge.runtimeVersion
    }
}
