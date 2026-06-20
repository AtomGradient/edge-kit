// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

struct SendableBox<Value>: @unchecked Sendable {
    private let value: Value

    init(_ value: Value) {
        self.value = value
    }

    func consume() -> Value {
        value
    }
}
