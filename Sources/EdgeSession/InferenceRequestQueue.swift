// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public actor InferenceRequestQueue {
    public static let shared = InferenceRequestQueue()

    public private(set) var isBusy: Bool = false
    public var queueDepth: Int { waiters.count }

    private var waiters: [Waiter] = []

    public init() {}

    public func enqueue<T: Sendable>(
        priority: TaskPriority? = nil,
        work: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        _ = priority
        try await acquire()
        do {
            try Task.checkCancellation()
            let value = try await work()
            releaseNext()
            return value
        } catch {
            releaseNext()
            throw error
        }
    }

    private func acquire() async throws {
        if !isBusy {
            isBusy = true
            return
        }

        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id: id) }
        }
    }

    private func releaseNext() {
        while !waiters.isEmpty {
            let next = waiters.removeFirst()
            isBusy = true
            next.continuation.resume()
            return
        }
        isBusy = false
    }

    private func cancelWaiter(id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else {
            return
        }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private struct Waiter: Sendable {
        let id: UUID
        let continuation: CheckedContinuation<Void, any Error>
    }
}
