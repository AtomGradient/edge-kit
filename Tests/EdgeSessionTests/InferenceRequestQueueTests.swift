// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

@testable import EdgeSession
import XCTest

final class InferenceRequestQueueTests: XCTestCase {
    func test_enqueueRunsWorkExclusively() async throws {
        let queue = InferenceRequestQueue()
        let firstStarted = expectation(description: "first started")

        let first = Task {
            try await queue.enqueue {
                firstStarted.fulfill()
                try await Task.sleep(nanoseconds: 80_000_000)
                return "first"
            }
        }

        await fulfillment(of: [firstStarted], timeout: 1)

        let second = Task {
            try await queue.enqueue {
                "second"
            }
        }

        try await waitForQueueDepth(1, queue: queue)
        let busyWhileWaiting = await queue.isBusy
        XCTAssertTrue(busyWhileWaiting)

        let values = try await [first.value, second.value]
        XCTAssertEqual(values, ["first", "second"])
        let busyAfterCompletion = await queue.isBusy
        let depthAfterCompletion = await queue.queueDepth
        XCTAssertFalse(busyAfterCompletion)
        XCTAssertEqual(depthAfterCompletion, 0)
    }

    func test_cancelledWaiterIsRemovedFromQueue() async throws {
        let queue = InferenceRequestQueue()
        let firstStarted = expectation(description: "first started")

        let first = Task {
            try await queue.enqueue {
                firstStarted.fulfill()
                try await Task.sleep(nanoseconds: 120_000_000)
            }
        }

        await fulfillment(of: [firstStarted], timeout: 1)

        let second = Task {
            try await queue.enqueue {
                XCTFail("cancelled waiter should not run")
            }
        }

        try await waitForQueueDepth(1, queue: queue)
        second.cancel()
        try await Task.sleep(nanoseconds: 20_000_000)
        let depthAfterCancel = await queue.queueDepth
        XCTAssertEqual(depthAfterCancel, 0)

        do {
            try await second.value
            XCTFail("expected cancellation")
        } catch is CancellationError {
        }

        try await first.value
        let busyAfterFirstCompletion = await queue.isBusy
        XCTAssertFalse(busyAfterFirstCompletion)
    }

    private func waitForQueueDepth(
        _ expected: Int,
        queue: InferenceRequestQueue,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        for _ in 0..<50 {
            if await queue.queueDepth == expected {
                return
            }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTFail("queueDepth did not become \(expected)", file: file, line: line)
    }
}
