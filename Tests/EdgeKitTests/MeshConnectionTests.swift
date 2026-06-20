// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

@available(iOS 17.0, macOS 14.0, *)
final class MeshConnectionTests: XCTestCase {

    func testAwaitReady_TimesOutWhenNeverReady() async throws {
        let identity = try CertificateManager.generate(
            peerId: "test-\(UUID().uuidString)",
            displayName: "test"
        )
        let config = MeshConnection.Config(
            host: "127.0.0.1",
            port: 1,
            expectedFingerprint: String(repeating: "0", count: 64),
            identity: identity
        )
        let conn = MeshConnection(config: config)

        do {
            try await conn.awaitReady(timeout: 0.3)
            XCTFail("awaitReady should have thrown on timeout")
        } catch let err as MeshError {
            if case .connectionFailed(_, let reason) = err {
                XCTAssertTrue(
                    reason.contains("timeout") || reason.contains("not ready"),
                    "expected timeout error, got: \(reason)"
                )
            } else {
                XCTFail("expected .connectionFailed, got \(err)")
            }
        }
    }

    func testAwaitReady_RespectsTimeoutBudget() async throws {
        let identity = try CertificateManager.generate(
            peerId: "test-\(UUID().uuidString)",
            displayName: "test"
        )
        let config = MeshConnection.Config(
            host: "127.0.0.1",
            port: 1,
            expectedFingerprint: String(repeating: "0", count: 64),
            identity: identity
        )
        let conn = MeshConnection(config: config)

        let start = Date()
        do {
            try await conn.awaitReady(timeout: 0.2)
            XCTFail("should have thrown")
        } catch {
            let elapsed = Date().timeIntervalSince(start)
            XCTAssertGreaterThanOrEqual(elapsed, 0.15, "returned too early (\(elapsed)s)")
            XCTAssertLessThan(elapsed, 0.5, "took too long (\(elapsed)s)")
        }
    }
}
