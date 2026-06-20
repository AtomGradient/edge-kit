// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class JointInferenceTransportTests: XCTestCase {
    @available(iOS 17.0, macOS 14.0, *)
    final class ClientHarness: @unchecked Sendable {
        private let lock = NSLock()
        private var _handler: JointInferenceClient.EventHandler?
        private var _sent: [JointInferenceRequestPayload] = []
        private var _cancels: [JointInferenceCancelPayload] = []
        private var _events: [JointInferenceEventType] = []

        var sent: [JointInferenceRequestPayload] {
            lock.lock(); defer { lock.unlock() }
            return _sent
        }

        var events: [JointInferenceEventType] {
            lock.lock(); defer { lock.unlock() }
            return _events
        }

        var cancels: [JointInferenceCancelPayload] {
            lock.lock(); defer { lock.unlock() }
            return _cancels
        }

        func register(_ handler: @escaping JointInferenceClient.EventHandler) {
            lock.lock(); defer { lock.unlock() }
            _handler = handler
        }

        func send(_ payload: JointInferenceRequestPayload) {
            lock.lock(); defer { lock.unlock() }
            _sent.append(payload)
        }

        func cancel(_ payload: JointInferenceCancelPayload) {
            lock.lock(); defer { lock.unlock() }
            _cancels.append(payload)
        }

        func recordEvent(_ event: JointInferenceEventPayload) {
            lock.lock(); defer { lock.unlock() }
            _events.append(event.type)
        }

        func emit(_ event: JointInferenceEventPayload) {
            lock.lock()
            let handler = _handler
            lock.unlock()
            handler?(event)
        }
    }

    func testRequestMessageUsesStableWireShape() throws {
        var payload = JointInferenceRequestPayload(
            requestID: "req-001",
            conversationID: "conv-001",
            peerID: "ios-peer",
            modelID: "host-loaded-model",
            prompt: nil,
            messages: [
                JointInferenceMessage(role: "system", content: "You are helpful."),
                JointInferenceMessage(role: "user", content: "hello"),
            ],
            maxTokens: 128,
            temperature: 0.1,
            topK: 20,
            topP: 0.9,
            enableThinking: false,
            useNeuralImprint: true,
            routeReason: "controlled_smoke"
        )
        payload.useNeuralImprint = true
        let message = JointInferenceRequestMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "joint_inference_request")

        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(encodedPayload["schema_version"] as? String, JointInferenceRequestPayload.schemaVersion)
        XCTAssertEqual(encodedPayload["request_id"] as? String, "req-001")
        XCTAssertEqual(encodedPayload["conversation_id"] as? String, "conv-001")
        XCTAssertEqual(encodedPayload["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(encodedPayload["model_id"] as? String, "host-loaded-model")
        XCTAssertEqual(encodedPayload["max_tokens"] as? Int, 128)
        XCTAssertEqual(encodedPayload["temperature"] as? Double, 0.1)
        XCTAssertEqual(encodedPayload["top_k"] as? Int, 20)
        XCTAssertEqual(encodedPayload["top_p"] as? Double, 0.9)
        XCTAssertEqual(encodedPayload["enable_thinking"] as? Bool, false)
        XCTAssertEqual(encodedPayload["use_neural_imprint"] as? Bool, true)
        XCTAssertNil(encodedPayload["use_persona_kv"])
        XCTAssertEqual(encodedPayload["route_reason"] as? String, "controlled_smoke")

        let messages = try XCTUnwrap(encodedPayload["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0]["role"] as? String, "system")
        XCTAssertEqual(messages[1]["content"] as? String, "hello")

        let decoded = try JSONDecoder().decode(JointInferenceRequestMessage.self, from: data)
        XCTAssertEqual(decoded, message)
        XCTAssertTrue(decoded.payload.useNeuralImprint)
        XCTAssertEqual(decoded.payload.conversationID, "conv-001")
    }

    func testRequestPayloadDecodesLegacyPersonaKVWireKey() throws {
        let json = """
        {
          "schema_version": "\(JointInferenceRequestPayload.schemaVersion)",
          "request_id": "req-legacy",
          "messages": [{"role": "user", "content": "hello"}],
          "max_tokens": 64,
          "temperature": 0.2,
          "use_persona_kv": true
        }
        """.data(using: .utf8)!

        let payload = try JSONDecoder().decode(JointInferenceRequestPayload.self, from: json)

        XCTAssertTrue(payload.useNeuralImprint)
        XCTAssertNil(payload.conversationID)
    }

    func testEventMessageUsesStableWireShape() throws {
        let payload = JointInferenceEventPayload(
            requestID: "req-001",
            type: .complete,
            sequence: 3,
            fullText: "done",
            totalTokens: 4,
            tokensPerSecond: 12.5,
            prefillTime: 0.2,
            totalTime: 0.5,
            modelID: "host-loaded-model",
            modelPath: "/models/qwen"
        )
        let message = JointInferenceEventMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "joint_inference_event")

        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(encodedPayload["schema_version"] as? String, JointInferenceEventPayload.schemaVersion)
        XCTAssertEqual(encodedPayload["request_id"] as? String, "req-001")
        XCTAssertEqual(encodedPayload["type"] as? String, "complete")
        XCTAssertEqual(encodedPayload["sequence"] as? Int, 3)
        XCTAssertEqual(encodedPayload["full_text"] as? String, "done")
        XCTAssertEqual(encodedPayload["total_tokens"] as? Int, 4)
        XCTAssertEqual(encodedPayload["tokens_per_sec"] as? Double, 12.5)
        XCTAssertEqual(encodedPayload["prefill_time"] as? Double, 0.2)
        XCTAssertEqual(encodedPayload["total_time"] as? Double, 0.5)
        XCTAssertEqual(encodedPayload["model_id"] as? String, "host-loaded-model")
        XCTAssertEqual(encodedPayload["model_path"] as? String, "/models/qwen")

        let decoded = try JSONDecoder().decode(JointInferenceEventMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func testCancelMessageUsesStableWireShape() throws {
        let payload = JointInferenceCancelPayload(
            requestID: "req-001",
            peerID: "ios-peer",
            reason: "user_cancelled"
        )
        let message = JointInferenceCancelMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "joint_inference_cancel")

        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(encodedPayload["schema_version"] as? String, JointInferenceCancelPayload.schemaVersion)
        XCTAssertEqual(encodedPayload["request_id"] as? String, "req-001")
        XCTAssertEqual(encodedPayload["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(encodedPayload["reason"] as? String, "user_cancelled")

        let decoded = try JSONDecoder().decode(JointInferenceCancelMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func testEventFrameDecoderIgnoresOtherOps() throws {
        let event = JointInferenceEventPayload(
            requestID: "req-001",
            type: .token,
            sequence: 1,
            token: "你",
            tokenID: 42
        )
        let eventFrame = try JSONEncoder().encode(JointInferenceEventMessage(payload: event))
        XCTAssertEqual(JointInferenceFrameDecoder.decodeEventFrame(eventFrame), event)

        let otherFrame = try JSONSerialization.data(withJSONObject: [
            "op": "classify_response",
            "payload": ["request_id": "req-001"],
        ] as [String: Any])
        XCTAssertNil(JointInferenceFrameDecoder.decodeEventFrame(otherFrame))
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientStreamsTokenEventsAndCompletesFromTokenAccumulator() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            configuration: .init(timeoutSeconds: 2),
            sendRequest: { payload in
                harness.send(payload)
            },
            registerEventHandler: { handler in
                harness.register(handler)
            }
        )
        let payload = JointInferenceRequestPayload(
            requestID: "req-client-001",
            peerID: "ios-peer",
            modelID: "Qwen3.5-9B-4bit",
            messages: [JointInferenceMessage(role: "user", content: "hello")],
            maxTokens: 32,
            temperature: 0.2,
            enableThinking: false,
            useNeuralImprint: true,
            routeReason: "unit_test"
        )

        let task = Task {
            try await client.generate(payload) { event in
                harness.recordEvent(event)
            }
        }
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(harness.sent.map(\.requestID), ["req-client-001"])
        harness.emit(.init(requestID: "req-client-001", type: .accepted, sequence: 0))
        harness.emit(.init(requestID: "req-client-001", type: .token, sequence: 1, token: "hel"))
        harness.emit(.init(requestID: "req-client-001", type: .token, sequence: 2, token: "lo"))
        harness.emit(.init(requestID: "req-client-001", type: .complete, sequence: 3, totalTokens: 2))

        let output = try await task.value
        XCTAssertEqual(output, "hello")
        XCTAssertEqual(harness.events, [.accepted, .token, .token, .complete])
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientCompletesFromFullTextWhenProvided() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            sendRequest: { payload in harness.send(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        )
        let payload = JointInferenceRequestPayload(
            requestID: "req-client-002",
            messages: [JointInferenceMessage(role: "user", content: "hello")]
        )

        let task = Task { try await client.generate(payload, timeoutSeconds: 2) }
        try await Task.sleep(nanoseconds: 20_000_000)
        harness.emit(.init(requestID: "req-client-002", type: .token, token: "partial"))
        harness.emit(.init(requestID: "req-client-002", type: .complete, fullText: "authoritative"))

        let output = try await task.value
        XCTAssertEqual(output, "authoritative")
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientQueuesSecondRequestUntilFirstCompletes() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            configuration: .init(timeoutSeconds: 2, maxConcurrentRequests: 1, maxQueuedRequests: 2),
            sendRequest: { payload in harness.send(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        )
        let first = JointInferenceRequestPayload(
            requestID: "req-queue-1",
            messages: [JointInferenceMessage(role: "user", content: "first")]
        )
        let second = JointInferenceRequestPayload(
            requestID: "req-queue-2",
            messages: [JointInferenceMessage(role: "user", content: "second")]
        )

        let firstTask = Task {
            try await client.generate(first) { event in harness.recordEvent(event) }
        }
        try await Task.sleep(nanoseconds: 20_000_000)
        let secondTask = Task {
            try await client.generate(second) { event in harness.recordEvent(event) }
        }
        try await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(harness.sent.map(\.requestID), ["req-queue-1"])
        XCTAssertTrue(harness.events.contains(.queued))

        harness.emit(.init(requestID: "req-queue-1", type: .complete, fullText: "first done"))
        try await Task.sleep(nanoseconds: 20_000_000)
        XCTAssertEqual(harness.sent.map(\.requestID), ["req-queue-1", "req-queue-2"])

        harness.emit(.init(requestID: "req-queue-2", type: .complete, fullText: "second done"))
        let firstOutput = try await firstTask.value
        let secondOutput = try await secondTask.value
        XCTAssertEqual(firstOutput, "first done")
        XCTAssertEqual(secondOutput, "second done")
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientSendsCancelForActiveRequestCancellation() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            configuration: .init(timeoutSeconds: 2),
            sendRequest: { payload in harness.send(payload) },
            sendCancel: { payload in harness.cancel(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        )
        let payload = JointInferenceRequestPayload(
            requestID: "req-cancel-active",
            peerID: "ios-peer",
            messages: [JointInferenceMessage(role: "user", content: "hello")]
        )

        let task = Task { try await client.generate(payload) }
        try await Task.sleep(nanoseconds: 20_000_000)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch let error as JointInferenceClientError {
            guard case .cancelled = error else {
                XCTFail("Expected cancellation, got \(error)")
                return
            }
        }
        XCTAssertEqual(harness.cancels.map(\.requestID), ["req-cancel-active"])
        XCTAssertEqual(harness.cancels.first?.peerID, "ios-peer")
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientIgnoresEventsForOtherRequests() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            configuration: .init(timeoutSeconds: 2),
            sendRequest: { payload in harness.send(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        )
        let payload = JointInferenceRequestPayload(
            requestID: "req-client-003",
            messages: [JointInferenceMessage(role: "user", content: "hello")]
        )

        let task = Task { try await client.generate(payload) }
        try await Task.sleep(nanoseconds: 20_000_000)
        harness.emit(.init(requestID: "other-request", type: .complete, fullText: "wrong"))
        harness.emit(.init(requestID: "req-client-003", type: .complete, fullText: "right"))

        let output = try await task.value
        XCTAssertEqual(output, "right")
    }

    @available(iOS 17.0, macOS 14.0, *)
    func testClientTimesOutPendingRequest() async throws {
        let harness = ClientHarness()
        let client = JointInferenceClient(
            sendRequest: { payload in harness.send(payload) },
            registerEventHandler: { handler in harness.register(handler) }
        )
        let payload = JointInferenceRequestPayload(
            requestID: "req-client-timeout",
            messages: [JointInferenceMessage(role: "user", content: "hello")]
        )

        do {
            _ = try await client.generate(payload, timeoutSeconds: 0.02)
            XCTFail("Expected timeout")
        } catch let error as JointInferenceClientError {
            guard case .timedOut(let requestID, _) = error else {
                XCTFail("Expected timeout, got \(error)")
                return
            }
            XCTAssertEqual(requestID, "req-client-timeout")
        }
    }

    func testEventStatusLabelUsesHumanReadableText() {
        XCTAssertEqual(
            JointInferenceEventPayload(requestID: "r", type: .complete, totalTokens: 7).statusLabel,
            "Mac completed · 7 tokens"
        )
        XCTAssertEqual(
            JointInferenceEventPayload(requestID: "r", type: .status, message: "Loading").statusLabel,
            "Loading"
        )
        XCTAssertEqual(
            JointInferenceEventPayload(requestID: "r", type: .queued).statusLabel,
            "Queued for Mac inference"
        )
    }
}
