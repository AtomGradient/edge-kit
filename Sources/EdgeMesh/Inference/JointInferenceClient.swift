// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum JointInferenceClientError: Error, Equatable, LocalizedError {
    case emptyRequest
    case duplicateRequest(String)
    case queueFull(limit: Int)
    case timedOut(requestID: String, seconds: TimeInterval)
    case serverError(String)
    case cancelled(String)

    public var errorDescription: String? {
        switch self {
        case .emptyRequest:
            return "joint inference request requires messages or prompt"
        case .duplicateRequest(let requestID):
            return "joint inference request already pending: \(requestID)"
        case .queueFull(let limit):
            return "joint inference queue is full (limit \(limit))"
        case .timedOut(_, let seconds):
            return "joint inference timed out after \(Int(seconds))s"
        case .serverError(let message):
            return message
        case .cancelled(let message):
            return message
        }
    }
}

@available(iOS 17.0, macOS 14.0, *)
public final class JointInferenceClient: @unchecked Sendable {
    public typealias EventHandler = @Sendable (JointInferenceEventPayload) -> Void
    public typealias RequestSender = @Sendable (JointInferenceRequestPayload) throws -> Void
    public typealias CancelSender = @Sendable (JointInferenceCancelPayload) throws -> Void
    public typealias EventRegistrar = (@escaping EventHandler) -> Void

    public struct Configuration: Sendable, Equatable {
        public var timeoutSeconds: TimeInterval
        public var maxConcurrentRequests: Int
        public var maxQueuedRequests: Int

        public init(
            timeoutSeconds: TimeInterval = 180,
            maxConcurrentRequests: Int = 1,
            maxQueuedRequests: Int = 8
        ) {
            self.timeoutSeconds = timeoutSeconds
            self.maxConcurrentRequests = max(1, maxConcurrentRequests)
            self.maxQueuedRequests = max(0, maxQueuedRequests)
        }
    }

    private final class PendingRequest: @unchecked Sendable {
        let payload: JointInferenceRequestPayload
        let timeout: TimeInterval
        let continuation: CheckedContinuation<String, Error>
        let onEvent: EventHandler?
        var tokenParts: [String] = []
        var isActive = false
        var timeoutTask: Task<Void, Never>?

        init(
            payload: JointInferenceRequestPayload,
            timeout: TimeInterval,
            continuation: CheckedContinuation<String, Error>,
            onEvent: EventHandler?
        ) {
            self.payload = payload
            self.timeout = timeout
            self.continuation = continuation
            self.onEvent = onEvent
        }
    }

    private let sendRequest: RequestSender
    private let sendCancel: CancelSender
    private let configuration: Configuration
    private let lock = NSLock()
    private var pending: [String: PendingRequest] = [:]
    private var activeRequestIDs: Set<String> = []
    private var queuedRequestIDs: [String] = []

    public init(
        configuration: Configuration = Configuration(),
        sendRequest: @escaping RequestSender,
        sendCancel: @escaping CancelSender = { _ in },
        registerEventHandler: EventRegistrar
    ) {
        self.configuration = configuration
        self.sendRequest = sendRequest
        self.sendCancel = sendCancel
        registerEventHandler { [weak self] event in
            self?.handle(event)
        }
    }

    public convenience init(
        connection: MeshConnection,
        configuration: Configuration = Configuration()
    ) {
        self.init(
            configuration: configuration,
            sendRequest: { payload in
                try connection.sendJointInferenceRequest(payload)
            },
            sendCancel: { payload in
                try connection.sendJointInferenceCancel(payload)
            },
            registerEventHandler: { handler in
                connection.onFrame { data in
                    guard let event = JointInferenceFrameDecoder.decodeEventFrame(data) else {
                        return
                    }
                    handler(event)
                }
            }
        )
    }

    @discardableResult
    public func generate(
        _ payload: JointInferenceRequestPayload,
        timeoutSeconds: TimeInterval? = nil,
        onEvent: EventHandler? = nil
    ) async throws -> String {
        guard !payload.messages.isEmpty || !(payload.prompt ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw JointInferenceClientError.emptyRequest
        }

        let timeout = timeoutSeconds ?? configuration.timeoutSeconds
        let requestID = payload.requestID

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let pendingRequest = PendingRequest(
                    payload: payload,
                    timeout: timeout,
                    continuation: continuation,
                    onEvent: onEvent
                )
                let admission = admit(pendingRequest)
                switch admission {
                case .duplicate:
                    continuation.resume(
                        throwing: JointInferenceClientError.duplicateRequest(requestID)
                    )
                case .queueFull:
                    continuation.resume(
                        throwing: JointInferenceClientError.queueFull(
                            limit: configuration.maxQueuedRequests
                        )
                    )
                case .queued:
                    onEvent?(
                        JointInferenceEventPayload(
                            requestID: requestID,
                            type: .queued,
                            sequence: 0,
                            message: "Queued on device"
                        )
                    )
                case .start:
                    start(pendingRequest)
                }
            }
        } onCancel: {
            cancel(
                requestID: requestID,
                reason: "joint inference cancelled",
                result: .failure(
                    JointInferenceClientError.cancelled("joint inference cancelled")
                )
            )
        }
    }

    private enum AdmissionDecision {
        case duplicate
        case queueFull
        case queued
        case start
    }

    private func admit(_ request: PendingRequest) -> AdmissionDecision {
        lock.lock()
        defer { lock.unlock() }

        let requestID = request.payload.requestID
        if pending[requestID] != nil {
            return .duplicate
        }

        if activeRequestIDs.count >= configuration.maxConcurrentRequests,
           queuedRequestIDs.count >= configuration.maxQueuedRequests {
            return .queueFull
        }

        pending[requestID] = request
        if activeRequestIDs.count < configuration.maxConcurrentRequests {
            request.isActive = true
            activeRequestIDs.insert(requestID)
            return .start
        }
        queuedRequestIDs.append(requestID)
        return .queued
    }

    private func start(_ request: PendingRequest) {
        do {
            try sendRequest(request.payload)
        } catch {
            finish(requestID: request.payload.requestID, result: .failure(error))
            return
        }

        request.timeoutTask = Task { [weak self] in
            let nanoseconds = UInt64(max(request.timeout, 0) * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)
            self?.cancel(
                requestID: request.payload.requestID,
                reason: "joint inference timed out",
                result: .failure(
                    JointInferenceClientError.timedOut(
                        requestID: request.payload.requestID,
                        seconds: request.timeout
                    )
                )
            )
        }
    }

    private func handle(_ event: JointInferenceEventPayload) {
        var callback: EventHandler?
        var completion: Result<String, Error>?
        lock.lock()
        if let pendingRequest = pending[event.requestID] {
            callback = pendingRequest.onEvent
            if event.type == .token, let token = event.token, !token.isEmpty {
                pendingRequest.tokenParts.append(token)
            }
            switch event.type {
            case .complete:
                let text = event.fullText ?? pendingRequest.tokenParts.joined()
                completion = .success(text)
            case .error:
                completion = .failure(
                    JointInferenceClientError.serverError(
                        event.error ?? "joint inference failed"
                    )
                )
            case .cancelled:
                completion = .failure(
                    JointInferenceClientError.cancelled(
                        event.error ?? "joint inference cancelled"
                    )
                )
            case .queued, .accepted, .status, .token:
                break
            }
        }
        lock.unlock()

        callback?(event)

        if let completion {
            finish(requestID: event.requestID, result: completion)
        }
    }

    private func finish(
        requestID: String,
        result: Result<String, Error>
    ) {
        var continuation: CheckedContinuation<String, Error>?
        var nextRequests: [PendingRequest] = []
        lock.lock()
        if let pendingRequest = pending.removeValue(forKey: requestID) {
            pendingRequest.timeoutTask?.cancel()
            activeRequestIDs.remove(requestID)
            queuedRequestIDs.removeAll { $0 == requestID }
            continuation = pendingRequest.continuation
            nextRequests = admitQueuedLocked()
        }
        lock.unlock()

        guard let continuation else { return }
        switch result {
        case .success(let text):
            continuation.resume(returning: text)
        case .failure(let error):
            continuation.resume(throwing: error)
        }

        for request in nextRequests {
            start(request)
        }
    }

    private func cancel(
        requestID: String,
        reason: String,
        result: Result<String, Error>
    ) {
        var cancelPayload: JointInferenceCancelPayload?
        lock.lock()
        if let request = pending[requestID], request.isActive {
            cancelPayload = JointInferenceCancelPayload(
                requestID: requestID,
                peerID: request.payload.peerID,
                reason: reason
            )
        }
        lock.unlock()

        if let cancelPayload {
            try? sendCancel(cancelPayload)
        }
        finish(requestID: requestID, result: result)
    }

    private func admitQueuedLocked() -> [PendingRequest] {
        var requests: [PendingRequest] = []
        while activeRequestIDs.count < configuration.maxConcurrentRequests,
              !queuedRequestIDs.isEmpty {
            let nextID = queuedRequestIDs.removeFirst()
            guard let request = pending[nextID] else {
                continue
            }
            request.isActive = true
            activeRequestIDs.insert(nextID)
            requests.append(request)
        }
        return requests
    }

}

public extension JointInferenceEventPayload {
    var statusLabel: String {
        switch type {
        case .queued:
            return message ?? "Queued for Mac inference"
        case .accepted:
            return "Mac accepted request"
        case .status:
            return message ?? "Mac is preparing"
        case .token:
            return "Mac is generating"
        case .complete:
            if let totalTokens {
                return "Mac completed · \(totalTokens) tokens"
            }
            return "Mac completed"
        case .error:
            return error ?? "Mac inference failed"
        case .cancelled:
            return "Mac inference cancelled"
        }
    }
}
