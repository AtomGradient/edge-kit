// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum JointInferenceTransportOps {
    public static let request = "joint_inference_request"
    public static let event = "joint_inference_event"
    public static let cancel = "joint_inference_cancel"
}

public enum JointInferenceEventType: String, Codable, Equatable, Sendable {
    case queued
    case accepted
    case status
    case token
    case complete
    case error
    case cancelled
}

public struct JointInferenceMessage: Codable, Equatable, Sendable {
    public var role: String
    public var content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct JointInferenceRequestPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.joint_inference_request.v1"

    public var schemaVersion: String
    public var requestID: String
    public var conversationID: String?
    public var peerID: String?
    public var modelID: String?
    public var prompt: String?
    public var messages: [JointInferenceMessage]
    public var maxTokens: Int
    public var temperature: Double
    public var topK: Int?
    public var topP: Double?
    public var enableThinking: Bool?
    public var useNeuralImprint: Bool
    public var routeReason: String?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case requestID = "request_id"
        case conversationID = "conversation_id"
        case peerID = "peer_id"
        case modelID = "model_id"
        case prompt
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topK = "top_k"
        case topP = "top_p"
        case enableThinking = "enable_thinking"
        case useNeuralImprint = "use_neural_imprint"
        case legacyUsePersonaKV = "use_persona_kv"
        case routeReason = "route_reason"
    }

    public init(
        requestID: String = UUID().uuidString,
        conversationID: String? = nil,
        peerID: String? = nil,
        modelID: String? = nil,
        prompt: String? = nil,
        messages: [JointInferenceMessage],
        maxTokens: Int = 512,
        temperature: Double = 0.2,
        topK: Int? = nil,
        topP: Double? = nil,
        enableThinking: Bool? = nil,
        useNeuralImprint: Bool = false,
        routeReason: String? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.requestID = requestID
        self.conversationID = conversationID
        self.peerID = peerID
        self.modelID = modelID
        self.prompt = prompt
        self.messages = messages
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.enableThinking = enableThinking
        self.useNeuralImprint = useNeuralImprint
        self.routeReason = routeReason
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.schemaVersion = try container.decodeIfPresent(String.self, forKey: .schemaVersion) ?? Self.schemaVersion
        self.requestID = try container.decode(String.self, forKey: .requestID)
        self.conversationID = try container.decodeIfPresent(String.self, forKey: .conversationID)
        self.peerID = try container.decodeIfPresent(String.self, forKey: .peerID)
        self.modelID = try container.decodeIfPresent(String.self, forKey: .modelID)
        self.prompt = try container.decodeIfPresent(String.self, forKey: .prompt)
        self.messages = try container.decodeIfPresent([JointInferenceMessage].self, forKey: .messages) ?? []
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 512
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.2
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.topP = try container.decodeIfPresent(Double.self, forKey: .topP)
        self.enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking)
        let decodedUseNeuralImprint = try container.decodeIfPresent(Bool.self, forKey: .useNeuralImprint)
        let decodedLegacyUsePersonaKV = try container.decodeIfPresent(Bool.self, forKey: .legacyUsePersonaKV)
        self.useNeuralImprint = decodedUseNeuralImprint ?? decodedLegacyUsePersonaKV ?? false
        self.routeReason = try container.decodeIfPresent(String.self, forKey: .routeReason)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(schemaVersion, forKey: .schemaVersion)
        try container.encode(requestID, forKey: .requestID)
        try container.encodeIfPresent(conversationID, forKey: .conversationID)
        try container.encodeIfPresent(peerID, forKey: .peerID)
        try container.encodeIfPresent(modelID, forKey: .modelID)
        try container.encodeIfPresent(prompt, forKey: .prompt)
        try container.encode(messages, forKey: .messages)
        try container.encode(maxTokens, forKey: .maxTokens)
        try container.encode(temperature, forKey: .temperature)
        try container.encodeIfPresent(topK, forKey: .topK)
        try container.encodeIfPresent(topP, forKey: .topP)
        try container.encodeIfPresent(enableThinking, forKey: .enableThinking)
        try container.encode(useNeuralImprint, forKey: .useNeuralImprint)
        try container.encodeIfPresent(routeReason, forKey: .routeReason)
    }
}

public struct JointInferenceRequestMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: JointInferenceRequestPayload

    public init(payload: JointInferenceRequestPayload) {
        self.op = JointInferenceTransportOps.request
        self.payload = payload
    }
}

public struct JointInferenceCancelPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.joint_inference_cancel.v1"

    public var schemaVersion: String
    public var requestID: String
    public var peerID: String?
    public var reason: String?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case requestID = "request_id"
        case peerID = "peer_id"
        case reason
    }

    public init(
        requestID: String,
        peerID: String? = nil,
        reason: String? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.requestID = requestID
        self.peerID = peerID
        self.reason = reason
    }
}

public struct JointInferenceCancelMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: JointInferenceCancelPayload

    public init(payload: JointInferenceCancelPayload) {
        self.op = JointInferenceTransportOps.cancel
        self.payload = payload
    }
}

public struct JointInferenceEventPayload: Codable, Equatable, Sendable {
    public static let schemaVersion = "edgestudio.joint_inference_event.v1"

    public var schemaVersion: String
    public var requestID: String
    public var type: JointInferenceEventType
    public var sequence: Int?
    public var message: String?
    public var token: String?
    public var tokenID: Int?
    public var fullText: String?
    public var totalTokens: Int?
    public var tokensPerSecond: Double?
    public var prefillTime: Double?
    public var totalTime: Double?
    public var modelID: String?
    public var modelPath: String?
    public var error: String?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case requestID = "request_id"
        case type
        case sequence
        case message
        case token
        case tokenID = "token_id"
        case fullText = "full_text"
        case totalTokens = "total_tokens"
        case tokensPerSecond = "tokens_per_sec"
        case prefillTime = "prefill_time"
        case totalTime = "total_time"
        case modelID = "model_id"
        case modelPath = "model_path"
        case error
    }

    public init(
        requestID: String,
        type: JointInferenceEventType,
        sequence: Int? = nil,
        message: String? = nil,
        token: String? = nil,
        tokenID: Int? = nil,
        fullText: String? = nil,
        totalTokens: Int? = nil,
        tokensPerSecond: Double? = nil,
        prefillTime: Double? = nil,
        totalTime: Double? = nil,
        modelID: String? = nil,
        modelPath: String? = nil,
        error: String? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.requestID = requestID
        self.type = type
        self.sequence = sequence
        self.message = message
        self.token = token
        self.tokenID = tokenID
        self.fullText = fullText
        self.totalTokens = totalTokens
        self.tokensPerSecond = tokensPerSecond
        self.prefillTime = prefillTime
        self.totalTime = totalTime
        self.modelID = modelID
        self.modelPath = modelPath
        self.error = error
    }
}

public struct JointInferenceEventMessage: Codable, Equatable, Sendable {
    public var op: String
    public var payload: JointInferenceEventPayload

    public init(payload: JointInferenceEventPayload) {
        self.op = JointInferenceTransportOps.event
        self.payload = payload
    }
}

@available(iOS 17.0, macOS 14.0, *)
public extension MeshConnection {
    func sendJointInferenceRequest(_ payload: JointInferenceRequestPayload) throws {
        try sendJSON(JointInferenceRequestMessage(payload: payload))
    }

    func sendJointInferenceCancel(_ payload: JointInferenceCancelPayload) throws {
        try sendJSON(JointInferenceCancelMessage(payload: payload))
    }

    func sendJointInferenceRequestAndWait(_ payload: JointInferenceRequestPayload) async throws {
        try await sendJSONAndWait(JointInferenceRequestMessage(payload: payload))
    }
}

public enum JointInferenceFrameDecoder {
    public static func decodeEventFrame(_ data: Data) -> JointInferenceEventPayload? {
        guard let envelope = try? JSONDecoder().decode(JointInferenceEventMessage.self, from: data),
              envelope.op == JointInferenceTransportOps.event else {
            return nil
        }
        return envelope.payload
    }
}
