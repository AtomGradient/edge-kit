// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import CryptoKit
import XCTest
@testable import EdgeInference
@testable import EdgeMesh

final class PersonaSourceUploadTests: XCTestCase {
    func testPersonaSourceUploadMessageUsesStableWireShape() throws {
        let toolSnapshot = try Self.toolSchemaSnapshot()
        let profileBody = "用户画像：餐饮和通勤支出稳定，偶发大额转账不应被当作奢侈消费。"
        let payload = PersonaSourceUploadPayload(
            peerID: "ios-peer",
            appID: "com.atomgradient.dogfood",
            baseModelID: "Qwen3.5-9B-4bit",
            toolSchemaSnapshot: toolSnapshot,
            profileBody: profileBody,
            rppRunID: "rpp_123",
            sourceKind: .deviceRPPProfile,
            createdAt: 1_779_600_000
        )
        let message = PersonaSourceUploadMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "persona_source_upload")

        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(encodedPayload["schema_version"] as? String, PersonaSourceUploadPayload.schemaVersion)
        XCTAssertEqual(encodedPayload["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(encodedPayload["app_id"] as? String, "com.atomgradient.dogfood")
        XCTAssertEqual(encodedPayload["base_model_id"] as? String, "Qwen3.5-9B-4bit")
        XCTAssertEqual(encodedPayload["tool_schema_sha256"] as? String, toolSnapshot.sha256)
        XCTAssertEqual(encodedPayload["profile_body"] as? String, profileBody)
        XCTAssertEqual(encodedPayload["profile_body_sha256"] as? String, Self.sha256Hex(profileBody))
        XCTAssertEqual(encodedPayload["rpp_run_id"] as? String, "rpp_123")
        XCTAssertEqual(encodedPayload["source_kind"] as? String, "device_rpp_profile")
        XCTAssertEqual(encodedPayload["created_at"] as? Double, 1_779_600_000)

        let export = try XCTUnwrap(encodedPayload["tool_schema_export"] as? [String: Any])
        XCTAssertEqual(export["schema_version"] as? String, "edgestudio.tool_schema_export.v1")
        let tools = try XCTUnwrap(export["tools"] as? [[String: Any]])
        XCTAssertEqual(tools.map { $0["name"] as? String }, ["query_expenses"])

        let decoded = try JSONDecoder().decode(PersonaSourceUploadMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func testToolSchemaOnlyPayloadOmitsProfileFields() throws {
        let toolSnapshot = try Self.toolSchemaSnapshot()
        let payload = PersonaSourceUploadPayload(
            peerID: "ios-peer",
            appID: "com.atomgradient.scaffold",
            baseModelID: "Qwen3.5-4B-6bit",
            toolSchemaSnapshot: toolSnapshot,
            sourceKind: .toolSchemaOnly,
            createdAt: 1_779_600_001
        )

        XCTAssertNil(payload.profileBody)
        XCTAssertNil(payload.profileBodySHA256)

        let data = try JSONEncoder().encode(PersonaSourceUploadMessage(payload: payload))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertNil(encodedPayload["profile_body"])
        XCTAssertNil(encodedPayload["profile_body_sha256"])
        XCTAssertNil(encodedPayload["rpp_run_id"])
        XCTAssertEqual(encodedPayload["source_kind"] as? String, "tool_schema_only")
    }

    func testAckPayloadUsesStableWireKeys() throws {
        let ack = PersonaSourceUploadAckPayload(
            ok: true,
            peerID: "ios-peer",
            sourceID: "source_abc",
            sourceSHA256: Self.sha256Hex("source"),
            message: "stored"
        )

        let data = try JSONEncoder().encode(ack)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["ok"] as? Bool, true)
        XCTAssertEqual(object["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(object["source_id"] as? String, "source_abc")
        XCTAssertEqual(object["source_sha256"] as? String, Self.sha256Hex("source"))
        XCTAssertEqual(object["message"] as? String, "stored")
    }

    func testUploadClientDecodesAckEnvelope() throws {
        let ack = PersonaSourceUploadAckPayload(
            ok: true,
            peerID: "ios-peer",
            sourceID: "source_abc",
            sourceSHA256: Self.sha256Hex("source"),
            message: "stored"
        )
        let envelope = [
            "op": PersonaSourceUploadOps.uploadAck,
            "payload": try JSONSerialization.jsonObject(with: JSONEncoder().encode(ack)),
        ] as [String: Any]
        let data = try JSONSerialization.data(withJSONObject: envelope)

        let decoded = try XCTUnwrap(PersonaSourceUploadClient.decodeAckFrame(data))
        XCTAssertEqual(decoded, ack)
    }

    func testUploadClientIgnoresNonAckEnvelope() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "op": "persona_source_upload",
            "payload": [:],
        ] as [String: Any])

        XCTAssertNil(PersonaSourceUploadClient.decodeAckFrame(data))
    }

    private static func toolSchemaSnapshot() throws -> ToolSchemaSnapshot {
        let registry = ToolRegistry()
        registry.register(
            RegisteredTool(
                metadata: ToolMetadata(
                    name: "query_expenses",
                    description: "Query local expense facts.",
                    argumentsSchema: .jsonSchema(#"{"type":"object","properties":{"month":{"type":"string"}}}"#),
                    permissions: [],
                    sensitivity: .sensitive,
                    timeoutSeconds: 3,
                    intentTags: [.aggregateFact]
                )
            ) { _ in
                "{}"
            }
        )
        return try registry.toolSchemaSnapshot()
    }

    private static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}
