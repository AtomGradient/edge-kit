// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class PersonaRPPInputUploadTests: XCTestCase {
    func testPersonaRPPInputUploadMessageUsesStableWireShape() throws {
        let records = Self.records()
        let payload = try PersonaRPPInputUploadPayload(
            peerID: "ios-peer",
            appID: "com.atomgradient.validation-app",
            baseModelID: "Qwen3.5-9B-4bit",
            sourceKind: .appFacts,
            records: records,
            createdAt: 1_779_600_000
        )
        let message = PersonaRPPInputUploadMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "persona_rpp_input_upload")

        let encodedPayload = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(encodedPayload["schema_version"] as? String, PersonaRPPInputUploadPayload.schemaVersion)
        XCTAssertEqual(encodedPayload["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(encodedPayload["app_id"] as? String, "com.atomgradient.validation-app")
        XCTAssertEqual(encodedPayload["base_model_id"] as? String, "Qwen3.5-9B-4bit")
        XCTAssertEqual(encodedPayload["source_kind"] as? String, "app_facts")
        XCTAssertEqual(encodedPayload["created_at"] as? Double, 1_779_600_000)
        XCTAssertEqual(encodedPayload["records_sha256"] as? String, Self.pythonCanonicalRecordsSHA256)

        let encodedRecords = try XCTUnwrap(encodedPayload["records"] as? [[String: Any]])
        XCTAssertEqual(encodedRecords.count, 2)
        XCTAssertEqual(encodedRecords[0]["record_id"] as? String, "fact-001")
        XCTAssertEqual(encodedRecords[0]["kind"] as? String, "habit_pattern")
        XCTAssertEqual(encodedRecords[0]["text"] as? String, "用户经常在工作日午餐外食。")
        XCTAssertEqual(encodedRecords[0]["tags"] as? [String], ["routine", "lunch"])

        let decoded = try JSONDecoder().decode(PersonaRPPInputUploadMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func testRecordsSHA256MatchesPythonCanonicalFixture() throws {
        XCTAssertEqual(
            try PersonaRPPInputUploadPayload.recordsSHA256(Self.records()),
            Self.pythonCanonicalRecordsSHA256
        )
    }

    func testAckPayloadUsesStableWireKeys() throws {
        let ack = PersonaRPPInputUploadAckPayload(
            ok: true,
            peerID: "ios-peer",
            inputID: "persona_rpp_input-abc",
            inputSHA256: String(repeating: "a", count: 64),
            recordsSHA256: Self.pythonCanonicalRecordsSHA256,
            recordCount: 2,
            message: "stored"
        )

        let data = try JSONEncoder().encode(ack)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["ok"] as? Bool, true)
        XCTAssertEqual(object["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(object["input_id"] as? String, "persona_rpp_input-abc")
        XCTAssertEqual(object["input_sha256"] as? String, String(repeating: "a", count: 64))
        XCTAssertEqual(object["records_sha256"] as? String, Self.pythonCanonicalRecordsSHA256)
        XCTAssertEqual(object["record_count"] as? Int, 2)
        XCTAssertEqual(object["message"] as? String, "stored")
    }

    func testUploadClientDecodesAckEnvelope() throws {
        let ack = PersonaRPPInputUploadAckPayload(
            ok: true,
            peerID: "ios-peer",
            inputID: "persona_rpp_input-abc",
            inputSHA256: String(repeating: "a", count: 64),
            recordsSHA256: Self.pythonCanonicalRecordsSHA256,
            recordCount: 2,
            message: "stored"
        )
        let envelope = [
            "op": PersonaRPPInputUploadOps.uploadAck,
            "payload": try JSONSerialization.jsonObject(with: JSONEncoder().encode(ack)),
        ] as [String: Any]
        let data = try JSONSerialization.data(withJSONObject: envelope)

        let decoded = try XCTUnwrap(PersonaRPPInputUploadClient.decodeAckFrame(data))
        XCTAssertEqual(decoded, ack)
    }

    func testUploadClientIgnoresNonAckEnvelope() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "op": "persona_rpp_input_upload",
            "payload": [:],
        ] as [String: Any])

        XCTAssertNil(PersonaRPPInputUploadClient.decodeAckFrame(data))
    }

    private static let pythonCanonicalRecordsSHA256 =
        "3fb393c56997b53a55ebfc4a1e6604b23e927bf5d1710d567020b5b3f8ba84c9"

    private static func records() -> [PersonaRPPInputRecord] {
        [
            PersonaRPPInputRecord(
                recordID: "fact-001",
                kind: "habit_pattern",
                text: "用户经常在工作日午餐外食。",
                tags: ["routine", "lunch"]
            ),
            PersonaRPPInputRecord(
                recordID: "fact-002",
                kind: "preference",
                text: "用户偏好咖啡和地铁通勤。"
            ),
        ]
    }
}
