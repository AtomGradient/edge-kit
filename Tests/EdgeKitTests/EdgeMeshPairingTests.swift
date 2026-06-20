// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class EdgeMeshPairingTests: XCTestCase {

    func testFrameCodec_RoundTrip_Small() throws {
        let payload = Data("hello".utf8)
        let framed = try FrameCodec.encode(payload)
        XCTAssertEqual(framed.count, 4 + payload.count)

        let buffer = FrameCodec.Buffer()
        buffer.append(framed)
        let decoded = try buffer.pop()
        XCTAssertEqual(decoded, payload)
        XCTAssertNil(try buffer.pop())
    }

    func testFrameCodec_Buffer_HandlesPartialChunks() throws {
        let payload = Data(repeating: 0xAB, count: 1000)
        let framed = try FrameCodec.encode(payload)

        let buffer = FrameCodec.Buffer()
        var i = 0
        while i < framed.count {
            let end = min(i + 3, framed.count)
            buffer.append(framed.subdata(in: i..<end))
            i = end
        }
        let decoded = try buffer.pop()
        XCTAssertEqual(decoded, payload)
    }

    func testFrameCodec_RejectsOversizedFrames() {
        let buffer = FrameCodec.Buffer()
        var header = Data(count: 4)
        let bigLen = UInt32(FrameCodec.maxFrameBytes + 1).bigEndian
        header.withUnsafeMutableBytes { $0.baseAddress!.storeBytes(of: bigLen, as: UInt32.self) }
        buffer.append(header)
        XCTAssertThrowsError(try buffer.pop())
    }

    func testQRCodec_EncodeDecodeRoundTrip() throws {
        let nonce = QRPairingPayload.generateNonce()
        let payload = QRPairingPayload(
            peerId: "mac-studio-abc",
            displayName: "Alex's Mac Studio",
            role: .brain,
            endpoint: .init(serviceType: "_edgemesh._tcp", serviceName: "edgestudio-mac-studio-abc", ipv4: "192.168.0.109", port: 18843),
            certFingerprint: String(repeating: "a", count: 64),
            nonce: nonce,
            expiresAt: Int64(Date().timeIntervalSince1970) + 60
        )
        let json = try payload.jsonString()
        let decoded = try QRPairingPayload.decode(jsonString: json)
        XCTAssertEqual(decoded, payload)
    }

    func testQRCodec_ValidatesFingerprint() {
        let payload = QRPairingPayload(
            peerId: "p",
            displayName: "d",
            role: .brain,
            endpoint: .init(serviceType: "_edgemesh._tcp", serviceName: "n", ipv4: nil, port: 1),
            certFingerprint: "NOT-HEX",
            nonce: QRPairingPayload.generateNonce(),
            expiresAt: Int64(Date().timeIntervalSince1970) + 60
        )
        XCTAssertThrowsError(try payload.validateShape())
    }

    func testQRCodec_ExpiryDetection() {
        let expired = QRPairingPayload(
            peerId: "p",
            displayName: "d",
            role: .brain,
            endpoint: .init(serviceType: "_edgemesh._tcp", serviceName: "n", ipv4: nil, port: 1),
            certFingerprint: String(repeating: "0", count: 64),
            nonce: QRPairingPayload.generateNonce(),
            expiresAt: 0
        )
        XCTAssertTrue(expired.isExpired())

        let fresh = QRPairingPayload(
            peerId: "p",
            displayName: "d",
            role: .brain,
            endpoint: .init(serviceType: "_edgemesh._tcp", serviceName: "n", ipv4: nil, port: 1),
            certFingerprint: String(repeating: "0", count: 64),
            nonce: QRPairingPayload.generateNonce(),
            expiresAt: Int64(Date().timeIntervalSince1970) + 60
        )
        XCTAssertFalse(fresh.isExpired())
    }

    func testQRCodec_PinDerivation_Deterministic() {
        let nonce = "AAAAAAAAAAAAAAAAAAAAAA"
        let payload = QRPairingPayload(
            peerId: "p", displayName: "d", role: .brain,
            endpoint: .init(serviceType: "_edgemesh._tcp", serviceName: "n", ipv4: nil, port: 1),
            certFingerprint: String(repeating: "0", count: 64),
            nonce: nonce,
            expiresAt: Int64(Date().timeIntervalSince1970) + 60
        )
        let pin1 = payload.pin
        let pin2 = payload.pin
        XCTAssertEqual(pin1, pin2)
        XCTAssertEqual(pin1.count, 6)
    }

    func testTrustStore_InsertLookupRevoke() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("truststore_\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let store = try TrustStore(url: tmp)

        let fp = String(repeating: "a", count: 64)
        let peer = TrustStore.TrustedPeer(
            peerId: "mac-1",
            displayName: "Mac",
            fingerprint: fp,
            role: "brain",
            pairedAt: Date(),
            lastSeenAt: nil,
            revoked: false
        )
        try store.upsert(peer)

        let found = try store.lookup(peerId: "mac-1")
        XCTAssertEqual(found?.peerId, "mac-1")
        XCTAssertEqual(found?.fingerprint, fp)
        XCTAssertEqual(found?.revoked, false)

        let byFp = try store.lookup(fingerprint: fp)
        XCTAssertEqual(byFp?.peerId, "mac-1")

        let verified = try store.verify(fingerprint: fp)
        XCTAssertNotNil(verified)

        try store.revoke(peerId: "mac-1")
        XCTAssertThrowsError(try store.verify(fingerprint: fp)) { err in
            guard case MeshError.peerRevoked = err else {
                XCTFail("expected peerRevoked, got \(err)")
                return
            }
        }

        XCTAssertNil(try store.lookup(fingerprint: String(repeating: "b", count: 64)))
    }

    func testTrustStore_ListAll_OrdersByPairedAtDesc() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("truststore_\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: tmp) }
        let store = try TrustStore(url: tmp)

        let a = TrustStore.TrustedPeer(peerId: "a", displayName: "A", fingerprint: String(repeating: "a", count: 64), role: "peer", pairedAt: Date().addingTimeInterval(-100), lastSeenAt: nil, revoked: false)
        let b = TrustStore.TrustedPeer(peerId: "b", displayName: "B", fingerprint: String(repeating: "b", count: 64), role: "peer", pairedAt: Date(), lastSeenAt: nil, revoked: false)
        try store.upsert(a)
        try store.upsert(b)

        let list = try store.listAll()
        XCTAssertEqual(list.count, 2)
        XCTAssertEqual(list.first?.peerId, "b")
    }

    func testCertificateManager_GenerateDeterministicFingerprint() throws {
        let id = try CertificateManager.generate(peerId: "test-peer", displayName: "Test")
        XCTAssertEqual(id.fingerprint.count, 64)
        XCTAssertTrue(id.fingerprint.allSatisfy { c in
            ("0"..."9").contains(c) || ("a"..."f").contains(c)
        })
        XCTAssertEqual(CertificateManager.fingerprint(of: id.certificateDER), id.fingerprint)

        let pem = id.privateKeyPEM
        XCTAssertTrue(pem.contains("PRIVATE KEY"))
    }

    func testCertificateManager_DistinctIdentitiesHaveDistinctFingerprints() throws {
        let a = try CertificateManager.generate(peerId: "a", displayName: "A")
        let b = try CertificateManager.generate(peerId: "b", displayName: "B")
        XCTAssertNotEqual(a.fingerprint, b.fingerprint)
    }

    func testWireMessage_RoundTrip_PairHello() throws {
        let hello = PairHelloMessage(peerId: "p", displayName: "D", certFingerprint: String(repeating: "a", count: 64), nonce: "nonce_16_bytes__")
        let msg = WireMessage.pairHello(hello)
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(WireMessage.self, from: data)
        XCTAssertEqual(msg, decoded)
    }

    func testWireMessage_RoundTrip_PairAck() throws {
        let ack = PairAckMessage(peerId: "p", displayName: "D", certFingerprint: String(repeating: "b", count: 64))
        let msg = WireMessage.pairAck(ack)
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(WireMessage.self, from: data)
        XCTAssertEqual(msg, decoded)
    }

    func testPeerTrustDeletedMessage_EncodesStableOperation() throws {
        let payload = PeerTrustDeletedPayload(
            peerID: "ios-peer",
            reason: "user_deleted_peer",
            deletedAtUnixSeconds: 1_779_700_000
        )
        let data = try JSONEncoder().encode(PeerTrustDeletedMessage(payload: payload))
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let body = obj?["payload"] as? [String: Any]

        XCTAssertEqual(obj?["op"] as? String, PeerTrustOps.deleted)
        XCTAssertEqual(body?["schema_version"] as? String, PeerTrustDeletedPayload.schemaVersion)
        XCTAssertEqual(body?["peer_id"] as? String, "ios-peer")
        XCTAssertEqual(body?["reason"] as? String, "user_deleted_peer")
    }

    func testPeerTrustDeleteClient_DecodesAckFrame() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "op": PeerTrustOps.deletedAck,
            "payload": [
                "ok": true,
                "peer_id": "ios-peer",
                "was_known": true,
            ],
        ])

        let ack = PeerTrustDeleteClient.decodeAckFrame(data)

        XCTAssertEqual(ack, PeerTrustDeletedAckPayload(
            ok: true,
            peerID: "ios-peer",
            wasKnown: true
        ))
    }

    func testPeerTrustDeleteClient_IgnoresNonAckFrame() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "op": "ping",
            "payload": ["timestamp": 1_779_700_000],
        ])

        XCTAssertNil(PeerTrustDeleteClient.decodeAckFrame(data))
    }
}
