// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Network

/// Coordinates the client-side (iOS) pairing handshake.
///
/// Flow:
/// 1. User scans QR (or inputs PIN, which the UI layer exchanges for a QR payload via
///    the brain's `/api/mesh/pair/pin` endpoint). Either way we end up with a validated
///    `QRPairingPayload`.
/// 2. `completePairing(with:identity:trustStore:)` opens an mTLS connection to the
///    payload's endpoint with `bootstrapTrust=true` (the cert fingerprint from the QR
///    is the ground truth, not the TrustStore which is still empty for this peer).
/// 3. Upon TLS readiness, we send `pair_hello`, wait for `pair_ack`.
/// 4. On success, we insert the peer into the TrustStore and return a `MeshNode`.
///
/// This service is single-shot per pairing attempt. For a new pairing, instantiate a new one.
@available(iOS 17.0, macOS 14.0, *)
public final class PairingService: @unchecked Sendable {

    public enum Phase: Equatable {
        case idle
        case connecting
        case mTLSReady
        case awaitingAck
        case done(peerId: String)
        case failed(String)
    }

    public struct Result {
        public let peer: TrustStore.TrustedPeer
        public let node: MeshNode
    }

    private let localPeerId: String
    private let localDisplayName: String
    private let identity: CertificateManager.Identity
    private let trustStore: TrustStore

    private var connection: MeshConnection?
    private var continuation: CheckedContinuation<Result, Error>?
    private var phase: Phase = .idle
    private let lock = NSLock()

    public var onPhaseChange: ((Phase) -> Void)?

    public init(
        localPeerId: String,
        localDisplayName: String,
        identity: CertificateManager.Identity,
        trustStore: TrustStore
    ) {
        self.localPeerId = localPeerId
        self.localDisplayName = localDisplayName
        self.identity = identity
        self.trustStore = trustStore
    }

    /// Run the full pairing handshake with a payload received from QR or PIN.
    public func completePairing(with payload: QRPairingPayload) async throws -> Result {
        try payload.validateShape()
        if payload.isExpired() {
            throw MeshError.pairingExpired
        }

        let host = payload.endpoint.ipv4 ?? "\(payload.endpoint.serviceName).local"
        let connConfig = MeshConnection.Config(
            host: host,
            port: payload.endpoint.port,
            expectedFingerprint: payload.certFingerprint,
            identity: identity,
            bootstrapTrust: true
        )
        let conn = MeshConnection(config: connConfig, queueLabel: "com.edgemesh.pairing")
        self.connection = conn

        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont

            conn.onStateChange { [weak self] state in
                guard let self = self else { return }
                switch state {
                case .ready:
                    self.advance(.mTLSReady)
                    self.sendHello(nonce: payload.nonce)
                case .failed(let reason):
                    self.resolveFailure(MeshError.tlsSetupFailed(reason))
                case .cancelled:
                    self.resolveFailure(MeshError.connectionFailed(peer: payload.peerId, reason: "cancelled"))
                default:
                    break
                }
            }

            conn.onFrame { [weak self] frame in
                guard let self = self else { return }
                self.handleFrame(frame, payload: payload)
            }

            do {
                self.advance(.connecting)
                try conn.connect()
            } catch {
                self.resolveFailure(error)
            }
        }
    }

    private func sendHello(nonce: String) {
        let hello = PairHelloMessage(
            peerId: localPeerId,
            displayName: localDisplayName,
            certFingerprint: identity.fingerprint,
            nonce: nonce
        )
        let msg = WireMessage.pairHello(hello)
        do {
            try connection?.sendJSON(msg)
            self.advance(.awaitingAck)
        } catch {
            self.resolveFailure(error)
        }
    }

    private func handleFrame(_ frame: Data, payload: QRPairingPayload) {
        let decoder = JSONDecoder()
        guard let message = try? decoder.decode(WireMessage.self, from: frame) else {
            return
        }
        switch message {
        case .pairAck(let ack):
            finalizePairing(ack: ack, payload: payload)
        case .error(let err):
            resolveFailure(MeshError.pairingInvalid(err.message))
        case .pairHello, .ping:
            break
        }
    }

    private func finalizePairing(ack: PairAckMessage, payload: QRPairingPayload) {
        guard ack.peerId == payload.peerId else {
            resolveFailure(MeshError.pairingInvalid("peerId mismatch"))
            return
        }
        guard ack.certFingerprint == payload.certFingerprint else {
            resolveFailure(MeshError.fingerprintMismatch(
                expected: payload.certFingerprint,
                actual: ack.certFingerprint
            ))
            return
        }

        let now = Date()
        let trusted = TrustStore.TrustedPeer(
            peerId: ack.peerId,
            displayName: ack.displayName,
            fingerprint: ack.certFingerprint,
            role: payload.role.rawValue,
            pairedAt: now,
            lastSeenAt: now,
            revoked: false
        )

        do {
            try trustStore.upsert(trusted)
        } catch {
            resolveFailure(error)
            return
        }

        let node = MeshNode(
            id: ack.peerId,
            displayName: ack.displayName,
            capability: payload.role == .brain ? .both : .inference,
            deviceProfile: .init(chipName: "unknown", totalRAMGB: 0, availableRAMGB: 0, bandwidthGBs: 0, thermalState: .nominal),
            endpoint: .init(host: payload.endpoint.ipv4 ?? payload.endpoint.serviceName, port: payload.endpoint.port)
        )

        self.advance(.done(peerId: ack.peerId))
        let result = Result(peer: trusted, node: node)
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume(returning: result)
    }

    private func advance(_ newPhase: Phase) {
        lock.lock()
        phase = newPhase
        lock.unlock()
        onPhaseChange?(newPhase)
    }

    private func resolveFailure(_ error: Error) {
        advance(.failed(String(describing: error)))
        lock.lock()
        let c = continuation
        continuation = nil
        lock.unlock()
        c?.resume(throwing: error)
    }

    /// Cancel an ongoing pairing.
    public func cancel() {
        connection?.cancel()
        resolveFailure(MeshError.pairingInvalid("user cancelled"))
    }
}

/// Unified wire envelope for pairing / keepalive traffic.
/// More op types (event_upload, adapter_*) live in their dedicated handlers but share this frame.
public enum WireMessage: Codable, Equatable {
    case pairHello(PairHelloMessage)
    case pairAck(PairAckMessage)
    case ping(PingMessage)
    case error(ErrorMessage)

    private enum CodingKeys: String, CodingKey {
        case op
        case payload
    }
    private enum Op: String, Codable {
        case pairHello = "pair_hello"
        case pairAck   = "pair_ack"
        case ping
        case error
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let op = try c.decode(Op.self, forKey: .op)
        switch op {
        case .pairHello: self = .pairHello(try c.decode(PairHelloMessage.self, forKey: .payload))
        case .pairAck:   self = .pairAck(try c.decode(PairAckMessage.self, forKey: .payload))
        case .ping:      self = .ping(try c.decode(PingMessage.self, forKey: .payload))
        case .error:     self = .error(try c.decode(ErrorMessage.self, forKey: .payload))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .pairHello(let v): try c.encode(Op.pairHello, forKey: .op); try c.encode(v, forKey: .payload)
        case .pairAck(let v):   try c.encode(Op.pairAck, forKey: .op);   try c.encode(v, forKey: .payload)
        case .ping(let v):      try c.encode(Op.ping, forKey: .op);      try c.encode(v, forKey: .payload)
        case .error(let v):     try c.encode(Op.error, forKey: .op);     try c.encode(v, forKey: .payload)
        }
    }
}

/// Optional stats payload piggy-backed on keepalive ping.
///
/// All fields are optional so older clients can keep sending `PingMessage(stats: nil)`.
/// The payload lets a sensor report local event/fact counts without opening a second control channel.
public struct PingStats: Codable, Equatable {
    public let appId: String?
    public let eventStoreTotal: Int?
    public let factsClassified: Int?
    public let factsRawUnclassified: Int?

    public init(
        appId: String? = nil,
        eventStoreTotal: Int? = nil,
        factsClassified: Int? = nil,
        factsRawUnclassified: Int? = nil
    ) {
        self.appId = appId
        self.eventStoreTotal = eventStoreTotal
        self.factsClassified = factsClassified
        self.factsRawUnclassified = factsRawUnclassified
    }
}

public struct PingMessage: Codable, Equatable {
    public let timestamp: Int64
    /// Optional sensor-side stats.
    public let stats: PingStats?
    public init(
        timestamp: Int64 = Int64(Date().timeIntervalSince1970 * 1000),
        stats: PingStats? = nil
    ) {
        self.timestamp = timestamp
        self.stats = stats
    }
}

public struct ErrorMessage: Codable, Equatable {
    public let code: String
    public let message: String
    public init(code: String, message: String) {
        self.code = code
        self.message = message
    }
}
