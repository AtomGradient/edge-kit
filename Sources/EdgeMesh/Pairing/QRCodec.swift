// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Pairing payload carried by a QR code (primary channel) or a short PIN (fallback channel).
///
/// The payload is signed by nothing — its authenticity is bootstrapped out-of-band
/// (user physically sees the Mac screen). The fingerprint inside is the pin target:
/// once pairing succeeds, future connections verify the peer certificate against this fingerprint.
///
/// Both channels share the same `nonce` — once consumed, the partner deletes the pairing session
/// so the other channel cannot be reused.
public struct QRPairingPayload: Codable, Equatable {

    /// Schema version. Bump when making incompatible changes to payload shape.
    public let version: Int

    /// Peer's stable device identifier (stays the same across re-pairings).
    public let peerId: String

    /// Human-readable display name shown in UI ("Alex's Mac Studio").
    public let displayName: String

    /// How this peer wants to be treated in the mesh topology.
    public let role: PeerRole

    /// Endpoint to reach the peer. Typically Bonjour-resolved (service name + port),
    /// with `ipv4` as a fallback to dodge unreliable `.local` resolution on iOS.
    public let endpoint: Endpoint

    /// Hex-encoded SHA-256 of the peer's self-signed X.509 certificate (DER).
    /// Exactly 64 lowercase hex characters.
    public let certFingerprint: String

    /// 16-byte random nonce (base64url), consumed exactly once.
    public let nonce: String

    /// Unix seconds after which the payload must be refused.
    public let expiresAt: Int64

    public init(
        version: Int = 1,
        peerId: String,
        displayName: String,
        role: PeerRole,
        endpoint: Endpoint,
        certFingerprint: String,
        nonce: String,
        expiresAt: Int64
    ) {
        self.version = version
        self.peerId = peerId
        self.displayName = displayName
        self.role = role
        self.endpoint = endpoint
        self.certFingerprint = certFingerprint
        self.nonce = nonce
        self.expiresAt = expiresAt
    }

    public enum PeerRole: String, Codable, Equatable {
        /// Training and storage node, typically a Mac.
        case brain
        /// Data collection node, typically an iPhone or iPad.
        case sensor
        /// General-purpose mesh node.
        case peer
    }

    public struct Endpoint: Codable, Equatable {
        /// Bonjour service type, such as `_edgemesh._tcp`.
        public let serviceType: String
        /// Bonjour instance name.
        public let serviceName: String
        /// Direct IPv4 fallback when `.local` resolution is unavailable.
        public let ipv4: String?
        public let port: UInt16

        public init(serviceType: String, serviceName: String, ipv4: String?, port: UInt16) {
            self.serviceType = serviceType
            self.serviceName = serviceName
            self.ipv4 = ipv4
            self.port = port
        }
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return e
    }()

    private static let decoder: JSONDecoder = JSONDecoder()

    /// Encode to compact JSON bytes for QR payload.
    public func jsonData() throws -> Data {
        try Self.encoder.encode(self)
    }

    /// Encode to UTF-8 JSON string (what the QR actually carries).
    public func jsonString() throws -> String {
        let data = try jsonData()
        return String(data: data, encoding: .utf8) ?? ""
    }

    public static func decode(jsonData data: Data) throws -> QRPairingPayload {
        let payload = try decoder.decode(QRPairingPayload.self, from: data)
        try payload.validateShape()
        return payload
    }

    public static func decode(jsonString s: String) throws -> QRPairingPayload {
        guard let data = s.data(using: .utf8) else {
            throw MeshError.pairingInvalid("not utf-8")
        }
        return try decode(jsonData: data)
    }

    /// Structural validation (does not check expiration — caller does that with `isExpired(now:)`).
    public func validateShape() throws {
        guard version == 1 else {
            throw MeshError.pairingInvalid("unsupported version \(version)")
        }
        guard !peerId.isEmpty else { throw MeshError.pairingInvalid("empty peerId") }
        guard !displayName.isEmpty else { throw MeshError.pairingInvalid("empty displayName") }
        guard endpoint.port > 0 else { throw MeshError.pairingInvalid("invalid port") }
        guard Self.isValidFingerprint(certFingerprint) else {
            throw MeshError.pairingInvalid("fingerprint must be 64 lowercase hex chars")
        }
        guard Self.isValidNonce(nonce) else {
            throw MeshError.pairingInvalid("nonce must decode to at least 16 bytes")
        }
    }

    public func isExpired(now: Date = Date()) -> Bool {
        Int64(now.timeIntervalSince1970) >= expiresAt
    }

    /// Derive a short human-friendly PIN from the nonce.
    /// Uses the first 4 nonce bytes base32'd (6 characters, crockford-free for clarity).
    ///
    /// The PIN is *not* a cryptographic secret. It's a lookup token: the Mac side keeps
    /// a table (nonce prefix → full payload) and serves the full payload when the iPhone
    /// presents the PIN (with per-IP lockout after 3 failures).
    public var pin: String {
        guard let decoded = Self.base64URLDecode(nonce), decoded.count >= 4 else {
            return "000000"
        }
        let prefix = decoded.prefix(4)
        return Self.base32Encode(Data(prefix)).prefix(6).uppercased()
    }

    public static func generateNonce() -> String {
        var bytes = [UInt8](repeating: 0, count: 16)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        if status != errSecSuccess {
            for i in 0..<bytes.count {
                bytes[i] = UInt8(truncatingIfNeeded: Date().timeIntervalSince1970.bitPattern &+ UInt64(i &* 0x9E37))
            }
        }
        return base64URLEncode(Data(bytes))
    }

    public static func isValidFingerprint(_ s: String) -> Bool {
        s.count == 64 && s.allSatisfy { c in
            ("0"..."9").contains(c) || ("a"..."f").contains(c)
        }
    }

    public static func isValidNonce(_ s: String) -> Bool {
        guard let decoded = base64URLDecode(s) else { return false }
        return decoded.count >= 16
    }

    public static func base64URLEncode(_ data: Data) -> String {
        var s = data.base64EncodedString()
        s = s.replacingOccurrences(of: "+", with: "-")
             .replacingOccurrences(of: "/", with: "_")
             .replacingOccurrences(of: "=", with: "")
        return s
    }

    public static func base64URLDecode(_ s: String) -> Data? {
        var t = s.replacingOccurrences(of: "-", with: "+")
                 .replacingOccurrences(of: "_", with: "/")
        let padding = (4 - t.count % 4) % 4
        t += String(repeating: "=", count: padding)
        return Data(base64Encoded: t)
    }

    private static let base32Alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")

    public static func base32Encode(_ data: Data) -> String {
        var result = ""
        var buffer: UInt64 = 0
        var bits = 0
        for byte in data {
            buffer = (buffer << 8) | UInt64(byte)
            bits += 8
            while bits >= 5 {
                let idx = Int((buffer >> (bits - 5)) & 0x1F)
                result.append(base32Alphabet[idx])
                bits -= 5
            }
        }
        if bits > 0 {
            let idx = Int((buffer << (5 - bits)) & 0x1F)
            result.append(base32Alphabet[idx])
        }
        return result
    }
}

/// Sent by the joining peer (iOS) immediately after mTLS handshake completes.
/// The server cross-checks `nonce` against its pending pairing session table.
public struct PairHelloMessage: Codable, Equatable {
    public let peerId: String
    public let displayName: String
    public let certFingerprint: String
    public let nonce: String

    public init(peerId: String, displayName: String, certFingerprint: String, nonce: String) {
        self.peerId = peerId
        self.displayName = displayName
        self.certFingerprint = certFingerprint
        self.nonce = nonce
    }
}

/// Sent by the brain in response to a valid `pair_hello`.
public struct PairAckMessage: Codable, Equatable {
    public let peerId: String
    public let displayName: String
    public let certFingerprint: String

    public init(peerId: String, displayName: String, certFingerprint: String) {
        self.peerId = peerId
        self.displayName = displayName
        self.certFingerprint = certFingerprint
    }
}
