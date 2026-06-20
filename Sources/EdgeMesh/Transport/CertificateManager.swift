// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Security
import CryptoKit

/// Manages the local device's mTLS identity.
public final class CertificateManager {

    /// In-memory handle on the current identity.
    public struct Identity: Sendable {
        /// Raw X.509 DER certificate bytes.
        public let certificateDER: Data
        /// PEM-encoded EC private key.
        public let privateKeyPEM: String
        /// Lowercase SHA-256 certificate fingerprint.
        public let fingerprint: String

        public init(certificateDER: Data, privateKeyPEM: String, fingerprint: String) {
            self.certificateDER = certificateDER
            self.privateKeyPEM = privateKeyPEM
            self.fingerprint = fingerprint
        }
    }

    public static func fingerprint(of certificateDER: Data) -> String {
        let digest = SHA256.hash(data: certificateDER)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Generate a fresh P-256 self-signed identity for the given peerId.
    public static func generate(peerId: String, displayName: String) throws -> Identity {
        _ = displayName
        #if os(iOS)
        return try generateIOSKeychainIdentity(peerId: peerId)
        #else
        let privateKey = P256.Signing.PrivateKey()
        let publicKey = privateKey.publicKey

        let serial = randomSerial()
        let now = Date()
        let notBefore = now.addingTimeInterval(-300)
        let notAfter = now.addingTimeInterval(60 * 60 * 24 * 365 * 10)

        let tbsDER = try buildTBSCertificateDER(
            commonName: "EdgeStudio-\(peerId)",
            organization: "EdgeStudio",
            serial: serial,
            notBefore: notBefore,
            notAfter: notAfter,
            sanDNS: "edgestudio-\(peerId).local",
            publicKeyX963: publicKey.x963Representation
        )
        let cryptoSig = try privateKey.signature(for: tbsDER)
        let rawSig = cryptoSig.rawRepresentation
        let signatureDER = ASN1.sequence([
            ASN1.integer(bytes: Data(rawSig.prefix(32))),
            ASN1.integer(bytes: Data(rawSig.suffix(32))),
        ])
        let certDER = assembleCertDER(tbs: tbsDER, ecdsaSha256SignatureDER: signatureDER)

        let pem = try privateKeyPEM(from: privateKey)
        return Identity(
            certificateDER: certDER,
            privateKeyPEM: pem,
            fingerprint: fingerprint(of: certDER)
        )
        #endif
    }

    /// Load existing identity or generate + persist a new one.
    public static func loadOrCreate(peerId: String, displayName: String) throws -> Identity {
        if let existing = try load() { return existing }
        let fresh = try generate(peerId: peerId, displayName: displayName)
        #if os(iOS)
        #else
        try persist(fresh)
        #endif
        return fresh
    }

    /// Try to read a previously persisted identity. Returns nil if none exists.
    public static func load() throws -> Identity? {
        #if os(iOS)
        return try loadFromKeychain()
        #else
        return try loadFromP12File()
        #endif
    }

    public static func persist(_ identity: Identity) throws {
        #if os(iOS)
        #else
        try persistToP12File(identity)
        #endif
    }

    public static func reset() throws {
        #if os(iOS)
        try deleteFromKeychain()
        #else
        try deleteP12File()
        #endif
    }

    #if os(iOS)

    private static let keychainService = "com.edgestudio.edgemesh.identity"
    private static let keychainCertTag = "com.edgestudio.edgemesh.cert"
    private static let keychainKeyTag = "com.edgestudio.edgemesh.key"

    /// Load a previously created identity from the iOS Keychain.
    private static func loadFromKeychain() throws -> Identity? {
        let idQuery: [String: Any] = [
            kSecClass as String:      kSecClassIdentity,
            kSecAttrLabel as String:  keychainCertTag,
            kSecReturnRef as String:  true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var idRef: CFTypeRef?
        let idStatus = SecItemCopyMatching(idQuery as CFDictionary, &idRef)
        if idStatus == errSecItemNotFound {
            try? deleteFromKeychain()
            return nil
        }
        guard idStatus == errSecSuccess, let id = idRef else {
            throw MeshError.keychainError(idStatus)
        }
        let secIdentity = id as! SecIdentity

        var certOut: SecCertificate?
        let copyStatus = SecIdentityCopyCertificate(secIdentity, &certOut)
        guard copyStatus == errSecSuccess, let secCert = certOut else {
            throw MeshError.keychainError(copyStatus)
        }
        let certDER = SecCertificateCopyData(secCert) as Data

        return Identity(
            certificateDER: certDER,
            privateKeyPEM: "",
            fingerprint: fingerprint(of: certDER)
        )
    }

    private static func deleteFromKeychain() throws {
        let certQuery: [String: Any] = [
            kSecClass as String:     kSecClassCertificate,
            kSecAttrLabel as String: keychainCertTag,
        ]
        SecItemDelete(certQuery as CFDictionary)
        let keyQueryNew: [String: Any] = [
            kSecClass as String:              kSecClassKey,
            kSecAttrApplicationTag as String: Data(keychainKeyTag.utf8),
        ]
        SecItemDelete(keyQueryNew as CFDictionary)
        let keyQueryLegacy: [String: Any] = [
            kSecClass as String:              kSecClassKey,
            kSecAttrApplicationTag as String: keychainKeyTag,
        ]
        SecItemDelete(keyQueryLegacy as CFDictionary)
    }

    /// Build a `SecIdentity` for `NWProtocolTLS.Options.sec_protocol_options_set_local_identity`.
    public static func secIdentity(for identity: Identity) throws -> SecIdentity {
        _ = identity
        let idQuery: [String: Any] = [
            kSecClass as String:            kSecClassIdentity,
            kSecAttrLabel as String:        keychainCertTag,
            kSecReturnRef as String:        true
        ]
        var idRef: CFTypeRef?
        let status = SecItemCopyMatching(idQuery as CFDictionary, &idRef)
        guard status == errSecSuccess, let id = idRef else {
            throw MeshError.keychainError(status)
        }
        return id as! SecIdentity
    }

    #endif

    #if os(macOS)

    public static func p12URL() -> URL {
        let fm = FileManager.default
        let base = (try? fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        return base.appendingPathComponent("EdgeStudio", isDirectory: true)
                   .appendingPathComponent("identity.p12")
    }

    private static func loadFromP12File() throws -> Identity? {
        let url = p12URL()
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }

        let raw = try String(contentsOf: url, encoding: .utf8)
        let lines = raw.split(separator: "\n").map(String.init)
        guard lines.count >= 2,
              let certB64 = lines.first(where: { $0.hasPrefix("cert:") })?.dropFirst("cert:".count),
              let keyPemEnc = lines.first(where: { $0.hasPrefix("key:") })?.dropFirst("key:".count),
              let certDER = Data(base64Encoded: String(certB64)),
              let keyPemData = Data(base64Encoded: String(keyPemEnc)),
              let keyPem = String(data: keyPemData, encoding: .utf8)
        else {
            throw MeshError.identityUnavailable("malformed identity bundle")
        }
        return Identity(certificateDER: certDER, privateKeyPEM: keyPem, fingerprint: fingerprint(of: certDER))
    }

    private static func persistToP12File(_ identity: Identity) throws {
        let url = p12URL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let body = """
        cert:\(identity.certificateDER.base64EncodedString())
        key:\(Data(identity.privateKeyPEM.utf8).base64EncodedString())
        """
        try body.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    private static func deleteP12File() throws {
        let url = p12URL()
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    #endif

    /// Encode P256 private key as unencrypted PKCS#8 PEM.
    public static func privateKeyPEM(from key: P256.Signing.PrivateKey) throws -> String {
        return key.pemRepresentation
    }

    private static func privateScalarBytes(fromPEM pem: String) throws -> Data {
        let key = try P256.Signing.PrivateKey(pemRepresentation: pem)
        return key.rawRepresentation
    }

    /// Build X9.63 representation (0x04 || X || Y || D) from private scalar.
    private static func x963Representation(privateScalar: Data) throws -> Data {
        let key = try P256.Signing.PrivateKey(rawRepresentation: privateScalar)
        return key.x963Representation
    }

    private static func randomSerial() -> Data {
        var bytes = [UInt8](repeating: 0, count: 16)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        bytes[0] &= 0x7F
        return Data(bytes)
    }

    /// Build the unsigned TBSCertificate DER — shared by CryptoKit (macOS) and
    /// SecKey (iOS) signing paths. Takes the public key as raw x9.63 bytes
    /// (`0x04 || X || Y`) since that's the intersection of both SDKs' outputs.
    private static func buildTBSCertificateDER(
        commonName: String,
        organization: String,
        serial: Data,
        notBefore: Date,
        notAfter: Date,
        sanDNS: String,
        publicKeyX963: Data
    ) throws -> Data {
        let issuerSubject = ASN1.sequence([
            ASN1.set([
                ASN1.sequence([
                    ASN1.oid("2.5.4.10"),
                    ASN1.utf8String(organization)
                ])
            ]),
            ASN1.set([
                ASN1.sequence([
                    ASN1.oid("2.5.4.3"),
                    ASN1.utf8String(commonName)
                ])
            ])
        ])

        let validity = ASN1.sequence([
            ASN1.utcTime(notBefore),
            ASN1.utcTime(notAfter)
        ])

        let algorithmIdEC = ASN1.sequence([
            ASN1.oid("1.2.840.10045.2.1"),
            ASN1.oid("1.2.840.10045.3.1.7")
        ])
        let spki = ASN1.sequence([
            algorithmIdEC,
            ASN1.bitString(publicKeyX963)
        ])

        let ski = Data(Insecure.SHA1.hash(data: publicKeyX963))
        let skiExt = ASN1.sequence([
            ASN1.oid("2.5.29.14"),
            ASN1.octetString(ASN1.octetString(ski))
        ])
        let bcExt = ASN1.sequence([
            ASN1.oid("2.5.29.19"),
            ASN1.bool(true),
            ASN1.octetString(ASN1.sequence([ASN1.bool(false)]))
        ])
        let kuExt = ASN1.sequence([
            ASN1.oid("2.5.29.15"),
            ASN1.bool(true),
            ASN1.octetString(ASN1.bitStringFromByte(0xA0, unusedBits: 3))
        ])
        let ekuExt = ASN1.sequence([
            ASN1.oid("2.5.29.37"),
            ASN1.octetString(ASN1.sequence([
                ASN1.oid("1.3.6.1.5.5.7.3.1"),
                ASN1.oid("1.3.6.1.5.5.7.3.2")
            ]))
        ])
        let sanExt = ASN1.sequence([
            ASN1.oid("2.5.29.17"),
            ASN1.octetString(ASN1.sequence([
                ASN1.contextImplicit(tag: 2, content: Data(sanDNS.utf8))
            ]))
        ])

        let extensions = ASN1.contextExplicit(tag: 3, content: ASN1.sequence([
            skiExt, bcExt, kuExt, ekuExt, sanExt
        ]))

        let sigAlg = ASN1.sequence([
            ASN1.oid("1.2.840.10045.4.3.2")
        ])

        let tbs = ASN1.sequence([
            ASN1.contextExplicit(tag: 0, content: ASN1.integer(2)),
            ASN1.integer(bytes: serial),
            sigAlg,
            issuerSubject,
            validity,
            issuerSubject,
            spki,
            extensions
        ])
        return tbs
    }

    /// Wrap a TBS + ECDSA-SHA256 signature into a full X.509 Certificate SEQUENCE.
    private static func assembleCertDER(tbs: Data, ecdsaSha256SignatureDER: Data) -> Data {
        let sigAlg = ASN1.sequence([
            ASN1.oid("1.2.840.10045.4.3.2")
        ])
        return ASN1.sequence([
            tbs,
            sigAlg,
            ASN1.bitString(ecdsaSha256SignatureDER),
        ])
    }

    #if os(iOS)

    /// Create a fresh identity directly in the iOS Keychain.
    private static func generateIOSKeychainIdentity(peerId: String) throws -> Identity {
        try? deleteFromKeychain()

        let tagData = Data(keychainKeyTag.utf8)
        let privateKeyAttrs: [String: Any] = [
            kSecAttrIsPermanent as String:     true,
            kSecAttrApplicationTag as String:  tagData,
            kSecAttrAccessible as String:      kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]
        let keyParams: [String: Any] = [
            kSecAttrKeyType as String:       kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeySizeInBits as String: 256,
            kSecPrivateKeyAttrs as String:   privateKeyAttrs,
        ]

        var cfError: Unmanaged<CFError>?
        guard let privKey = SecKeyCreateRandomKey(keyParams as CFDictionary, &cfError) else {
            let msg = (cfError?.takeRetainedValue()).map { String(describing: $0) } ?? "unknown"
            throw MeshError.identityUnavailable("SecKeyCreateRandomKey failed: \(msg)")
        }

        guard let pubSecKey = SecKeyCopyPublicKey(privKey) else {
            throw MeshError.identityUnavailable("SecKeyCopyPublicKey failed")
        }
        guard let pubBytes = SecKeyCopyExternalRepresentation(pubSecKey, &cfError) as Data? else {
            let msg = (cfError?.takeRetainedValue()).map { String(describing: $0) } ?? "unknown"
            throw MeshError.identityUnavailable("SecKeyCopyExternalRepresentation(pub) failed: \(msg)")
        }
        guard pubBytes.count == 65, pubBytes[0] == 0x04 else {
            throw MeshError.identityUnavailable("unexpected pubkey byte layout: \(pubBytes.count)")
        }

        let serial = randomSerial()
        let now = Date()
        let notBefore = now.addingTimeInterval(-300)
        let notAfter = now.addingTimeInterval(60 * 60 * 24 * 365 * 10)
        let tbsDER = try buildTBSCertificateDER(
            commonName: "EdgeStudio-\(peerId)",
            organization: "EdgeStudio",
            serial: serial,
            notBefore: notBefore,
            notAfter: notAfter,
            sanDNS: "edgestudio-\(peerId).local",
            publicKeyX963: pubBytes
        )

        guard SecKeyIsAlgorithmSupported(privKey, .sign, .ecdsaSignatureMessageX962SHA256) else {
            throw MeshError.identityUnavailable("SecKey does not support ecdsaSignatureMessageX962SHA256")
        }
        guard let signatureDER = SecKeyCreateSignature(
            privKey,
            .ecdsaSignatureMessageX962SHA256,
            tbsDER as CFData,
            &cfError
        ) as Data? else {
            let msg = (cfError?.takeRetainedValue()).map { String(describing: $0) } ?? "unknown"
            throw MeshError.identityUnavailable("SecKeyCreateSignature failed: \(msg)")
        }
        let certDER = assembleCertDER(tbs: tbsDER, ecdsaSha256SignatureDER: signatureDER)

        guard let secCert = SecCertificateCreateWithData(nil, certDER as CFData) else {
            throw MeshError.identityUnavailable("SecCertificateCreateWithData failed")
        }
        let certAttrs: [String: Any] = [
            kSecClass as String:     kSecClassCertificate,
            kSecAttrLabel as String: keychainCertTag,
            kSecValueRef as String:  secCert,
        ]
        SecItemDelete(certAttrs as CFDictionary)
        let status = SecItemAdd(certAttrs as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw MeshError.keychainError(status)
        }

        return Identity(
            certificateDER: certDER,
            privateKeyPEM: "",
            fingerprint: fingerprint(of: certDER)
        )
    }

    #endif
}

enum ASN1 {

    static func lenBytes(_ length: Int) -> Data {
        if length < 0x80 {
            return Data([UInt8(length)])
        }
        var bytes: [UInt8] = []
        var n = length
        while n > 0 {
            bytes.insert(UInt8(n & 0xFF), at: 0)
            n >>= 8
        }
        return Data([0x80 | UInt8(bytes.count)]) + Data(bytes)
    }

    static func tlv(tag: UInt8, content: Data) -> Data {
        Data([tag]) + lenBytes(content.count) + content
    }

    static func sequence(_ children: [Data]) -> Data {
        tlv(tag: 0x30, content: children.reduce(Data(), +))
    }

    static func set(_ children: [Data]) -> Data {
        tlv(tag: 0x31, content: children.reduce(Data(), +))
    }

    static func integer(_ value: Int) -> Data {
        if value == 0 { return tlv(tag: 0x02, content: Data([0x00])) }
        var n = value
        var bytes: [UInt8] = []
        while n > 0 {
            bytes.insert(UInt8(n & 0xFF), at: 0)
            n >>= 8
        }
        if bytes.first! & 0x80 != 0 { bytes.insert(0x00, at: 0) }
        return tlv(tag: 0x02, content: Data(bytes))
    }

    static func integer(bytes: Data) -> Data {
        var b = [UInt8](bytes)
        while b.count > 1 && b.first == 0x00 && (b[1] & 0x80) == 0 {
            b.removeFirst()
        }
        if let first = b.first, first & 0x80 != 0 {
            b.insert(0x00, at: 0)
        }
        return tlv(tag: 0x02, content: Data(b))
    }

    static func bool(_ value: Bool) -> Data {
        tlv(tag: 0x01, content: Data([value ? 0xFF : 0x00]))
    }

    static func octetString(_ content: Data) -> Data {
        tlv(tag: 0x04, content: content)
    }

    static func bitString(_ content: Data) -> Data {
        tlv(tag: 0x03, content: Data([0x00]) + content)
    }

    static func bitStringFromByte(_ byte: UInt8, unusedBits: UInt8) -> Data {
        tlv(tag: 0x03, content: Data([unusedBits, byte]))
    }

    static func utf8String(_ s: String) -> Data {
        tlv(tag: 0x0C, content: Data(s.utf8))
    }

    static func utcTime(_ date: Date) -> Data {
        let df = DateFormatter()
        df.locale = Locale(identifier: "en_US_POSIX")
        df.dateFormat = "yyMMddHHmmss'Z'"
        df.timeZone = TimeZone(identifier: "UTC")
        let s = df.string(from: date)
        return tlv(tag: 0x17, content: Data(s.utf8))
    }

    static func oid(_ dotted: String) -> Data {
        let parts = dotted.split(separator: ".").compactMap { UInt64($0) }
        precondition(parts.count >= 2, "invalid OID: \(dotted)")
        var bytes: [UInt8] = [UInt8(parts[0] * 40 + parts[1])]
        for i in 2..<parts.count {
            var v = parts[i]
            var stack: [UInt8] = [UInt8(v & 0x7F)]
            v >>= 7
            while v > 0 {
                stack.insert(UInt8((v & 0x7F) | 0x80), at: 0)
                v >>= 7
            }
            bytes.append(contentsOf: stack)
        }
        return tlv(tag: 0x06, content: Data(bytes))
    }

    static func contextExplicit(tag: UInt8, content: Data) -> Data {
        tlv(tag: 0xA0 | tag, content: content)
    }

    static func contextImplicit(tag: UInt8, content: Data) -> Data {
        tlv(tag: 0x80 | tag, content: content)
    }
}
