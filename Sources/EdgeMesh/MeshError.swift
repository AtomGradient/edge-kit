// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// EdgeMesh error type.
public enum MeshError: Error, LocalizedError {
    case discoveryFailed(String)
    case connectionFailed(peer: String, reason: String)
    case routingFailed(String)
    case peerUnavailable(String)
    case summaryFailed(String)
    case alreadyRunning
    case notRunning

    case tlsSetupFailed(String)
    case fingerprintMismatch(expected: String, actual: String)
    case peerRevoked(String)
    case untrustedPeer(String)
    case identityUnavailable(String)
    case keychainError(OSStatus)
    case frameDecodeError(String)
    case frameTooLarge(Int)

    case pairingExpired
    case pairingReplay(String)
    case pairingInvalid(String)
    case pinInvalid
    case pinLocked

    case trustStoreError(String)

    public var errorDescription: String? {
        switch self {
        case .discoveryFailed(let r):            return "Discovery failed: \(r)"
        case .connectionFailed(let p, let r):    return "Connection to \(p) failed: \(r)"
        case .routingFailed(let r):              return "Routing failed: \(r)"
        case .peerUnavailable(let p):            return "Peer unavailable: \(p)"
        case .summaryFailed(let r):              return "Summary failed: \(r)"
        case .alreadyRunning:                    return "Mesh discovery is already running"
        case .notRunning:                        return "Mesh discovery is not running"
        case .tlsSetupFailed(let r):             return "TLS setup failed: \(r)"
        case .fingerprintMismatch(let e, let a): return "Peer fingerprint mismatch (expected \(e.prefix(16))… got \(a.prefix(16))…)"
        case .peerRevoked(let p):                return "Peer \(p) has been revoked"
        case .untrustedPeer(let p):              return "Peer \(p) not in trust store"
        case .identityUnavailable(let r):        return "Local identity unavailable: \(r)"
        case .keychainError(let s):              return "Keychain error: OSStatus=\(s)"
        case .frameDecodeError(let r):           return "Frame decode error: \(r)"
        case .frameTooLarge(let n):              return "Frame too large: \(n) bytes"
        case .pairingExpired:                    return "Pairing payload has expired"
        case .pairingReplay(let nonce):          return "Pairing nonce replay detected: \(nonce.prefix(8))…"
        case .pairingInvalid(let r):             return "Pairing invalid: \(r)"
        case .pinInvalid:                        return "PIN invalid or expired"
        case .pinLocked:                         return "PIN locked out due to repeated failures"
        case .trustStoreError(let r):            return "Trust store error: \(r)"
        }
    }
}
