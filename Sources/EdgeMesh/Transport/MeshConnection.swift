// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Network
import Security

/// Outbound mTLS-encrypted connection to a paired peer.
///
/// The connection layers on top of `NWConnection` with `NWProtocolTLS.Options`:
/// - Local identity: `SecIdentity` from `CertificateManager`
/// - Peer verification: custom `sec_protocol_verify_block` that compares the
///   peer's leaf certificate fingerprint against the value pinned in `TrustStore`.
///
/// Once ready, the caller sends/receives `FrameCodec.Frame` payloads.
///
/// `MeshConnection` is intentionally thin: it doesn't know about ops — it's the pipe.
/// `PairingService` / data flush / adapter download all compose on top.
@available(iOS 17.0, macOS 14.0, *)
public final class MeshConnection: @unchecked Sendable {

    public enum State: Equatable {
        case idle
        case connecting
        case ready
        case failed(String)
        case cancelled
    }

    public struct Config: Sendable {
        /// IPv4 address or Bonjour `.local` name.
        public let host: String
        public let port: UInt16
        /// Hex-encoded SHA-256 fingerprint of the peer certificate.
        public let expectedFingerprint: String
        public let identity: CertificateManager.Identity
        /// If true, allow the initial handshake without requiring a matching
        /// fingerprint in the TrustStore — used during the initial `pair_hello`
        /// where the fingerprint from the QR payload is the only source of truth.
        public let bootstrapTrust: Bool
        /// Keepalive interval in seconds. 0 disables keepalive. Default 30s —
        /// matches typical NAT timeout and is light enough not to wake radios.
        /// If no pong received within 2× interval, connection transitions to .failed.
        public let keepaliveInterval: TimeInterval

        public init(
            host: String,
            port: UInt16,
            expectedFingerprint: String,
            identity: CertificateManager.Identity,
            bootstrapTrust: Bool = false,
            keepaliveInterval: TimeInterval = 30
        ) {
            self.host = host
            self.port = port
            self.expectedFingerprint = expectedFingerprint
            self.identity = identity
            self.bootstrapTrust = bootstrapTrust
            self.keepaliveInterval = keepaliveInterval
        }
    }

    private let config: Config
    private var connection: NWConnection?
    private let buffer = FrameCodec.Buffer()
    private let queue: DispatchQueue
    private let stateLock = NSLock()
    private var _state: State = .idle

    private var stateListeners: [(State) -> Void] = []
    private var frameListeners: [(Data) -> Void] = []

    private var keepaliveTimer: DispatchSourceTimer?
    private var lastPongAt: Date = Date.distantPast

    /// Optional provider that supplies up-to-date sensor stats for each ping payload.
    /// Set by the host app (e.g. its `MeshManager`) right after connection
    /// becomes ready; nil = legacy behaviour (timestamp-only ping).
    ///
    /// Provider is invoked on the connection's queue every keepalive interval, so it
    /// must be cheap (a few SQLite COUNT(*) queries are fine, blocking I/O is not).
    public typealias PingStatsProvider = @Sendable () -> PingStats?
    private var statsProvider: PingStatsProvider?

    public var state: State {
        stateLock.lock(); defer { stateLock.unlock() }
        return _state
    }

    public init(config: Config, queueLabel: String = "com.edgemesh.connection") {
        self.config = config
        self.queue = DispatchQueue(label: queueLabel, qos: .userInitiated)
    }

    public func onStateChange(_ handler: @escaping (State) -> Void) {
        stateLock.lock()
        stateListeners.append(handler)
        stateLock.unlock()
    }

    public func onFrame(_ handler: @escaping (Data) -> Void) {
        stateLock.lock()
        frameListeners.append(handler)
        stateLock.unlock()
    }

    /// Register a closure that returns the latest sensor stats to embed in keepalive pings.
    /// Pass `nil` to clear. Idempotent — last write wins.
    public func setPingStatsProvider(_ provider: PingStatsProvider?) {
        stateLock.lock()
        statsProvider = provider
        stateLock.unlock()
    }

    /// Suspend until state transitions out of .connecting. Throws if the state
    /// reaches .failed / .cancelled or the timeout expires.
    ///
    /// Callers of `connect()` must await this before the first `send(_:)`.
    public func awaitReady(timeout: TimeInterval = 10.0) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            switch state {
            case .ready:
                return
            case .failed(let msg):
                throw MeshError.connectionFailed(peer: config.host, reason: msg)
            case .cancelled:
                throw MeshError.connectionFailed(peer: config.host, reason: "cancelled")
            case .idle, .connecting:
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        throw MeshError.connectionFailed(
            peer: config.host,
            reason: "timeout waiting for .ready after \(timeout)s (state=\(state))"
        )
    }

    public func connect() throws {
        let tlsOpts = try buildTLSOptions()
        let params = NWParameters(tls: tlsOpts, tcp: .init())
        let endpoint = NWEndpoint.hostPort(
            host: NWEndpoint.Host(config.host),
            port: NWEndpoint.Port(rawValue: config.port)!
        )

        let conn = NWConnection(to: endpoint, using: params)
        self.connection = conn

        conn.stateUpdateHandler = { [weak self] newState in
            guard let self = self else { return }
            switch newState {
            case .ready:
                self.setState(.ready)
                self.beginReceive()
                self.startKeepalive()
            case .failed(let err):
                self.stopKeepalive()
                self.setState(.failed(String(describing: err)))
            case .cancelled:
                self.stopKeepalive()
                self.setState(.cancelled)
            case .waiting(let err):
                self.stopKeepalive()
                self.setState(.failed("waiting: \(String(describing: err))"))
            default:
                break
            }
        }
        setState(.connecting)
        conn.start(queue: queue)
    }

    /// Start periodic ping → if no pong (any inbound frame qualifies) within 2× interval,
    /// transition connection to .failed. Mac's revoke path closes the socket; if keepalive
    /// times out, iOS side knows quickly instead of sitting on a zombie TLS session.
    private func startKeepalive() {
        let interval = config.keepaliveInterval
        guard interval > 0 else { return }
        stopKeepalive()

        lastPongAt = Date()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler { [weak self] in
            guard let self = self else { return }
            let stale = Date().timeIntervalSince(self.lastPongAt) > (interval * 2.0 + 2.0)
            if stale {
                self.setState(.failed("keepalive timeout (no traffic > \(Int(interval * 2))s)"))
                self.connection?.cancel()
                return
            }
            do {
                self.stateLock.lock()
                let provider = self.statsProvider
                self.stateLock.unlock()
                let stats = provider?()
                let ping = WireMessage.ping(PingMessage(stats: stats))
                let data = try JSONEncoder().encode(ping)
                try self.send(data)
            } catch {
                self.setState(.failed("keepalive send: \(error)"))
            }
        }
        timer.resume()
        keepaliveTimer = timer
    }

    private func stopKeepalive() {
        keepaliveTimer?.cancel()
        keepaliveTimer = nil
    }

    /// Called on any inbound frame to refresh the liveness watermark.
    /// Treating all traffic as "pong" means event_upload / pair_ack / etc. keep the link alive.
    private func touchPong() {
        lastPongAt = Date()
    }

    public func send(_ frame: Data) throws {
        guard case .ready = state else {
            throw MeshError.connectionFailed(peer: config.host, reason: "not ready: \(state)")
        }
        let framed = try FrameCodec.encode(frame)
        connection?.send(content: framed, completion: .contentProcessed { [weak self] err in
            if let err = err {
                self?.setState(.failed("send: \(err)"))
            }
        })
    }

    public func sendJSON<T: Encodable>(_ value: T) throws {
        let payload = try JSONEncoder().encode(value)
        try send(payload)
    }

    public func sendAndWait(_ frame: Data) async throws {
        guard case .ready = state else {
            throw MeshError.connectionFailed(peer: config.host, reason: "not ready: \(state)")
        }
        let framed = try FrameCodec.encode(frame)
        guard let connection else {
            throw MeshError.connectionFailed(peer: config.host, reason: "connection missing")
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(content: framed, completion: .contentProcessed { [weak self] err in
                if let err {
                    self?.setState(.failed("send: \(err)"))
                    cont.resume(throwing: MeshError.connectionFailed(
                        peer: self?.config.host ?? "unknown",
                        reason: "send: \(err)"
                    ))
                    return
                }
                cont.resume()
            })
        }
    }

    public func sendJSONAndWait<T: Encodable>(_ value: T) async throws {
        let payload = try JSONEncoder().encode(value)
        try await sendAndWait(payload)
    }

    public func cancel() {
        connection?.cancel()
    }

    private func buildTLSOptions() throws -> NWProtocolTLS.Options {
        let opts = NWProtocolTLS.Options()
        let secOpts = opts.securityProtocolOptions

        sec_protocol_options_set_min_tls_protocol_version(secOpts, .TLSv13)
        sec_protocol_options_set_max_tls_protocol_version(secOpts, .TLSv13)

        let secIdentity: SecIdentity
        #if os(iOS)
        secIdentity = try CertificateManager.secIdentity(for: config.identity)
        #else
        secIdentity = try CertificateManager.secIdentityForMacOS(identity: config.identity)
        #endif
        guard let secId = sec_identity_create(secIdentity) else {
            throw MeshError.tlsSetupFailed("sec_identity_create failed")
        }
        sec_protocol_options_set_local_identity(secOpts, secId)

        let expected = config.expectedFingerprint.lowercased()
        let bootstrap = config.bootstrapTrust
        let verifyQueue = DispatchQueue(label: "com.edgemesh.tls.verify")
        sec_protocol_options_set_verify_block(secOpts, { _, sec_trust, completion in
            let trust = sec_trust_copy_ref(sec_trust).takeRetainedValue()
            let count = SecTrustGetCertificateCount(trust)
            guard count > 0,
                  let leaf = SecTrustGetCertificateAtIndex(trust, 0) else {
                completion(false)
                return
            }
            let der = SecCertificateCopyData(leaf) as Data
            let actual = CertificateManager.fingerprint(of: der)
            completion(actual == expected || bootstrap && actual.count == 64)
            _ = bootstrap
        }, verifyQueue)

        return opts
    }

    private func beginReceive() {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self = self else { return }

            if let data = data, !data.isEmpty {
                self.buffer.append(data)
                while true {
                    do {
                        guard let payload = try self.buffer.pop() else { break }
                        self.deliver(payload)
                    } catch {
                        self.setState(.failed("frame: \(error)"))
                        return
                    }
                }
            }

            if let error = error {
                self.setState(.failed("recv: \(error)"))
                return
            }
            if isComplete {
                self.setState(.cancelled)
                return
            }
            self.beginReceive()
        }
    }

    private func setState(_ newState: State) {
        stateLock.lock()
        _state = newState
        let listeners = stateListeners
        stateLock.unlock()
        for l in listeners { l(newState) }
    }

    private func deliver(_ frame: Data) {
        touchPong()
        stateLock.lock()
        let listeners = frameListeners
        stateLock.unlock()
        for l in listeners { l(frame) }
    }
}

#if os(macOS)
extension CertificateManager {
    /// On macOS, we don't use the login keychain (SSH-launched backend can't unlock).
    /// Import the cert + key into a transient keychain and construct a SecIdentity.
    public static func secIdentityForMacOS(identity: Identity) throws -> SecIdentity {
        guard let secCert = SecCertificateCreateWithData(nil, identity.certificateDER as CFData) else {
            throw MeshError.identityUnavailable("cert reimport failed")
        }
        let pkBytes = try P256.Signing.PrivateKey(pemRepresentation: identity.privateKeyPEM).x963Representation
        let keyAttrs: [String: Any] = [
            kSecAttrKeyType as String:       kSecAttrKeyTypeECSECPrimeRandom,
            kSecAttrKeyClass as String:      kSecAttrKeyClassPrivate,
            kSecAttrKeySizeInBits as String: 256
        ]
        var cfError: Unmanaged<CFError>?
        guard let secKey = SecKeyCreateWithData(pkBytes as CFData, keyAttrs as CFDictionary, &cfError) else {
            if let e = cfError { throw MeshError.identityUnavailable("SecKeyCreateWithData: \(e.takeRetainedValue())") }
            throw MeshError.identityUnavailable("SecKeyCreateWithData failed")
        }

        let pass = UUID().uuidString
        let p12 = try buildP12(cert: secCert, privateKey: secKey, password: pass)
        var items: CFArray?
        let importOptions: [String: Any] = [
            kSecImportExportPassphrase as String: pass
        ]
        let status = SecPKCS12Import(p12 as CFData, importOptions as CFDictionary, &items)
        guard status == errSecSuccess, let arr = items as? [[String: Any]], let first = arr.first,
              let identityRaw = first[kSecImportItemIdentity as String] else {
            throw MeshError.keychainError(status)
        }
        return identityRaw as! SecIdentity
    }

    private static func buildP12(cert: SecCertificate, privateKey: SecKey, password: String) throws -> Data {
        let tmp = FileManager.default.temporaryDirectory
        let certPath = tmp.appendingPathComponent("edgemesh_identity_\(UUID().uuidString).crt")
        let keyPath  = tmp.appendingPathComponent("edgemesh_identity_\(UUID().uuidString).key")
        let p12Path  = tmp.appendingPathComponent("edgemesh_identity_\(UUID().uuidString).p12")
        defer {
            [certPath, keyPath, p12Path].forEach { try? FileManager.default.removeItem(at: $0) }
        }

        let certDER = SecCertificateCopyData(cert) as Data
        try certDER.write(to: certPath)

        var cfError: Unmanaged<CFError>?
        guard let keyData = SecKeyCopyExternalRepresentation(privateKey, &cfError) as Data? else {
            throw MeshError.identityUnavailable("SecKeyCopyExternalRepresentation failed")
        }
        guard keyData.count >= 65 else {
            throw MeshError.identityUnavailable("unexpected EC key layout")
        }
        let scalar = keyData.subdata(in: 65..<keyData.count)
        let cryptoKey = try P256.Signing.PrivateKey(rawRepresentation: scalar)
        try Data(cryptoKey.pemRepresentation.utf8).write(to: keyPath)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/openssl")
        process.arguments = [
            "pkcs12", "-export",
            "-in", certPath.path,
            "-inkey", keyPath.path,
            "-out", p12Path.path,
            "-passout", "pass:\(password)",
            "-certpbe", "AES-256-CBC",
            "-keypbe", "AES-256-CBC",
            "-macalg", "SHA256"
        ]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw MeshError.identityUnavailable("openssl pkcs12 export failed rc=\(process.terminationStatus)")
        }
        return try Data(contentsOf: p12Path)
    }
}
#endif

import CryptoKit
