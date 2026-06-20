// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Combine

/// Entry point for EdgeMesh discovery, routing, pairing, and trust state.
@MainActor
public final class MeshEngine: ObservableObject {

    @Published public private(set) var peers: [MeshNode] = []
    @Published public private(set) var topology: MeshTopology = MeshTopology()
    @Published public private(set) var isDiscovering: Bool = false

    private var discovery: MeshDiscovery?
    private var localNode: MeshNode?

    private var _identity: CertificateManager.Identity?
    private var _trustStore: TrustStore?

    public init() {}

    /// Starts local peer discovery.
    public func startDiscovery(as node: MeshNode? = nil) throws {
        guard !isDiscovering else { throw MeshError.alreadyRunning }

        let local = node ?? makeLocalNode()
        self.localNode = local

        var topo = topology
        topo.addNode(local)
        topology = topo

        let disc = MeshDiscovery(localNode: local)

        disc.onPeerDiscovered = { [weak self] peer in
            Task { @MainActor in
                self?.addPeer(peer)
            }
        }

        disc.onPeerLost = { [weak self] peerId in
            Task { @MainActor in
                self?.removePeer(id: peerId)
            }
        }

        try disc.start()
        discovery = disc
        isDiscovering = true
    }

    /// Stops local peer discovery.
    public func stopDiscovery() {
        discovery?.stop()
        discovery = nil
        isDiscovering = false
    }

    /// Connects to a trusted peer over mTLS.
    @discardableResult
    public func connect(to peer: MeshNode) async throws -> MeshConnection {
        guard peers.contains(where: { $0.id == peer.id }) else {
            throw MeshError.peerUnavailable(peer.displayName)
        }
        guard let identity = _identity, let trustStore = _trustStore else {
            throw MeshError.identityUnavailable("call setupSecurity(peerId:displayName:) first")
        }
        guard let trusted = try trustStore.lookup(peerId: peer.id) else {
            throw MeshError.untrustedPeer(peer.id)
        }
        guard !trusted.revoked else {
            throw MeshError.peerRevoked(peer.id)
        }

        let config = MeshConnection.Config(
            host: peer.endpoint.host,
            port: peer.endpoint.port,
            expectedFingerprint: trusted.fingerprint,
            identity: identity,
            bootstrapTrust: false
        )
        let conn = MeshConnection(config: config)
        try conn.connect()
        try await conn.awaitReady()
        try trustStore.touchLastSeen(peerId: peer.id)
        return conn
    }

    /// Initializes local mTLS identity and trust storage.
    public func setupSecurity(peerId: String, displayName: String, trustStoreURL: URL? = nil) throws {
        if _identity == nil {
            _identity = try CertificateManager.loadOrCreate(peerId: peerId, displayName: displayName)
        }
        if _trustStore == nil {
            let url = trustStoreURL ?? TrustStore.defaultURL()
            _trustStore = try TrustStore(url: url)
        }
    }

    /// Installs preloaded security state.
    public func installSecurity(identity: CertificateManager.Identity, trustStore: TrustStore) {
        if _identity == nil { _identity = identity }
        if _trustStore == nil { _trustStore = trustStore }
    }

    /// Local certificate fingerprint.
    public var localFingerprint: String? { _identity?.fingerprint }

    /// Completes pairing from a QR or PIN-exchanged payload.
    @discardableResult
    public func completePairing(
        with payload: QRPairingPayload,
        localPeerId: String,
        localDisplayName: String
    ) async throws -> PairingService.Result {
        guard let identity = _identity, let trustStore = _trustStore else {
            throw MeshError.identityUnavailable("call setupSecurity(peerId:displayName:) first")
        }
        let service = PairingService(
            localPeerId: localPeerId,
            localDisplayName: localDisplayName,
            identity: identity,
            trustStore: trustStore
        )
        let result = try await service.completePairing(with: payload)
        addPeer(result.node)
        return result
    }

    /// Returns a snapshot of trusted peers.
    public func listTrustedPeers() throws -> [TrustStore.TrustedPeer] {
        guard let trustStore = _trustStore else { return [] }
        return try trustStore.listAll()
    }

    /// Revokes a peer.
    public func revoke(peerId: String) throws {
        guard let trustStore = _trustStore else {
            throw MeshError.trustStoreError("not initialized")
        }
        try trustStore.revoke(peerId: peerId)
        if let idx = peers.firstIndex(where: { $0.id == peerId }) {
            var updated = peers[idx]
            updated.trustStatus = .revoked
            peers[idx] = updated
        }
    }

    /// Permanently deletes a peer from trust storage and memory.
    public func deletePeer(peerId: String) throws {
        guard let trustStore = _trustStore else {
            throw MeshError.trustStoreError("not initialized")
        }
        try trustStore.delete(peerId: peerId)
        peers.removeAll { $0.id == peerId }
        var topo = topology
        topo.removeNode(id: peerId)
        topology = topo
    }

    /// Returns the best node for a model size.
    public func bestNode(for modelSizeGB: Double, strategy: MeshRouter.Strategy = .bestFit) -> MeshNode? {
        MeshRouter.bestNode(for: modelSizeGB, in: topology, strategy: strategy)
    }

    /// Returns a full routing plan for a model size.
    public func routingPlan(for modelSizeGB: Double) -> RoutingPlan {
        MeshRouter.routingPlan(for: modelSizeGB, in: topology)
    }

    private func addPeer(_ peer: MeshNode) {
        if let idx = peers.firstIndex(where: { $0.id == peer.id }) {
            let preservedTrust = peers[idx].trustStatus
            let preservedDiscovered = peers[idx].discoveredAt
            peers[idx] = MeshNode(
                id: peer.id,
                displayName: peer.displayName,
                capability: peer.capability,
                deviceProfile: peer.deviceProfile,
                endpoint: peer.endpoint,
                httpPort: peer.httpPort,
                discoveredAt: preservedDiscovered,
                trustStatus: preservedTrust
            )
        } else {
            peers.append(peer)
        }
        var topo = topology
        topo.addNode(peer)
        topology = topo
    }

    private func removePeer(id: String) {
        peers.removeAll { $0.id == id }
        var topo = topology
        topo.removeNode(id: id)
        topology = topo
    }

    private nonisolated func makeLocalNode() -> MeshNode {
        let hostName = ProcessInfo.processInfo.hostName
        let ram = Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)

        let profile = MeshNode.MeshDeviceSnapshot(
            chipName: "unknown",
            totalRAMGB: ram,
            availableRAMGB: ram / 2,
            bandwidthGBs: 0,
            thermalState: .nominal
        )

        return MeshNode(
            displayName: hostName,
            capability: ram >= 8 ? .both : .data,
            deviceProfile: profile,
            endpoint: .init(host: "0.0.0.0", port: 0)
        )
    }
}
