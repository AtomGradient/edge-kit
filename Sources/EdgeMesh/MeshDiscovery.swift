// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Network

/// Discovers EdgeMesh peers on the local network with Bonjour.
public final class MeshDiscovery: @unchecked Sendable {

    /// Bonjour service type used by EdgeMesh peers.
    public static let serviceType = "_edgemesh._tcp"

    /// Called when a peer is discovered or updated.
    public var onPeerDiscovered: ((MeshNode) -> Void)?
    /// Called when a peer disappears from Bonjour discovery.
    public var onPeerLost: ((String) -> Void)?

    private var browser: NWBrowser?
    private var listener: NWListener?
    private let localNode: MeshNode
    private let queue = DispatchQueue(label: "com.edgemesh.discovery", qos: .utility)
    private var isRunning = false

    public init(localNode: MeshNode) {
        self.localNode = localNode
    }

    /// Starts advertising the local node and browsing for peers.
    public func start() throws {
        guard !isRunning else { throw MeshError.alreadyRunning }
        isRunning = true

        try startListener()
        startBrowser()
    }

    /// Stops advertising and browsing.
    public func stop() {
        isRunning = false
        listener?.cancel()
        listener = nil
        browser?.cancel()
        browser = nil
    }

    private func startListener() throws {
        let txtRecord = buildTXTRecord()
        let params = NWParameters.tcp
        let listener = try NWListener(using: params)

        listener.service = NWListener.Service(
            name: localNode.id,
            type: Self.serviceType,
            txtRecord: txtRecord
        )

        listener.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                if let port = self?.listener?.port {
                    _ = port
                }
            case .failed(let error):
                _ = error
                self?.stop()
            default:
                break
            }
        }

        listener.newConnectionHandler = { connection in
            connection.start(queue: self.queue)
        }

        listener.start(queue: queue)
        self.listener = listener
    }

    private func startBrowser() {
        let descriptor = NWBrowser.Descriptor.bonjourWithTXTRecord(type: Self.serviceType, domain: nil)
        let browser = NWBrowser(for: descriptor, using: .tcp)

        browser.browseResultsChangedHandler = { [weak self] results, changes in
            guard let self = self else { return }

            for change in changes {
                switch change {
                case .added(let result):
                    self.handleDiscovered(result)
                case .removed(let result):
                    self.handleLost(result)
                case .changed(_, let newResult, _):
                    self.handleDiscovered(newResult)
                default:
                    break
                }
            }
        }

        browser.stateUpdateHandler = { state in
            switch state {
            case .ready:
                break
            case .failed:
                break
            default:
                break
            }
        }

        browser.start(queue: queue)
        self.browser = browser
    }

    private func handleDiscovered(_ result: NWBrowser.Result) {
        guard case .service(let name, _, _, _) = result.endpoint else { return }
        guard name != localNode.id else { return }

        if let node = parseNode(from: result) {
            onPeerDiscovered?(node)
        }
    }

    private func handleLost(_ result: NWBrowser.Result) {
        guard case .service(let name, _, _, _) = result.endpoint else { return }
        onPeerLost?(name)
    }

    /// Protocol version of the EdgeMesh wire format. Bump when TXT keys or frame layout change.
    public static let protocolVersion = "1"

    /// Minimum TLS version required by this peer (informational — actual enforcement happens
    /// in `MeshConnection.buildTLSOptions`).
    public static let tlsVersion = "1.3"

    private func buildTXTRecord() -> NWTXTRecord {
        var txt = NWTXTRecord()
        txt["node_id"] = localNode.id
        txt["name"] = localNode.displayName
        txt["cap"] = localNode.capability.rawValue
        txt["chip"] = localNode.deviceProfile.chipName
        txt["ram"] = String(localNode.deviceProfile.totalRAMGB)
        txt["avail_ram"] = String(localNode.deviceProfile.availableRAMGB)
        txt["bw"] = String(format: "%.1f", localNode.deviceProfile.bandwidthGBs)
        txt["thermal"] = localNode.deviceProfile.thermalState.rawValue
        txt["tls_version"] = Self.tlsVersion
        txt["proto_ver"] = Self.protocolVersion
        switch localNode.capability {
        case .inference: txt["role"] = "peer"
        case .data:      txt["role"] = "sensor"
        case .both:      txt["role"] = "brain"
        }
        return txt
    }

    private func parseNode(from result: NWBrowser.Result) -> MeshNode? {
        guard case .service(let name, _, _, _) = result.endpoint else { return nil }

        let metadata = result.metadata
        guard case .bonjour(let txtRecord) = metadata else {
            return MeshNode(
                id: name,
                displayName: name,
                capability: .inference,
                deviceProfile: .init(chipName: "unknown", totalRAMGB: 0, availableRAMGB: 0, bandwidthGBs: 0, thermalState: .nominal),
                endpoint: .init(host: "", port: 0),
                httpPort: nil
            )
        }

        let cap: MeshNode.Capability = {
            if let role = txtRecord["role"] {
                switch role {
                case "brain":  return .both
                case "sensor": return .data
                case "peer":   return .inference
                default: break
                }
            }
            return txtRecord["cap"].flatMap { MeshNode.Capability(rawValue: $0) } ?? .inference
        }()
        let chip = txtRecord["chip"] ?? "unknown"
        let ram = txtRecord["ram"].flatMap { Int($0) } ?? 0
        let availRam = txtRecord["avail_ram"].flatMap { Int($0) } ?? 0
        let bw = txtRecord["bw"].flatMap { Double($0) } ?? 0
        let thermal = txtRecord["thermal"].flatMap { MeshNode.MeshDeviceSnapshot.ThermalLevel(rawValue: $0) } ?? .nominal

        let profile = MeshNode.MeshDeviceSnapshot(
            chipName: chip,
            totalRAMGB: ram,
            availableRAMGB: availRam,
            bandwidthGBs: bw,
            thermalState: thermal
        )

        let ipv4 = txtRecord["ipv4"] ?? ""
        let meshPort = txtRecord["mesh_port"].flatMap { UInt16($0) } ?? 0
        let httpPort = txtRecord["http_port"].flatMap { UInt16($0) }

        return MeshNode(
            id: txtRecord["peer_id"] ?? txtRecord["node_id"] ?? name,
            displayName: txtRecord["display_name"] ?? txtRecord["name"] ?? name,
            capability: cap,
            deviceProfile: profile,
            endpoint: .init(host: ipv4, port: meshPort),
            httpPort: httpPort
        )
    }
}

private extension NWTXTRecord {
    subscript(key: String) -> String? {
        get {
            guard let entry = self.getEntry(for: key) else { return nil }
            if case .string(let value) = entry {
                return value
            }
            return nil
        }
        set {
            if let value = newValue {
                self.setEntry(.string(value), for: key)
            } else {
                self.removeEntry(key: key)
            }
        }
    }
}
