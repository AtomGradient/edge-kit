// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
#if canImport(EdgeInference)
import EdgeInference
#endif

/// Identity and capability snapshot for an EdgeMesh node.
public struct MeshNode: Identifiable, Sendable, Hashable {

    public let id: String
    public let displayName: String
    public let capability: Capability
    public let deviceProfile: MeshDeviceSnapshot
    public let endpoint: Endpoint
    /// Peer FastAPI HTTP port parsed from the Bonjour TXT `http_port` field.
    public let httpPort: UInt16?
    public let discoveredAt: Date
    public var trustStatus: TrustStatus

    public init(
        id: String = UUID().uuidString,
        displayName: String,
        capability: Capability,
        deviceProfile: MeshDeviceSnapshot,
        endpoint: Endpoint,
        httpPort: UInt16? = nil,
        discoveredAt: Date = Date(),
        trustStatus: TrustStatus = .unknown
    ) {
        self.id = id
        self.displayName = displayName
        self.capability = capability
        self.deviceProfile = deviceProfile
        self.endpoint = endpoint
        self.httpPort = httpPort
        self.discoveredAt = discoveredAt
        self.trustStatus = trustStatus
    }

    /// Pairing / trust state relative to the local device.
    public enum TrustStatus: String, Sendable, Codable, Hashable {
        /// Discovered through Bonjour but not paired.
        case unknown
        /// Present in the trust store and not revoked.
        case trusted
        /// Present in the trust store but revoked by the user.
        case revoked
    }

    /// Capability advertised by the node.
    public enum Capability: String, Sendable, Codable, Hashable {
        /// Runs model inference.
        case inference
        /// Collects local data.
        case data
        /// Runs inference and collects data.
        case both
    }

    /// Network endpoint for mesh transport.
    public struct Endpoint: Sendable, Hashable {
        public let host: String
        public let port: UInt16

        public init(host: String, port: UInt16) {
            self.host = host
            self.port = port
        }
    }

    /// Lightweight device capability snapshot exchanged over EdgeMesh.
    public struct MeshDeviceSnapshot: Sendable, Hashable {
        public let chipName: String
        public let totalRAMGB: Int
        public let availableRAMGB: Int
        public let bandwidthGBs: Double
        public let thermalState: ThermalLevel

        public init(
            chipName: String,
            totalRAMGB: Int,
            availableRAMGB: Int,
            bandwidthGBs: Double,
            thermalState: ThermalLevel
        ) {
            self.chipName = chipName
            self.totalRAMGB = totalRAMGB
            self.availableRAMGB = availableRAMGB
            self.bandwidthGBs = bandwidthGBs
            self.thermalState = thermalState
        }

        public enum ThermalLevel: String, Sendable, Codable, Hashable {
            case nominal, fair, serious, critical
        }

        /// Builds a mesh snapshot from EdgeInference device profile data.
        #if canImport(EdgeInference)
        public static func fromDevice(profile: DeviceProfile, benchmark: DeviceBenchmark?) -> MeshDeviceSnapshot {
            let thermal: ThermalLevel = {
                switch profile.thermalState {
                case .nominal:  return .nominal
                case .fair:     return .fair
                case .serious:  return .serious
                case .critical: return .critical
                }
            }()
            return MeshDeviceSnapshot(
                chipName: profile.metalDeviceName ?? "Metal Family \(profile.metalFamilyTier)",
                totalRAMGB: profile.totalRAMGB,
                availableRAMGB: profile.availableRAMGB,
                bandwidthGBs: benchmark?.effectiveBandwidthGBs ?? 0,
                thermalState: thermal
            )
        }
        #endif
    }
}
