// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Three-tier mesh topology for data collection, daily inference, and panoramic insight.
///
/// - Tier0: Data collection devices.
/// - Tier1: Daily inference devices.
/// - Tier2: Panoramic insight devices.
public struct MeshTopology: Sendable {

    /// Data collection nodes.
    public private(set) var tier0: [MeshNode]
    /// Daily inference nodes.
    public private(set) var tier1: [MeshNode]
    /// Panoramic insight nodes.
    public private(set) var tier2: [MeshNode]

    public init(
        tier0: [MeshNode] = [],
        tier1: [MeshNode] = [],
        tier2: [MeshNode] = []
    ) {
        self.tier0 = tier0
        self.tier1 = tier1
        self.tier2 = tier2
    }

    /// All known nodes.
    public var allNodes: [MeshNode] {
        tier0 + tier1 + tier2
    }

    /// Total number of known nodes.
    public var count: Int {
        tier0.count + tier1.count + tier2.count
    }

    /// Assigns a topology tier from the node capability and device profile.
    public static func assignTier(for node: MeshNode) -> Tier {
        let ram = node.deviceProfile.totalRAMGB
        let bw = node.deviceProfile.bandwidthGBs

        switch node.capability {
        case .data:
            return .tier0
        case .inference, .both:
            if ram >= 32 || bw >= 200 {
                return .tier2
            }
            if ram >= 8 {
                return .tier1
            }
            return .tier0
        }
    }

    /// Adds or replaces a node and assigns it to the appropriate tier.
    public mutating func addNode(_ node: MeshNode) {
        removeNode(id: node.id)
        let tier = Self.assignTier(for: node)
        switch tier {
        case .tier0: tier0.append(node)
        case .tier1: tier1.append(node)
        case .tier2: tier2.append(node)
        }
    }

    /// Removes a node by identifier.
    @discardableResult
    public mutating func removeNode(id: String) -> MeshNode? {
        if let idx = tier0.firstIndex(where: { $0.id == id }) { return tier0.remove(at: idx) }
        if let idx = tier1.firstIndex(where: { $0.id == id }) { return tier1.remove(at: idx) }
        if let idx = tier2.firstIndex(where: { $0.id == id }) { return tier2.remove(at: idx) }
        return nil
    }

    /// Finds a node by identifier.
    public func findNode(id: String) -> MeshNode? {
        allNodes.first { $0.id == id }
    }

    public enum Tier: String, Sendable, CaseIterable {
        case tier0 = "data"
        case tier1 = "inference"
        case tier2 = "panoramic"

        public var displayName: String {
            switch self {
            case .tier0: return "Data Collection"
            case .tier1: return "Daily Inference"
            case .tier2: return "Panoramic Insight"
            }
        }
    }
}
