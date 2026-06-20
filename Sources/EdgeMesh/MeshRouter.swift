// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Selects the best inference node for a model and the current mesh topology.
public struct MeshRouter: Sendable {

    /// Routing strategy used when multiple nodes can serve the request.
    public enum Strategy: Sendable {
        /// Prefer the node with the best model-size, bandwidth, and thermal fit.
        case bestFit
        /// Prefer the node with the most available memory.
        case leastLoaded
        /// Prefer the node with the highest memory bandwidth.
        case fastest
    }

    /// Returns the best inference node for a model size.
    ///
    /// - Parameters:
    ///   - modelSizeGB: Model size in GB.
    ///   - topology: Current mesh topology.
    ///   - strategy: Routing strategy.
    /// - Returns: The selected node, or `nil` when no inference-capable node is available.
    public static func bestNode(
        for modelSizeGB: Double,
        in topology: MeshTopology,
        strategy: Strategy = .bestFit
    ) -> MeshNode? {
        let candidates = (topology.tier1 + topology.tier2)
            .filter { $0.capability != .data }
            .filter { $0.deviceProfile.thermalState != .critical }

        guard !candidates.isEmpty else { return nil }

        let minRAM = Int(modelSizeGB * 1.2)
        let eligible = candidates.filter { $0.deviceProfile.availableRAMGB >= minRAM }

        let pool = eligible.isEmpty ? candidates : eligible

        switch strategy {
        case .bestFit:
            return bestFitNode(pool, modelSizeGB: modelSizeGB)
        case .leastLoaded:
            return pool.max(by: { $0.deviceProfile.availableRAMGB < $1.deviceProfile.availableRAMGB })
        case .fastest:
            return pool.max(by: { $0.deviceProfile.bandwidthGBs < $1.deviceProfile.bandwidthGBs })
        }
    }

    /// Builds a routing plan for an inference request.
    public static func routingPlan(
        for modelSizeGB: Double,
        in topology: MeshTopology
    ) -> RoutingPlan {
        if let single = bestNode(for: modelSizeGB, in: topology) {
            return RoutingPlan(
                mode: .singleNode,
                primaryNode: single,
                auxiliaryNodes: [],
                estimatedLatencyMs: estimateLatency(node: single, modelSizeGB: modelSizeGB)
            )
        }

        let all = (topology.tier1 + topology.tier2)
            .filter { $0.capability != .data }
            .sorted { $0.deviceProfile.totalRAMGB > $1.deviceProfile.totalRAMGB }

        if let fallback = all.first {
            return RoutingPlan(
                mode: .singleNode,
                primaryNode: fallback,
                auxiliaryNodes: [],
                estimatedLatencyMs: estimateLatency(node: fallback, modelSizeGB: modelSizeGB)
            )
        }

        return RoutingPlan(mode: .unavailable, primaryNode: nil, auxiliaryNodes: [], estimatedLatencyMs: 0)
    }

    private static func bestFitNode(_ candidates: [MeshNode], modelSizeGB: Double) -> MeshNode? {
        candidates.max { a, b in
            score(a, modelSizeGB: modelSizeGB) < score(b, modelSizeGB: modelSizeGB)
        }
    }

    private static func score(_ node: MeshNode, modelSizeGB: Double) -> Double {
        let ram = Double(node.deviceProfile.availableRAMGB)
        let bw = node.deviceProfile.bandwidthGBs

        let ramFit: Double = ram >= modelSizeGB ? (1.0 / (1.0 + (ram - modelSizeGB) * 0.1)) : (ram / modelSizeGB)

        let bwScore = min(bw / 800.0, 1.0)

        let thermalPenalty: Double = {
            switch node.deviceProfile.thermalState {
            case .nominal:  return 1.0
            case .fair:     return 0.8
            case .serious:  return 0.5
            case .critical: return 0.0
            }
        }()

        return (ramFit * 0.4 + bwScore * 0.4 + thermalPenalty * 0.2)
    }

    private static func estimateLatency(node: MeshNode, modelSizeGB: Double) -> Double {
        let bw = max(node.deviceProfile.bandwidthGBs, 1.0)
        let secondsPerToken = modelSizeGB / bw
        return secondsPerToken * 1000
    }
}

/// A routing decision for an inference request.
public struct RoutingPlan: Sendable {
    public let mode: Mode
    public let primaryNode: MeshNode?
    public let auxiliaryNodes: [MeshNode]
    public let estimatedLatencyMs: Double

    public enum Mode: String, Sendable {
        case singleNode
        case distributed
        case unavailable
    }
}
