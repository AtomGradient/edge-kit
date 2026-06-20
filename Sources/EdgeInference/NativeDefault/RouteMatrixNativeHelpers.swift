// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct RouteMatrixHiddenState: Sendable, Equatable {
    public var shape: [Int]
    public var values: [Float]

    public init(shape: [Int], values: [Float]) {
        self.shape = shape
        self.values = values
    }
}

internal func routeMatrixResolvedCaptureLayerIndex(_ layerIndex: Int) throws -> Int {
    if layerIndex >= 0 {
        return layerIndex
    }
    if layerIndex == -1 {
        throw EdgeRuntimeError.loadFailed(
            "route matrix encoder layer_index=-1 requires final hidden capture; train an explicit non-negative layer_index artifact or add final hidden support to edge-engine"
        )
    }
    throw EdgeRuntimeError.loadFailed(
        "route matrix encoder layer index unsupported: \(layerIndex)"
    )
}

internal func routeMatrixMeanPooledEmbedding(
    from hidden: RouteMatrixHiddenState,
    selectedTokenIndices: [Int],
    expectedHiddenSize: Int
) throws -> [Float] {
    let seqLen: Int
    let hiddenSize: Int
    let flat: [Float]

    switch hidden.shape.count {
    case 3:
        guard hidden.shape[0] == 1 else {
            throw EdgeRuntimeError.loadFailed(
                "route matrix hidden batch unsupported: expected 1, got \(hidden.shape[0])"
            )
        }
        seqLen = hidden.shape[1]
        hiddenSize = hidden.shape[2]
        flat = hidden.values
    case 2:
        seqLen = hidden.shape[0]
        hiddenSize = hidden.shape[1]
        flat = hidden.values
    default:
        throw EdgeRuntimeError.loadFailed(
            "route matrix hidden rank unsupported: expected 2D or 3D, got shape \(hidden.shape)"
        )
    }

    guard seqLen > 0 else {
        throw EdgeRuntimeError.loadFailed("route matrix hidden sequence is empty")
    }
    guard hiddenSize == expectedHiddenSize else {
        throw EdgeRuntimeError.loadFailed(
            "route matrix hidden size mismatch: expected \(expectedHiddenSize), got \(hiddenSize), shape \(hidden.shape)"
        )
    }
    guard flat.count == seqLen * hiddenSize else {
        throw EdgeRuntimeError.loadFailed(
            "route matrix hidden flatten mismatch: expected \(seqLen * hiddenSize), got \(flat.count), shape \(hidden.shape)"
        )
    }

    let validIndices = selectedTokenIndices.filter { $0 >= 0 && $0 < seqLen }
    let poolIndices = validIndices.isEmpty ? Array(0..<seqLen) : validIndices
    var pooled = [Float](repeating: 0, count: hiddenSize)
    for tokenIndex in poolIndices {
        let offset = tokenIndex * hiddenSize
        for column in 0..<hiddenSize {
            pooled[column] += flat[offset + column]
        }
    }
    let divisor = Float(poolIndices.count)
    for column in 0..<hiddenSize {
        pooled[column] /= divisor
    }
    return pooled
}
