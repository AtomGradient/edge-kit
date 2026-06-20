// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation

public enum RouteMatrixIntentScorerError: Error, Equatable, Sendable {
    case emptyIntentVocab
    case missingIntentMatrix
    case unsupportedMatrixDType(String)
    case matrixFileMissing(String)
    case matrixLoadFailed(String)
    case matrixTensorMissing(String, available: [String])
    case biasTensorMissing(String, available: [String])
    case matrixShapeMismatch(expected: [Int], actual: [Int], tensor: String)
    case weightsCountMismatch(expected: Int, actual: Int)
    case biasCountMismatch(expected: Int, actual: Int)
    case embeddingSizeMismatch(expected: Int, actual: Int)
    case zeroEmbedding
    case invalidTemperature(Double)
    case missingCalibrationThreshold(String)
    case invalidProbability
}

public struct RouteMatrixIntentPrediction: Codable, Equatable, Sendable {
    public var label: String
    public var intentTag: PersonalIntentTag?
    public var probability: Double
    public var threshold: Double
    public var thresholdPassed: Bool
    public var probabilitiesByIntent: [String: Double]

    public init(
        label: String,
        intentTag: PersonalIntentTag?,
        probability: Double,
        threshold: Double,
        thresholdPassed: Bool,
        probabilitiesByIntent: [String: Double]
    ) {
        self.label = label
        self.intentTag = intentTag
        self.probability = probability
        self.threshold = threshold
        self.thresholdPassed = thresholdPassed
        self.probabilitiesByIntent = probabilitiesByIntent
    }
}

/// Audit-only scorer for R2.1 route-matrix artifacts.
///
/// This type does not create route decisions or execute tools. It only turns a
/// caller-provided embedding into calibrated intent probabilities, so callers can
/// shadow-log matrix behavior before any runtime routing replacement exists.
public struct RouteMatrixIntentScorer: Sendable {
    public var hiddenSize: Int
    public var intentVocab: [String]
    public var weights: [Float]
    public var bias: [Float]

    public init(
        hiddenSize: Int,
        intentVocab: [String],
        weights: [Float],
        bias: [Float]
    ) throws {
        guard !intentVocab.isEmpty else {
            throw RouteMatrixIntentScorerError.emptyIntentVocab
        }
        let expectedWeights = hiddenSize * intentVocab.count
        guard weights.count == expectedWeights else {
            throw RouteMatrixIntentScorerError.weightsCountMismatch(
                expected: expectedWeights,
                actual: weights.count
            )
        }
        guard bias.count == intentVocab.count else {
            throw RouteMatrixIntentScorerError.biasCountMismatch(
                expected: intentVocab.count,
                actual: bias.count
            )
        }
        self.hiddenSize = hiddenSize
        self.intentVocab = intentVocab
        self.weights = weights
        self.bias = bias
    }

    public static func load(
        adapterDirectory: URL,
        manifest: RouteRouterManifest
    ) throws -> RouteMatrixIntentScorer {
        guard let spec = manifest.matrices["intent"] else {
            throw RouteMatrixIntentScorerError.missingIntentMatrix
        }
        let dtype = spec.dtype.lowercased()
        guard dtype == "float16" || dtype == "float32" else {
            throw RouteMatrixIntentScorerError.unsupportedMatrixDType(spec.dtype)
        }
        let matrixURL = adapterDirectory.appendingPathComponent(spec.file)
        guard FileManager.default.fileExists(atPath: matrixURL.path) else {
            throw RouteMatrixIntentScorerError.matrixFileMissing(spec.file)
        }

        let source: SafeTensorsShardFile
        do {
            source = try SafeTensorsShardFile(url: matrixURL)
        } catch {
            throw RouteMatrixIntentScorerError.matrixLoadFailed(String(describing: error))
        }
        let available = source.tensors.keys.sorted()
        guard let weightMetadata = source.tensors[spec.tensor] else {
            throw RouteMatrixIntentScorerError.matrixTensorMissing(
                spec.tensor,
                available: available
            )
        }
        guard weightMetadata.shape == spec.shape else {
            throw RouteMatrixIntentScorerError.matrixShapeMismatch(
                expected: spec.shape,
                actual: weightMetadata.shape,
                tensor: spec.tensor
            )
        }
        let expectedShape = [manifest.encoder.hiddenSize, manifest.intentVocab.count]
        guard spec.shape == expectedShape else {
            throw RouteMatrixIntentScorerError.matrixShapeMismatch(
                expected: expectedShape,
                actual: spec.shape,
                tensor: spec.tensor
            )
        }

        let biasValues: [Float]
        if let biasTensor = spec.biasTensor {
            guard let biasMetadata = source.tensors[biasTensor] else {
                throw RouteMatrixIntentScorerError.biasTensorMissing(
                    biasTensor,
                    available: available
                )
            }
            guard biasMetadata.shape == [manifest.intentVocab.count] else {
                throw RouteMatrixIntentScorerError.matrixShapeMismatch(
                    expected: [manifest.intentVocab.count],
                    actual: biasMetadata.shape,
                    tensor: biasTensor
                )
            }
            biasValues = try EdgeSafeTensorFloatReader.load1DFloatValues(
                named: biasTensor,
                from: source
            )
        } else {
            biasValues = Array(repeating: 0, count: manifest.intentVocab.count)
        }

        return try RouteMatrixIntentScorer(
            hiddenSize: manifest.encoder.hiddenSize,
            intentVocab: manifest.intentVocab,
            weights: try EdgeSafeTensorFloatReader.loadTensor(
                named: spec.tensor,
                from: source
            ).values,
            bias: biasValues
        )
    }

    public func predict(
        embedding rawEmbedding: [Float],
        calibration: RouteRouterCalibration,
        normalizeEmbedding: Bool = true
    ) throws -> RouteMatrixIntentPrediction {
        guard rawEmbedding.count == hiddenSize else {
            throw RouteMatrixIntentScorerError.embeddingSizeMismatch(
                expected: hiddenSize,
                actual: rawEmbedding.count
            )
        }
        guard calibration.intentTemperature > 0 else {
            throw RouteMatrixIntentScorerError.invalidTemperature(calibration.intentTemperature)
        }

        let embedding = try normalizeEmbedding ? Self.normalized(rawEmbedding) : rawEmbedding
        let classCount = intentVocab.count
        var logits = bias.map(Double.init)
        for row in 0..<hiddenSize {
            let value = Double(embedding[row])
            let offset = row * classCount
            for col in 0..<classCount {
                logits[col] += value * Double(weights[offset + col])
            }
        }

        let probabilities = try Self.softmax(
            logits.map { $0 / calibration.intentTemperature }
        )
        guard let topIndex = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) else {
            throw RouteMatrixIntentScorerError.invalidProbability
        }
        let label = intentVocab[topIndex]
        guard let threshold = calibration.intentThresholds[label] else {
            throw RouteMatrixIntentScorerError.missingCalibrationThreshold(label)
        }

        var byIntent: [String: Double] = [:]
        for (index, intent) in intentVocab.enumerated() {
            byIntent[intent] = probabilities[index]
        }
        return RouteMatrixIntentPrediction(
            label: label,
            intentTag: PersonalIntentTag(rawValue: label),
            probability: probabilities[topIndex],
            threshold: threshold,
            thresholdPassed: probabilities[topIndex] >= threshold,
            probabilitiesByIntent: byIntent
        )
    }

    private static func normalized(_ values: [Float]) throws -> [Float] {
        let norm = sqrt(values.reduce(Double(0)) { sum, value in
            sum + Double(value) * Double(value)
        })
        guard norm > 1e-12 else {
            throw RouteMatrixIntentScorerError.zeroEmbedding
        }
        return values.map { Float(Double($0) / norm) }
    }

    private static func softmax(_ values: [Double]) throws -> [Double] {
        guard let maxValue = values.max(), maxValue.isFinite else {
            throw RouteMatrixIntentScorerError.invalidProbability
        }
        let exps = values.map { exp($0 - maxValue) }
        let denom = exps.reduce(0, +)
        guard denom.isFinite, denom > 0 else {
            throw RouteMatrixIntentScorerError.invalidProbability
        }
        return exps.map { $0 / denom }
    }
}
