// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum RouteRouterManifestError: Error, Equatable, Sendable {
    case unsupportedSchemaVersion(String)
    case unsupportedRouterType(String)
    case missingIntentMatrix
    case invalidIntentMatrixShape([Int])
    case baseModelMismatch(expected: String, actual: String)
    case tokenizerMismatch(expected: String, actual: String)
    case runtimeVersionUnsupported(minimum: String, actual: String)
}

public struct RouteRouterEncoderSpec: Codable, Equatable, Sendable {
    public var kind: String
    public var hiddenSize: Int
    public var layerIndex: Int
    public var pooling: String
    public var baseModelID: String
    public var tokenizerSHA256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case hiddenSize = "hidden_size"
        case layerIndex = "layer_index"
        case pooling
        case baseModelID = "base_model_id"
        case tokenizerSHA256 = "tokenizer_sha256"
    }

    public init(
        kind: String,
        hiddenSize: Int,
        layerIndex: Int,
        pooling: String,
        baseModelID: String,
        tokenizerSHA256: String
    ) {
        self.kind = kind
        self.hiddenSize = hiddenSize
        self.layerIndex = layerIndex
        self.pooling = pooling
        self.baseModelID = baseModelID
        self.tokenizerSHA256 = tokenizerSHA256
    }
}

public struct RouteRouterMatrixSpec: Codable, Equatable, Sendable {
    public var file: String
    public var tensor: String
    public var biasTensor: String?
    public var shape: [Int]
    public var dtype: String

    enum CodingKeys: String, CodingKey {
        case file
        case tensor
        case biasTensor = "bias_tensor"
        case shape
        case dtype
    }

    public init(
        file: String,
        tensor: String,
        biasTensor: String? = nil,
        shape: [Int],
        dtype: String
    ) {
        self.file = file
        self.tensor = tensor
        self.biasTensor = biasTensor
        self.shape = shape
        self.dtype = dtype
    }
}

/// R2.1 matrix-router artifact manifest.
///
/// This is a contract reader only. Runtime matrix routing is introduced later;
/// unsupported or mismatched manifests should fall back to evidence/base routing.
public struct RouteRouterManifest: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.route_router_manifest.v0"
    public static let supportedRouterType = "matrix_v0"

    public var schemaVersion: String
    public var routerType: String
    public var encoder: RouteRouterEncoderSpec
    public var intentVocab: [String]
    public var matrices: [String: RouteRouterMatrixSpec]
    public var calibrationFile: String
    public var minRuntimeVersion: String
    public var trainingRunID: String
    public var manifestSHA256: String
    public var fallbackChain: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case routerType = "router_type"
        case encoder
        case intentVocab = "intent_vocab"
        case matrices
        case calibrationFile = "calibration_file"
        case minRuntimeVersion = "min_runtime_version"
        case trainingRunID = "training_run_id"
        case manifestSHA256 = "manifest_sha256"
        case fallbackChain = "fallback_chain"
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        routerType: String = Self.supportedRouterType,
        encoder: RouteRouterEncoderSpec,
        intentVocab: [String],
        matrices: [String: RouteRouterMatrixSpec],
        calibrationFile: String,
        minRuntimeVersion: String,
        trainingRunID: String,
        manifestSHA256: String,
        fallbackChain: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.routerType = routerType
        self.encoder = encoder
        self.intentVocab = intentVocab
        self.matrices = matrices
        self.calibrationFile = calibrationFile
        self.minRuntimeVersion = minRuntimeVersion
        self.trainingRunID = trainingRunID
        self.manifestSHA256 = manifestSHA256
        self.fallbackChain = fallbackChain
    }

    public static func load(from url: URL) throws -> RouteRouterManifest {
        let data = try Data(contentsOf: url)
        let manifest = try JSONDecoder().decode(RouteRouterManifest.self, from: data)
        try manifest.validate()
        return manifest
    }

    public func validate(
        expectedBaseModelID: String? = nil,
        expectedTokenizerSHA256: String? = nil,
        currentRuntimeVersion: String? = nil
    ) throws {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw RouteRouterManifestError.unsupportedSchemaVersion(schemaVersion)
        }
        guard routerType == Self.supportedRouterType else {
            throw RouteRouterManifestError.unsupportedRouterType(routerType)
        }
        guard let intentMatrix = matrices["intent"] else {
            throw RouteRouterManifestError.missingIntentMatrix
        }
        guard intentMatrix.shape == [encoder.hiddenSize, intentVocab.count] else {
            throw RouteRouterManifestError.invalidIntentMatrixShape(intentMatrix.shape)
        }
        if let expectedBaseModelID, expectedBaseModelID != encoder.baseModelID {
            throw RouteRouterManifestError.baseModelMismatch(
                expected: expectedBaseModelID,
                actual: encoder.baseModelID
            )
        }
        if let expectedTokenizerSHA256, expectedTokenizerSHA256 != encoder.tokenizerSHA256 {
            throw RouteRouterManifestError.tokenizerMismatch(
                expected: expectedTokenizerSHA256,
                actual: encoder.tokenizerSHA256
            )
        }
        if let currentRuntimeVersion,
           Self.compareVersion(currentRuntimeVersion, minRuntimeVersion) == .orderedAscending {
            throw RouteRouterManifestError.runtimeVersionUnsupported(
                minimum: minRuntimeVersion,
                actual: currentRuntimeVersion
            )
        }
    }

    private static func compareVersion(_ lhs: String, _ rhs: String) -> ComparisonResult {
        let left = versionComponents(lhs)
        let right = versionComponents(rhs)
        let count = max(left.count, right.count)
        for index in 0..<count {
            let a = index < left.count ? left[index] : 0
            let b = index < right.count ? right[index] : 0
            if a < b { return .orderedAscending }
            if a > b { return .orderedDescending }
        }
        return .orderedSame
    }

    private static func versionComponents(_ value: String) -> [Int] {
        value
            .split(separator: ".")
            .map { part in
                let digits = part.prefix { $0.isNumber }
                return Int(digits) ?? 0
            }
    }
}

public enum RouteRouterCalibrationError: Error, Equatable, Sendable {
    case unsupportedSchemaVersion(String)
    case emptyIntentThresholds
    case missingIntentThreshold(String)
    case invalidTemperature(Double)
    case invalidIntentThreshold(label: String, value: Double)
    case invalidToolThresholdDefault(Double)
    case invalidCalibrationSetSize(Int)
    case invalidECE(Double)
}

public struct RouteRouterCalibration: Codable, Equatable, Sendable {
    public static let supportedSchemaVersion = "edgestudio.route_router_calibration.v0"

    public var schemaVersion: String
    public var intentTemperature: Double
    public var intentThresholds: [String: Double]
    public var toolThresholdDefault: Double
    public var calibrationSetSize: Int
    public var calibrationECE: Double?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case intentTemperature = "intent_temperature"
        case intentThresholds = "intent_thresholds"
        case toolThresholdDefault = "tool_threshold_default"
        case calibrationSetSize = "calibration_set_size"
        case calibrationECE = "calibration_ece"
    }

    public init(
        schemaVersion: String = Self.supportedSchemaVersion,
        intentTemperature: Double,
        intentThresholds: [String: Double],
        toolThresholdDefault: Double,
        calibrationSetSize: Int,
        calibrationECE: Double?
    ) {
        self.schemaVersion = schemaVersion
        self.intentTemperature = intentTemperature
        self.intentThresholds = intentThresholds
        self.toolThresholdDefault = toolThresholdDefault
        self.calibrationSetSize = calibrationSetSize
        self.calibrationECE = calibrationECE
    }

    public static func load(from url: URL) throws -> RouteRouterCalibration {
        let data = try Data(contentsOf: url)
        let calibration = try JSONDecoder().decode(RouteRouterCalibration.self, from: data)
        try calibration.validate(intentVocab: [])
        return calibration
    }

    public func validate(intentVocab: [String]) throws {
        guard schemaVersion == Self.supportedSchemaVersion else {
            throw RouteRouterCalibrationError.unsupportedSchemaVersion(schemaVersion)
        }
        guard intentTemperature > 0 else {
            throw RouteRouterCalibrationError.invalidTemperature(intentTemperature)
        }
        guard !intentThresholds.isEmpty else {
            throw RouteRouterCalibrationError.emptyIntentThresholds
        }
        for (label, threshold) in intentThresholds where threshold < 0 || threshold > 1 {
            throw RouteRouterCalibrationError.invalidIntentThreshold(label: label, value: threshold)
        }
        guard toolThresholdDefault >= 0, toolThresholdDefault <= 1 else {
            throw RouteRouterCalibrationError.invalidToolThresholdDefault(toolThresholdDefault)
        }
        guard calibrationSetSize >= 0 else {
            throw RouteRouterCalibrationError.invalidCalibrationSetSize(calibrationSetSize)
        }
        if let calibrationECE, calibrationECE < 0 || calibrationECE > 1 {
            throw RouteRouterCalibrationError.invalidECE(calibrationECE)
        }
        for label in intentVocab where intentThresholds[label] == nil {
            throw RouteRouterCalibrationError.missingIntentThreshold(label)
        }
    }
}
