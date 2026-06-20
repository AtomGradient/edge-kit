// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeEngine
import Foundation

enum EdgeSafeTensorFloatReaderError: Error, Equatable {
    case unsupportedDType(tensor: String, dtype: String)
    case byteCountMismatch(tensor: String, expected: Int, actual: Int)
    case rankMismatch(tensor: String, expected: Int, actual: Int)
}

struct EdgeSafeTensorFloatTensor: Sendable {
    var name: String
    var dtype: String
    var shape: [Int]
    var values: [Float]
}

enum EdgeSafeTensorFloatReader {
    static func loadTensor(
        named name: String,
        from source: SafeTensorsSource
    ) throws -> EdgeSafeTensorFloatTensor {
        let metadata = try source.metadata(named: name)
        let values = try floatValues(
            from: source.tensorSlice(named: name),
            metadata: metadata
        )
        return EdgeSafeTensorFloatTensor(
            name: name,
            dtype: metadata.dtype,
            shape: metadata.shape,
            values: values
        )
    }

    static func load1DFloatValues(
        named name: String,
        from source: SafeTensorsSource
    ) throws -> [Float] {
        let tensor = try loadTensor(named: name, from: source)
        guard tensor.shape.count == 1 else {
            throw EdgeSafeTensorFloatReaderError.rankMismatch(
                tensor: name,
                expected: 1,
                actual: tensor.shape.count
            )
        }
        return tensor.values
    }

    static func floatValues(
        from data: SafeTensorDataSlice,
        metadata: SafeTensorMetadata
    ) throws -> [Float] {
        switch metadata.dtype {
        case "F32":
            return try typedValues(
                from: data,
                metadata: metadata,
                stride: MemoryLayout<UInt32>.stride
            ) { bytes, offset in
                let bits = bytes.loadUnaligned(
                    fromByteOffset: offset,
                    as: UInt32.self
                )
                return Float(bitPattern: UInt32(littleEndian: bits))
            }

        case "F16":
            return try typedValues(
                from: data,
                metadata: metadata,
                stride: MemoryLayout<UInt16>.stride
            ) { bytes, offset in
                let bits = bytes.loadUnaligned(
                    fromByteOffset: offset,
                    as: UInt16.self
                )
                return Float(Float16(bitPattern: UInt16(littleEndian: bits)))
            }

        case "BF16":
            return try typedValues(
                from: data,
                metadata: metadata,
                stride: MemoryLayout<UInt16>.stride
            ) { bytes, offset in
                let bits = bytes.loadUnaligned(
                    fromByteOffset: offset,
                    as: UInt16.self
                )
                return Float(bitPattern: UInt32(UInt16(littleEndian: bits)) << 16)
            }

        default:
            throw EdgeSafeTensorFloatReaderError.unsupportedDType(
                tensor: metadata.name,
                dtype: metadata.dtype
            )
        }
    }

    private static func typedValues(
        from data: SafeTensorDataSlice,
        metadata: SafeTensorMetadata,
        stride: Int,
        convert: (UnsafeRawBufferPointer, Int) -> Float
    ) throws -> [Float] {
        let expectedByteCount = metadata.elementCount * stride
        guard data.count == expectedByteCount else {
            throw EdgeSafeTensorFloatReaderError.byteCountMismatch(
                tensor: metadata.name,
                expected: expectedByteCount,
                actual: data.count
            )
        }
        return data.withUnsafeBytes { bytes in
            (0..<metadata.elementCount).map { index in
                convert(bytes, index * stride)
            }
        }
    }
}
