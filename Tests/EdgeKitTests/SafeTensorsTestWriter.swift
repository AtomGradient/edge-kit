// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

enum SafeTensorsTestWriter {
    struct Tensor {
        var shape: [Int]
        var values: [Float]

        init(shape: [Int], values: [Float]) {
            self.shape = shape
            self.values = values
        }
    }

    static func writeF32(tensors: [String: Tensor], to url: URL) throws {
        var offset = 0
        var header: [String: Any] = [:]
        for name in tensors.keys.sorted() {
            let tensor = tensors[name]!
            let byteCount = tensor.values.count * MemoryLayout<Float>.size
            header[name] = [
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [offset, offset + byteCount],
            ]
            offset += byteCount
        }

        var headerData = try JSONSerialization.data(
            withJSONObject: header,
            options: [.sortedKeys]
        )
        let padding = (8 - (headerData.count % 8)) % 8
        if padding > 0 {
            headerData.append(Data(repeating: 0x20, count: padding))
        }

        var out = Data()
        var headerLength = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLength) { out.append(contentsOf: $0) }
        out.append(headerData)

        for name in tensors.keys.sorted() {
            for value in tensors[name]!.values {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { out.append(contentsOf: $0) }
            }
        }

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try out.write(to: url, options: .atomic)
    }
}
