// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Length-prefix frame codec used on top of mTLS-encrypted `NWConnection`.
///
/// Wire format (big-endian):
/// ```
/// ┌────────────┬──────────────────────────────┐
/// │ len (u32)  │ payload (len bytes)          │
/// └────────────┴──────────────────────────────┘
/// ```
///
/// Payload is either:
/// - UTF-8 encoded JSON (`Frame.json`) — ops, requests, responses, streaming control
/// - Raw binary (`Frame.binary`) — adapter download chunks, audio buffers, etc.
///
/// The codec does not know the payload kind from the wire — that is the caller's responsibility,
/// decided by the preceding JSON frame's metadata (e.g. a `"adapter_download"` response announces
/// subsequent binary chunks via `num_chunks`).
public enum FrameCodec {

    /// Maximum single frame size (safety limit against resource exhaustion).
    /// 64 MiB is generous enough for a single adapter chunk while still bounding worst-case buffering.
    public static let maxFrameBytes: Int = 64 * 1024 * 1024

    public enum Frame: Equatable {
        case json(Data)
        case binary(Data)

        public var bytes: Data {
            switch self {
            case .json(let d), .binary(let d): return d
            }
        }
    }

    /// Encode a payload into a framed `Data` suitable for `NWConnection.send`.
    public static func encode(_ payload: Data) throws -> Data {
        guard payload.count <= maxFrameBytes else {
            throw MeshError.frameTooLarge(payload.count)
        }
        var header = Data(count: 4)
        let len = UInt32(payload.count).bigEndian
        header.withUnsafeMutableBytes { raw in
            raw.baseAddress!.storeBytes(of: len, as: UInt32.self)
        }
        return header + payload
    }

    /// Encode a Codable JSON message.
    public static func encodeJSON<T: Encodable>(_ value: T, encoder: JSONEncoder = .init()) throws -> Data {
        let payload = try encoder.encode(value)
        return try encode(payload)
    }

    /// Streaming decoder — accumulates received bytes and yields complete frames.
    ///
    /// `NWConnection.receive(minimumIncompleteLength:maximumLength:)` returns arbitrary chunks;
    /// feed each chunk into `append(_:)`, then drain `pop()` until it returns nil.
    public final class Buffer {
        private var storage = Data()

        public init() {}

        public func append(_ chunk: Data) {
            storage.append(chunk)
        }

        /// Return the next complete framed payload, or nil if insufficient bytes buffered.
        /// Throws `MeshError.frameTooLarge` if the advertised length exceeds `maxFrameBytes`.
        public func pop() throws -> Data? {
            guard storage.count >= 4 else { return nil }

            let len = storage.prefix(4).withUnsafeBytes { raw -> UInt32 in
                raw.load(as: UInt32.self).bigEndian
            }
            let payloadLen = Int(len)

            guard payloadLen <= Self.maxAllowed else {
                throw MeshError.frameTooLarge(payloadLen)
            }
            guard storage.count >= 4 + payloadLen else { return nil }

            let payload = storage.subdata(in: 4..<(4 + payloadLen))
            storage.removeSubrange(0..<(4 + payloadLen))
            return payload
        }

        private static let maxAllowed = FrameCodec.maxFrameBytes

        public var bufferedBytes: Int { storage.count }

        public func reset() { storage.removeAll(keepingCapacity: true) }
    }
}
