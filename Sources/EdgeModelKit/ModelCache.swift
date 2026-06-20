// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

/// Local model cache manager.
public final class ModelCache {

    public static let shared = ModelCache()

    private let cacheRoot: URL

    private init() {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        cacheRoot = caches.appendingPathComponent("AtomGradient/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    }

    /// Returns the cache directory for a model configuration.
    public func cachedURL(for config: ModelConfig) -> URL {
        cacheRoot.appendingPathComponent(config.modelID, isDirectory: true)
    }

    /// Returns whether a model configuration has a local cache directory.
    public func isCached(_ config: ModelConfig) -> Bool {
        let url = cachedURL(for: config)
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Returns the cached size for a model configuration.
    public func cacheSize(for config: ModelConfig) -> Int64 {
        let url = cachedURL(for: config)
        return directorySize(at: url)
    }

    /// Returns the total size of the model cache.
    public func totalCacheSize() -> Int64 {
        directorySize(at: cacheRoot)
    }

    /// Removes the cached files for a model configuration.
    public func evict(_ config: ModelConfig) throws {
        let url = cachedURL(for: config)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    /// Removes all cached model files.
    public func evictAll() throws {
        let contents = try FileManager.default.contentsOfDirectory(
            at: cacheRoot, includingPropertiesForKeys: nil
        )
        for url in contents {
            try FileManager.default.removeItem(at: url)
        }
    }

    private func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }
}
