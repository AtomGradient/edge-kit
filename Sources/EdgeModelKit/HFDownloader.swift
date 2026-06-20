// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import EdgeInference

/// Downloads Hugging Face Hub model files into the local EdgeKit model cache.
public final class HFDownloader: NSObject {

    public static let shared = HFDownloader()

    public typealias ProgressHandler = @Sendable (Double) -> Void

    private var activeTasks: [String: URLSessionDownloadTask] = [:]
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(withIdentifier: "com.atomgradient.hfdownloader")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private override init() {}

    /// Downloads a model bundle into the local cache.
    public func download(
        config: ModelConfig,
        progress: ProgressHandler? = nil
    ) async throws {
        let destination = ModelCache.shared.cachedURL(for: config)

        if ModelCache.shared.isCached(config) { return }

        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

        let files = try await fetchFileList(repo: config.huggingFaceRepo)

        for (index, file) in files.enumerated() {
            let fileURL = destination.appendingPathComponent(file.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) { continue }

            let downloadURL = huggingFaceURL(repo: config.huggingFaceRepo, file: file.filename)
            try await downloadFile(from: downloadURL, to: fileURL) { p in
                let overall = (Double(index) + p) / Double(files.count)
                progress?(overall)
            }
        }
    }

    public func cancel(modelID: String) {
        activeTasks[modelID]?.cancel()
        activeTasks.removeValue(forKey: modelID)
    }

    private struct HFFile: Decodable {
        let filename: String
        let size: Int?
    }

    private func fetchFileList(repo: String) async throws -> [HFFile] {
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repo)")!
        let (data, _) = try await URLSession.shared.data(from: apiURL)
        struct Response: Decodable { let siblings: [HFFile] }
        let response = try JSONDecoder().decode(Response.self, from: data)
        return response.siblings
    }

    private func huggingFaceURL(repo: String, file: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
    }

    private func downloadFile(
        from url: URL,
        to destination: URL,
        progress: ProgressHandler?
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let task = session.downloadTask(with: url)
            task.resume()
        }
    }
}

extension HFDownloader: URLSessionDownloadDelegate {
    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let dest = downloadTask.currentRequest?.url.flatMap({ url -> URL? in
            return nil
        }) else { return }
        try? FileManager.default.moveItem(at: location, to: dest)
    }

    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        _ = progress
    }
}
