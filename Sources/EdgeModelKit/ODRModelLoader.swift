// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import Foundation

#if canImport(UIKit)
import UIKit

/// On-Demand Resources model loader for iOS model bundles.
@MainActor
public final class ODRModelLoader: ObservableObject {
    public typealias ProgressHandler = @MainActor @Sendable (Double) -> Void

    @Published public private(set) var isLoading: Bool
    @Published public private(set) var progress: Double
    @Published public private(set) var error: String?

    private var currentRequest: NSBundleResourceRequest?
    private var currentTags: Set<String> = []

    public init() {
        self.isLoading = false
        self.progress = 0
        self.error = nil
    }

    /// Checks whether the tagged ODR resource can be accessed without starting a download.
    public func isAvailable(tag: String) async -> Bool {
        await isAvailable(tags: [tag])
    }

    /// Checks whether the tagged ODR resources can be accessed without starting a download.
    public func isAvailable(tags: Set<String>) async -> Bool {
        let request = NSBundleResourceRequest(tags: tags)
        let available = await request.conditionallyBeginAccessingResources()
        if available {
            request.endAccessingResources()
        }
        return available
    }

    /// Downloads or opens a tagged ODR resource and returns its bundle.
    public func download(
        tag: String,
        onProgress: ProgressHandler? = nil
    ) async throws -> Bundle {
        try await download(tags: [tag], onProgress: onProgress)
    }

    /// Downloads or opens tagged ODR resources and returns their bundle.
    public func download(
        tags: Set<String>,
        onProgress: ProgressHandler? = nil
    ) async throws -> Bundle {
        isLoading = true
        progress = 0
        error = nil

        let request = NSBundleResourceRequest(tags: tags)
        request.loadingPriority = NSBundleResourceRequestLoadingPriorityUrgent
        currentRequest = request
        currentTags = tags

        let observation = request.progress.observe(\.fractionCompleted) { [weak self] observedProgress, _ in
            let fraction = observedProgress.fractionCompleted
            Task { @MainActor [weak self] in
                self?.progress = fraction
                onProgress?(fraction)
            }
        }

        defer {
            observation.invalidate()
            isLoading = false
        }

        do {
            try await request.beginAccessingResources()
            progress = 1.0
            onProgress?(1.0)
            return request.bundle
        } catch {
            self.error = error.localizedDescription
            throw error
        }
    }

    /// Cancels the current ODR download request.
    public func cancel() {
        currentRequest?.progress.cancel()
        currentRequest = nil
        currentTags.removeAll()
        isLoading = false
    }

    /// Releases the current ODR access lease when it covers the supplied tag.
    public func endAccessing(tag: String) {
        endAccessing(tags: [tag])
    }

    /// Releases the current ODR access lease when it covers the supplied tags.
    public func endAccessing(tags: Set<String>) {
        guard currentTags == tags || !currentTags.isDisjoint(with: tags) else { return }
        currentRequest?.endAccessingResources()
        currentRequest = nil
        currentTags.removeAll()
    }

    /// Marks cached ODR resources as disposable and releases the active lease if it matches.
    public func invalidateCache(tag: String) {
        invalidateCache(tags: [tag])
    }

    /// Marks cached ODR resources as disposable and releases the active lease if it matches.
    public func invalidateCache(tags: Set<String>) {
        endAccessing(tags: tags)
        Bundle.main.setPreservationPriority(0, forTags: tags)
    }
}
#endif
