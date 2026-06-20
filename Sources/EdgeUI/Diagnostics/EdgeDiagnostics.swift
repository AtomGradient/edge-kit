// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

public final class EdgeDiagnostics: ObservableObject {
    public static let shared = EdgeDiagnostics()

    private static let detailedMetricsKey = "com.atomgradient.edge.diagnostics.detailedMetricsEnabled"

    @Published public private(set) var isDetailedMetricsEnabled: Bool {
        didSet {
            UserDefaults.standard.set(isDetailedMetricsEnabled, forKey: Self.detailedMetricsKey)
        }
    }

    private init() {
        isDetailedMetricsEnabled = UserDefaults.standard.bool(forKey: Self.detailedMetricsKey)
    }

    public func setDetailedMetricsEnabled(_ enabled: Bool) {
        isDetailedMetricsEnabled = enabled
    }

    @discardableResult
    public func toggleDetailedMetrics() -> Bool {
        isDetailedMetricsEnabled.toggle()
        return isDetailedMetricsEnabled
    }
}

public struct EdgeDiagnosticTapModifier: ViewModifier {
    @ObservedObject private var diagnostics = EdgeDiagnostics.shared
    @State private var tapCount = 0
    @State private var lastTapAt: Date?
    @State private var toastMessage: String?
    @State private var toastTask: Task<Void, Never>?

    private let requiredTapCount: Int
    private let maximumTapInterval: TimeInterval

    public init(requiredTapCount: Int = 5, maximumTapInterval: TimeInterval = 0.5) {
        self.requiredTapCount = requiredTapCount
        self.maximumTapInterval = maximumTapInterval
    }

    public func body(content: Content) -> some View {
        content
            .contentShape(Rectangle())
            .onTapGesture {
                handleTap()
            }
            .overlay(alignment: .topTrailing) {
                if let toastMessage {
                    Text(toastMessage)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(.black.opacity(0.82), in: Capsule())
                        .offset(y: -28)
                        .transition(.opacity.combined(with: .scale(scale: 0.95)))
                        .allowsHitTesting(false)
                }
            }
    }

    private func handleTap() {
        let now = Date()
        if let lastTapAt, now.timeIntervalSince(lastTapAt) <= maximumTapInterval {
            tapCount += 1
        } else {
            tapCount = 1
        }
        lastTapAt = now

        guard tapCount >= requiredTapCount else { return }
        tapCount = 0
        lastTapAt = nil

        let enabled = diagnostics.toggleDetailedMetrics()
        fireHaptic()
        showToast("Developer metrics \(enabled ? "ON" : "OFF")")
    }

    private func showToast(_ message: String) {
        toastTask?.cancel()
        withAnimation(.easeOut(duration: 0.16)) {
            toastMessage = message
        }
        toastTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !Task.isCancelled else { return }
            withAnimation(.easeIn(duration: 0.18)) {
                toastMessage = nil
            }
        }
    }

    private func fireHaptic() {
        #if canImport(UIKit)
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        #endif
    }
}

public extension View {
    func edgeDiagnosticTapGesture(
        requiredTapCount: Int = 5,
        maximumTapInterval: TimeInterval = 0.5
    ) -> some View {
        modifier(EdgeDiagnosticTapModifier(
            requiredTapCount: requiredTapCount,
            maximumTapInterval: maximumTapInterval
        ))
    }
}
