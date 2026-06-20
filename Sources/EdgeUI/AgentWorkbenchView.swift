// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

//
//  AgentWorkbenchView.swift
//  EdgeUI
//
//  Generic agent transcript and activity chrome for Edge-based apps.
//

import SwiftUI

#if os(iOS)
import UIKit

public struct EdgeActivityStatus: Equatable, Sendable {
    public let title: String
    public let detail: String
    public let systemImage: String
    public let actionLabel: String

    public init(
        title: String,
        detail: String,
        systemImage: String,
        actionLabel: String? = nil
    ) {
        self.title = title
        self.detail = detail
        self.systemImage = systemImage
        self.actionLabel = actionLabel ?? title
    }
}

public enum EdgeActivityMotion: Equatable, Sendable {
    case standard
    case lightweight

    fileprivate var updateInterval: TimeInterval {
        switch self {
        case .standard: return 0.22
        case .lightweight: return 1.0
        }
    }

    fileprivate var pulseRate: Double {
        switch self {
        case .standard: return 3.0
        case .lightweight: return 1.0
        }
    }

    fileprivate var usesShimmer: Bool {
        switch self {
        case .standard: return true
        case .lightweight: return false
        }
    }
}

public struct EdgeActivityStatusView: View {
    private let status: EdgeActivityStatus
    private let startedAt: Date?
    private let compact: Bool
    private let outputTokens: Int
    private let metricLabel: String?
    private let isActive: Bool
    private let accentColor: Color
    private let motion: EdgeActivityMotion

    public init(
        status: EdgeActivityStatus,
        startedAt: Date?,
        compact: Bool = false,
        outputTokens: Int = 0,
        metricLabel: String? = nil,
        isActive: Bool = true,
        accentColor: Color = .indigo,
        motion: EdgeActivityMotion = .standard
    ) {
        self.status = status
        self.startedAt = startedAt
        self.compact = compact
        self.outputTokens = outputTokens
        self.metricLabel = metricLabel
        self.isActive = isActive
        self.accentColor = accentColor
        self.motion = motion
    }

    public var body: some View {
        TimelineView(.periodic(from: .now, by: motion.updateInterval)) { context in
            let seconds = elapsedSeconds(at: context.date)
            let phase = pulsePhase(at: context.date)
            let sweep = shimmerPhase(at: context.date)
            content(seconds: seconds, phase: phase, shimmerPhase: sweep)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(status.title), \(seconds) seconds, \(progressLabel), \(status.detail)")
        }
    }

    @ViewBuilder
    private func content(seconds: Int, phase: Int, shimmerPhase: Double) -> some View {
        if compact {
            HStack(spacing: 6) {
                EdgeActivityPulseView(
                    systemImage: status.systemImage,
                    phase: phase,
                    compact: true,
                    accentColor: accentColor
                )
                Text(status.actionLabel)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
                Text("\(seconds)s · \(progressLabel)")
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .lineLimit(1)
            }
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .edgeActivityShimmer(phase: shimmerPhase, active: isActive && motion.usesShimmer, accentColor: accentColor)
        } else {
            HStack(spacing: 10) {
                EdgeActivityPulseView(
                    systemImage: status.systemImage,
                    phase: phase,
                    compact: false,
                    accentColor: accentColor
                )
                VStack(alignment: .leading, spacing: 3) {
                    Text(status.actionLabel)
                        .font(.system(size: 14, weight: .semibold, design: .rounded))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.9)

                    HStack(spacing: 8) {
                        EdgeActivityMetaLabel(text: "\(seconds)s")
                        EdgeActivityMetaLabel(text: progressLabel)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(accentColor.opacity(0.08), in: Capsule())
            .overlay {
                Capsule()
                    .stroke(accentColor.opacity(0.14), lineWidth: 1)
            }
            .edgeActivityShimmer(phase: shimmerPhase, active: isActive && motion.usesShimmer, accentColor: accentColor)
        }
    }

    private var progressLabel: String {
        metricLabel ?? "↑ \(outputTokens) tokens"
    }

    private func elapsedSeconds(at date: Date) -> Int {
        guard let startedAt else { return 0 }
        return max(0, Int(date.timeIntervalSince(startedAt)))
    }

    private func pulsePhase(at date: Date) -> Int {
        Int(date.timeIntervalSinceReferenceDate * motion.pulseRate).edgeModulo(3)
    }

    private func shimmerPhase(at date: Date) -> Double {
        let cycle = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 1.8)
        return cycle / 1.8
    }
}

public struct EdgeActivityPulseView: View {
    private let systemImage: String
    private let phase: Int
    private let compact: Bool
    private let accentColor: Color

    public init(systemImage: String, phase: Int, compact: Bool, accentColor: Color = .indigo) {
        self.systemImage = systemImage
        self.phase = phase
        self.compact = compact
        self.accentColor = accentColor
    }

    public var body: some View {
        HStack(spacing: compact ? 2 : 3) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(accentColor)
                    .frame(width: dotSize(for: index), height: dotSize(for: index))
                    .opacity(index == phase ? 0.95 : 0.28)
                    .scaleEffect(index == phase ? 1.12 : 0.82)
                    .animation(.easeInOut(duration: 0.22), value: phase)
            }
        }
        .frame(width: compact ? 18 : 24, height: compact ? 14 : 18)
        .overlay(alignment: .center) {
            if !compact {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(accentColor.opacity(0.72))
                    .offset(y: -12)
            }
        }
    }

    private func dotSize(for index: Int) -> CGFloat {
        let base: CGFloat = compact ? 3.4 : 4.4
        return index == phase ? base + 1.2 : base
    }
}

public struct EdgeActivityMetaLabel: View {
    private let text: String

    public init(text: String) {
        self.text = text
    }

    public var body: some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .foregroundStyle(.secondary)
            .monospacedDigit()
            .contentTransition(.numericText())
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(.primary.opacity(0.05), in: Capsule())
    }
}

public struct EdgeActivityShimmer: ViewModifier {
    private let phase: Double
    private let active: Bool
    private let accentColor: Color

    public init(phase: Double, active: Bool, accentColor: Color = .indigo) {
        self.phase = phase
        self.active = active
        self.accentColor = accentColor
    }

    public func body(content: Content) -> some View {
        if active {
            content
                .overlay {
                    GeometryReader { proxy in
                        let width = max(proxy.size.width, 1)
                        LinearGradient(
                            colors: [
                                .clear,
                                .white.opacity(0.16),
                                accentColor.opacity(0.18),
                                .clear,
                            ],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                        .frame(width: max(42, width * 0.38))
                        .offset(x: -width * 0.45 + width * 1.25 * phase)
                    }
                    .allowsHitTesting(false)
                    .mask(content)
                }
        } else {
            content
        }
    }
}

public extension View {
    func edgeActivityShimmer(phase: Double, active: Bool, accentColor: Color = .indigo) -> some View {
        modifier(EdgeActivityShimmer(phase: phase, active: active, accentColor: accentColor))
    }
}

public struct EdgeAgentTranscriptRow<Actions: View, ToolTraceContent: View, Content: View>: View {
    private let title: String
    private let systemImage: String
    private let timestamp: Date?
    private let accentColor: Color
    private let sideBarOpacity: Double
    private let copyText: String?
    private let contentLeadingPadding: CGFloat
    private let actions: Actions
    private let toolTraceContent: ToolTraceContent
    private let content: Content

    public init(
        title: String,
        systemImage: String,
        timestamp: Date? = nil,
        accentColor: Color = .indigo,
        sideBarOpacity: Double = 0.45,
        copyText: String? = nil,
        contentLeadingPadding: CGFloat = 26,
        @ViewBuilder actions: () -> Actions,
        @ViewBuilder toolTraceContent: () -> ToolTraceContent,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.systemImage = systemImage
        self.timestamp = timestamp
        self.accentColor = accentColor
        self.sideBarOpacity = sideBarOpacity
        self.copyText = copyText
        self.contentLeadingPadding = contentLeadingPadding
        self.actions = actions()
        self.toolTraceContent = toolTraceContent()
        self.content = content()
    }

    public var body: some View {
        row
            .contextMenu {
                if let copyText, !copyText.isEmpty {
                    Button {
                        UIPasteboard.general.string = copyText
                    } label: {
                        Label("Copy", systemImage: "doc.on.doc")
                    }
                    ShareLink(item: copyText) {
                        Label("Share", systemImage: "square.and.arrow.up")
                    }
                }
            }
    }

    private var row: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(accentColor)
                    .frame(width: 18)

                Text(title)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(.secondary)

                if let timestamp {
                    Text(timestamp, style: .time)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                Spacer()
                actions
            }

            toolTraceContent
                .padding(.leading, contentLeadingPadding)

            content
                .padding(.leading, contentLeadingPadding)
        }
        .padding(.vertical, 8)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(accentColor.opacity(sideBarOpacity))
                .frame(width: 2)
                .padding(.vertical, 4)
        }
    }
}

public struct EdgeAgentWorkbench<StatusContent: View, ToolTraceContent: View, ResponseContent: View>: View {
    private let accentColor: Color
    private let sideBarOpacity: Double
    private let statusContent: StatusContent
    private let toolTraceContent: ToolTraceContent
    private let responseContent: ResponseContent

    public init(
        accentColor: Color = .indigo,
        sideBarOpacity: Double = 0.45,
        @ViewBuilder statusContent: () -> StatusContent,
        @ViewBuilder toolTraceContent: () -> ToolTraceContent,
        @ViewBuilder responseContent: () -> ResponseContent
    ) {
        self.accentColor = accentColor
        self.sideBarOpacity = sideBarOpacity
        self.statusContent = statusContent()
        self.toolTraceContent = toolTraceContent()
        self.responseContent = responseContent()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusContent
            toolTraceContent
            responseContent
        }
        .padding(.vertical, 8)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(accentColor.opacity(sideBarOpacity))
                .frame(width: 2)
                .padding(.vertical, 4)
        }
    }
}

private extension Int {
    func edgeModulo(_ divisor: Int) -> Int {
        let result = self % divisor
        return result >= 0 ? result : result + divisor
    }
}

#endif
