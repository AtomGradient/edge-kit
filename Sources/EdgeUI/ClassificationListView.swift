// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import SwiftUI
import Foundation
import EdgeData

#if os(iOS)

@MainActor
final class ClassificationListStore: ObservableObject {

    @Published var rawFacts: [Fact] = []
    @Published var lowConfidenceFacts: [Fact] = []
    @Published var failedFacts: [Fact] = []
    @Published var loadError: String?
    @Published var isLoading: Bool = false

    private let namespace: String
    private let lowConfidenceThreshold: Double = 0.7

    init(namespace: String) {
        self.namespace = namespace
    }

    func reload() {
        isLoading = true
        defer { isLoading = false }
        do {
            let raw = try Edge.queryFacts(
                namespace: namespace,
                status: .rawUnclassified,
                limit: 500
            )
            let classified = try Edge.queryFacts(
                namespace: namespace,
                status: .classifiedOnly,
                limit: 1000
            )
            let failed = try Edge.queryFacts(
                namespace: namespace,
                status: .classificationFailed,
                limit: 500
            )
            let lowConf = classified.filter {
                ($0.classificationConfidence ?? 1.0) < lowConfidenceThreshold
            }
            self.rawFacts = raw
            self.lowConfidenceFacts = lowConf
            self.failedFacts = failed
            self.loadError = nil
            NSLog("[ClassificationListStore] reload: raw=\(raw.count) lowConf=\(lowConf.count) failed=\(failed.count)")
        } catch {
            self.loadError = error.localizedDescription
            NSLog("[ClassificationListStore] reload ❌ \(error.localizedDescription)")
        }
    }

    var totalToReview: Int {
        rawFacts.count + lowConfidenceFacts.count + failedFacts.count
    }
}

enum ClassificationRowKind {
    case raw, lowConfidence, failed

    var icon: String {
        switch self {
        case .raw: return "questionmark.circle.fill"
        case .lowConfidence: return "exclamationmark.triangle.fill"
        case .failed: return "xmark.octagon.fill"
        }
    }

    var color: Color {
        switch self {
        case .raw: return .blue
        case .lowConfidence: return .orange
        case .failed: return .red
        }
    }

    var sectionTitle: String {
        switch self {
        case .raw: return "待分类"
        case .lowConfidence: return "低置信度 (< 0.7)"
        case .failed: return "分类失败"
        }
    }
}

public struct ClassificationListView: View {

    @StateObject private var store: ClassificationListStore
    @State private var selectedFact: Fact?

    public init(namespace: String) {
        _store = StateObject(wrappedValue: ClassificationListStore(namespace: namespace))
    }

    public var body: some View {
        Group {
            if store.totalToReview == 0 && store.loadError == nil {
                emptyState
            } else {
                listContent
            }
        }
        .navigationTitle("Classification Inbox")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    store.reload()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Refresh")
            }
        }
        .onAppear { store.reload() }
        .sheet(item: $selectedFact) { fact in
            ClassificationCorrectionSheet(
                fact: fact,
                onComplete: {
                    selectedFact = nil
                    store.reload()
                }
            )
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(.green)
            Text("All facts are classified")
                .font(.headline)
            Text("AI 已自动分类全部数据，且置信度都不低。\n校准入口将在新的 raw_unclassified / 低置信度 fact 出现时再次激活。")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var listContent: some View {
        List {
            if let err = store.loadError {
                Section {
                    Text("Load error: \(err)")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }

            section(kind: .raw, facts: store.rawFacts)
            section(kind: .lowConfidence, facts: store.lowConfidenceFacts)
            section(kind: .failed, facts: store.failedFacts)
        }
        .listStyle(.insetGrouped)
    }

    @ViewBuilder
    private func section(kind: ClassificationRowKind, facts: [Fact]) -> some View {
        if !facts.isEmpty {
            Section(header: sectionHeader(kind: kind, count: facts.count)) {
                ForEach(facts) { fact in
                    Button {
                        selectedFact = fact
                    } label: {
                        ClassificationFactRow(fact: fact, kind: kind)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func sectionHeader(kind: ClassificationRowKind, count: Int) -> some View {
        HStack(spacing: 8) {
            Image(systemName: kind.icon)
                .foregroundStyle(kind.color)
            Text(kind.sectionTitle)
            Text("\(count)")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 2)
                .background(kind.color.opacity(0.15))
                .foregroundStyle(kind.color)
                .clipShape(Capsule())
        }
    }
}

private struct ClassificationFactRow: View {
    let fact: Fact
    let kind: ClassificationRowKind

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: kind.icon)
                .foregroundStyle(kind.color)
                .font(.title3)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                Text(displayPreview)
                    .font(.subheadline)
                    .lineLimit(2)
                    .foregroundStyle(.primary)

                HStack(spacing: 8) {
                    Text(timestampText)
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    Text(fact.schema)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        #if os(iOS)
                        .background(Color(uiColor: UIColor.tertiarySystemBackground))
                        #else
                        .background(Color.secondary.opacity(0.15))
                        #endif
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.secondary)

                    if let conf = fact.classificationConfidence, kind != .raw {
                        Text("conf \(String(format: "%.2f", conf))")
                            .font(.caption2)
                            .foregroundStyle(kind.color)
                    }
                }
            }

            Spacer(minLength: 8)

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }

    private var displayPreview: String {
        for key in ["description", "ocrText", "merchant", "counterparty"] {
            if let s = fact.payload[key] as? String, !s.isEmpty, s.lowercased() != "null" {
                return String(s.prefix(80))
            }
        }
        let keys = fact.payload.keys.sorted()
        let kv = keys.prefix(3)
            .compactMap { key -> String? in
                guard let value = fact.payload[key] else { return nil }
                return "\(key)=\(String(describing: value).prefix(20))"
            }
            .joined(separator: ", ")
        return kv.isEmpty ? "(empty payload)" : kv
    }

    private var timestampText: String {
        let date = Date(timeIntervalSince1970: TimeInterval(fact.tsMs) / 1000)
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

#endif
