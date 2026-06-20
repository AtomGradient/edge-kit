// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Produces privacy-preserving summaries instead of raw records.
public protocol MeshSummarizable {
    /// Generates a statistical summary for the requested day range.
    func summarize(days: Int) -> MeshSummary
}

/// Statistical summary data that contains no raw records.
public struct MeshSummary: Sendable, Codable {

    /// Data domain, such as health, nutrition, mood, or sleep.
    public let domain: String
    /// Summary period in days.
    public let periodDays: Int
    public let generatedAt: Date
    public let metrics: [Metric]
    public let trends: [Trend]
    public let alerts: [Alert]

    public init(
        domain: String,
        periodDays: Int,
        generatedAt: Date = Date(),
        metrics: [Metric],
        trends: [Trend] = [],
        alerts: [Alert] = []
    ) {
        self.domain = domain
        self.periodDays = periodDays
        self.generatedAt = generatedAt
        self.metrics = metrics
        self.trends = trends
        self.alerts = alerts
    }

    /// Estimated compression ratio between raw data points and summary size.
    public var estimatedCompressionRatio: Double {
        let rawEstimate = Double(periodDays * 24 * 4)
        let summarySize = Double(metrics.count + trends.count + alerts.count)
        return rawEstimate / max(summarySize, 1)
    }

    /// Statistical metric over the summary period.
    public struct Metric: Sendable, Codable {
        public let name: String
        public let mean: Double
        public let min: Double
        public let max: Double
        public let stdDev: Double
        public let sampleCount: Int

        public init(name: String, mean: Double, min: Double, max: Double, stdDev: Double, sampleCount: Int) {
            self.name = name
            self.mean = mean
            self.min = min
            self.max = max
            self.stdDev = stdDev
            self.sampleCount = sampleCount
        }
    }

    /// Trend signal for a metric.
    public struct Trend: Sendable, Codable {
        public let metric: String
        public let direction: Direction
        public let magnitude: Double
        public let confidence: Double

        public init(metric: String, direction: Direction, magnitude: Double, confidence: Double) {
            self.metric = metric
            self.direction = direction
            self.magnitude = magnitude
            self.confidence = confidence
        }

        public enum Direction: String, Sendable, Codable {
            case increasing, decreasing, stable
        }
    }

    /// Alert signal derived from summary-level data.
    public struct Alert: Sendable, Codable {
        public let level: Level
        public let message: String
        public let metric: String
        public let value: Double
        public let threshold: Double

        public init(level: Level, message: String, metric: String, value: Double, threshold: Double) {
            self.level = level
            self.message = message
            self.metric = metric
            self.value = value
            self.threshold = threshold
        }

        /// Alert severity level.
        public enum Level: String, Sendable, Codable, Comparable {
            case info  = "L1"
            case warning = "L2"
            case critical = "L3"

            public static func < (lhs: Level, rhs: Level) -> Bool {
                let order: [Level] = [.info, .warning, .critical]
                return (order.firstIndex(of: lhs) ?? 0) < (order.firstIndex(of: rhs) ?? 0)
            }
        }
    }
}

/// Aggregates domain summaries into a panoramic view.
public struct MeshSummaryAggregator: Sendable {

    /// Aggregates multiple domain summaries into a panoramic view.
    public static func aggregate(_ summaries: [MeshSummary]) -> PanoramicView {
        let domains = summaries.map { $0.domain }
        let allAlerts = summaries.flatMap { $0.alerts }.sorted { $0.level > $1.level }
        let crossDomainTrends = detectCrossDomainPatterns(summaries)

        return PanoramicView(
            domains: domains,
            totalMetrics: summaries.reduce(0) { $0 + $1.metrics.count },
            highestAlert: allAlerts.first?.level,
            alerts: allAlerts,
            crossDomainInsights: crossDomainTrends
        )
    }

    private static func detectCrossDomainPatterns(_ summaries: [MeshSummary]) -> [String] {
        var insights: [String] = []

        let worseningMetrics = summaries.flatMap { summary in
            summary.trends
                .filter { $0.direction == .decreasing && $0.confidence > 0.7 }
                .map { (domain: summary.domain, metric: $0.metric) }
        }

        if worseningMetrics.count >= 2 {
            let domains = Set(worseningMetrics.map { $0.domain })
            if domains.count >= 2 {
                insights.append("Multiple domains showing declining trends: \(domains.joined(separator: ", "))")
            }
        }

        return insights
    }
}

/// Cross-domain panoramic view.
public struct PanoramicView: Sendable {
    public let domains: [String]
    public let totalMetrics: Int
    public let highestAlert: MeshSummary.Alert.Level?
    public let alerts: [MeshSummary.Alert]
    public let crossDomainInsights: [String]
}
