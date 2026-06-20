// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Stable intent tags for logs, JSON eval fixtures, and Python tooling.
public enum PersonalIntentTag: String, Sendable, Codable, Equatable {
    case exactFact = "exact_fact"
    case aggregateFact = "aggregate_fact"
    case userProfile = "user_profile"
    case appAction = "app_action"
    case baseChat = "base_chat"
    case mixed = "mixed"
}

/// Top-level route intent for a single user input.
public indirect enum PersonalIntent: Sendable, Codable, Equatable {
    case exactFact(plan: FactQueryPlan)
    case aggregateFact(plan: FactQueryPlan)
    case userProfile(detail: ProfileDetail)
    case appAction(plan: ActionPlan)
    case baseChat
    /// Composite intent. INVARIANT: candidates must not contain `.mixed`.
    /// Routers should flatten nested mixed intents at construction time.
    case mixed(candidates: [PersonalIntent])

    public var tag: PersonalIntentTag {
        switch self {
        case .exactFact:
            return .exactFact
        case .aggregateFact:
            return .aggregateFact
        case .userProfile:
            return .userProfile
        case .appAction:
            return .appAction
        case .baseChat:
            return .baseChat
        case .mixed:
            return .mixed
        }
    }

    private enum CodingKeys: String, CodingKey {
        case tag
        case plan
        case detail
        case candidates
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let tag = try container.decode(PersonalIntentTag.self, forKey: .tag)
        switch tag {
        case .exactFact:
            self = .exactFact(plan: try container.decode(FactQueryPlan.self, forKey: .plan))
        case .aggregateFact:
            self = .aggregateFact(plan: try container.decode(FactQueryPlan.self, forKey: .plan))
        case .userProfile:
            self = .userProfile(detail: try container.decode(ProfileDetail.self, forKey: .detail))
        case .appAction:
            self = .appAction(plan: try container.decode(ActionPlan.self, forKey: .plan))
        case .baseChat:
            self = .baseChat
        case .mixed:
            self = .mixed(candidates: try container.decode([PersonalIntent].self, forKey: .candidates))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(tag, forKey: .tag)
        switch self {
        case .exactFact(let plan):
            try container.encode(plan, forKey: .plan)
        case .aggregateFact(let plan):
            try container.encode(plan, forKey: .plan)
        case .userProfile(let detail):
            try container.encode(detail, forKey: .detail)
        case .appAction(let plan):
            try container.encode(plan, forKey: .plan)
        case .baseChat:
            break
        case .mixed(let candidates):
            try container.encode(candidates, forKey: .candidates)
        }
    }
}

/// Router output consumed by runtime/tool orchestration and audit logging.
public struct RouteDecision: Sendable, Codable, Equatable {
    public var intent: PersonalIntent
    public var confidence: Double
    public var reason: String
    public var toolPlan: ToolCallPlan?
    public var personaSignal: PersonaSignal?
    public var fallbackChain: [PersonalIntentTag]
    public var auditPayload: [String: AuditValue]

    public init(
        intent: PersonalIntent,
        confidence: Double,
        reason: String,
        toolPlan: ToolCallPlan? = nil,
        personaSignal: PersonaSignal? = nil,
        fallbackChain: [PersonalIntentTag] = [],
        auditPayload: [String: AuditValue] = [:]
    ) {
        self.intent = intent
        self.confidence = confidence
        self.reason = reason
        self.toolPlan = toolPlan
        self.personaSignal = personaSignal
        self.fallbackChain = fallbackChain
        self.auditPayload = auditPayload
    }
}

public protocol PersonalIntentRouter: Sendable {
    func route(_ input: UserInputContext) async throws -> RouteDecision
}

/// User input plus app/runtime context available before route execution.
public struct UserInputContext: Sendable, Codable, Equatable {
    public var text: String
    public var localeIdentifier: String?
    public var timeZoneIdentifier: String?
    public var referenceDate: Date?
    public var conversationID: String?
    public var appID: String?
    public var appContext: [String: AuditValue]
    public var personaSignals: [PersonaSignal]

    public init(
        text: String,
        localeIdentifier: String? = nil,
        timeZoneIdentifier: String? = nil,
        referenceDate: Date? = nil,
        conversationID: String? = nil,
        appID: String? = nil,
        appContext: [String: AuditValue] = [:],
        personaSignals: [PersonaSignal] = []
    ) {
        self.text = text
        self.localeIdentifier = localeIdentifier
        self.timeZoneIdentifier = timeZoneIdentifier
        self.referenceDate = referenceDate
        self.conversationID = conversationID
        self.appID = appID
        self.appContext = appContext
        self.personaSignals = personaSignals
    }
}

/// Structured plan for exact and aggregate fact retrieval.
public struct FactQueryPlan: Sendable, Codable, Equatable {
    public var namespace: String?
    public var schema: String?
    public var filters: [FactQueryFilter]
    public var aggregation: FactAggregation?
    public var groupBy: [String]
    public var sort: [FactSort]
    public var limit: Int?
    public var requestedFields: [String]

    public init(
        namespace: String? = nil,
        schema: String? = nil,
        filters: [FactQueryFilter] = [],
        aggregation: FactAggregation? = nil,
        groupBy: [String] = [],
        sort: [FactSort] = [],
        limit: Int? = nil,
        requestedFields: [String] = []
    ) {
        self.namespace = namespace
        self.schema = schema
        self.filters = filters
        self.aggregation = aggregation
        self.groupBy = groupBy
        self.sort = sort
        self.limit = limit
        self.requestedFields = requestedFields
    }
}

public struct FactQueryFilter: Sendable, Codable, Equatable {
    public var field: String
    public var op: FactFilterOperator
    public var value: AuditValue

    public init(field: String, op: FactFilterOperator, value: AuditValue) {
        self.field = field
        self.op = op
        self.value = value
    }
}

public enum FactFilterOperator: String, Sendable, Codable, Equatable {
    case equals
    case notEquals = "not_equals"
    case contains
    case inList = "in"
    case notInList = "not_in"
    case greaterThan = "gt"
    case greaterThanOrEqual = "gte"
    case lessThan = "lt"
    case lessThanOrEqual = "lte"
    case between
    case exists
}

public struct FactAggregation: Sendable, Codable, Equatable {
    public var function: FactAggregationFunction
    public var field: String?

    public init(function: FactAggregationFunction, field: String? = nil) {
        self.function = function
        self.field = field
    }
}

public enum FactAggregationFunction: String, Sendable, Codable, Equatable {
    case count
    case sum
    case average = "avg"
    case minimum = "min"
    case maximum = "max"
}

public struct FactSort: Sendable, Codable, Equatable {
    public var field: String
    public var direction: SortDirection

    public init(field: String, direction: SortDirection = .ascending) {
        self.field = field
        self.direction = direction
    }
}

public enum SortDirection: String, Sendable, Codable, Equatable {
    case ascending = "asc"
    case descending = "desc"
}

public struct ProfileDetail: Sendable, Codable, Equatable {
    public var kind: ProfileDetailKind
    public var dimensions: [String]
    public var timeScope: String?
    public var minimumConfidence: Double?

    public init(
        kind: ProfileDetailKind,
        dimensions: [String] = [],
        timeScope: String? = nil,
        minimumConfidence: Double? = nil
    ) {
        self.kind = kind
        self.dimensions = dimensions
        self.timeScope = timeScope
        self.minimumConfidence = minimumConfidence
    }
}

public enum ProfileDetailKind: String, Sendable, Codable, Equatable {
    case summary
    case preference
    case habit
    case trait
    case routine
}

public struct PersonaSignal: Sendable, Codable, Equatable {
    public var source: PersonaSignalSource
    public var label: String
    public var confidence: Double
    /// Steering injection magnitude. v1 contract: keep this at 0 unless
    /// decode-time AS is explicitly enabled behind eval gates.
    public var activationSteeringAlpha: Double
    public var metadata: [String: AuditValue]

    public init(
        source: PersonaSignalSource,
        label: String,
        confidence: Double,
        activationSteeringAlpha: Double = 0,
        metadata: [String: AuditValue] = [:]
    ) {
        self.source = source
        self.label = label
        self.confidence = confidence
        self.activationSteeringAlpha = activationSteeringAlpha
        self.metadata = metadata
    }
}

public enum PersonaSignalSource: String, Sendable, Codable, Equatable {
    case rpp
    case pca
    case residualPCA = "residual_pca"
    case activationSteering = "activation_steering"
    case app
}

public struct ActionPlan: Sendable, Codable, Equatable {
    public var name: String
    public var arguments: [String: AuditValue]
    public var requiresConfirmation: Bool

    public init(
        name: String,
        arguments: [String: AuditValue] = [:],
        requiresConfirmation: Bool = true
    ) {
        self.name = name
        self.arguments = arguments
        self.requiresConfirmation = requiresConfirmation
    }
}

public struct ToolCallPlan: Sendable, Codable, Equatable {
    public var toolName: String
    public var arguments: [String: AuditValue]
    public var reason: String?

    public init(
        toolName: String,
        arguments: [String: AuditValue] = [:],
        reason: String? = nil
    ) {
        self.toolName = toolName
        self.arguments = arguments
        self.reason = reason
    }
}

/// JSON-compatible value type for route audits and normalized tool arguments.
///
/// NOTE: Integer literals, such as `1`, decode to `.int`, not `.double`.
/// Consumers that require strict Double semantics should normalize at the
/// tool boundary.
///
public enum AuditValue: Sendable, Codable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([AuditValue])
    case object([String: AuditValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([AuditValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: AuditValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Expected string, int, double, bool, null, array, or object"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }
}
