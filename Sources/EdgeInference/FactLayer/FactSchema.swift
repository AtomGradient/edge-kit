// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

/// Supported fact payload field types.
public enum FactFieldType: String, Sendable, Equatable {
    case str
    case int
    case float
    case datetime
    case bool
}

/// Definition for one payload field.
public struct FactFieldDef: Sendable, Equatable {
    public let type: FactFieldType
    public let required: Bool
    /// Whether the field is promoted to an indexed SQL column.
    public let indexed: Bool
    /// Whether the field may participate in semantic indexing.
    public let semantic: Bool
    public let description: String

    public init(
        type: FactFieldType,
        required: Bool = false,
        indexed: Bool = false,
        semantic: Bool = false,
        description: String = ""
    ) {
        self.type = type
        self.required = required
        self.indexed = indexed
        self.semantic = semantic
        self.description = description
    }

    /// Validates a value against this field definition.
    public func validate(_ value: Any?) throws {
        guard let value = value else {
            if required {
                throw FactSchemaError.requiredFieldMissing
            }
            return
        }

        let ok: Bool
        switch type {
        case .str:
            ok = value is String
        case .int:
            ok = (value is Int) || (value is Int64) || (value is Int32)
        case .float:
            ok = (value is Double) || (value is Float)
                || (value is Int) || (value is Int64) || (value is Int32)
        case .datetime:
            ok = (value is Int64) || (value is Int) || (value is Int32)
        case .bool:
            ok = value is Bool
        }

        if !ok {
            throw FactSchemaError.typeMismatch(
                expected: type.rawValue,
                got: String(describing: Swift.type(of: value))
            )
        }
    }
}

/// Domain schema for fact payload validation and indexing.
public struct FactSchema: Sendable, Equatable {
    public let name: String
    public let fields: [String: FactFieldDef]
    /// Primary timestamp field used by time-range queries.
    public let primaryTimeField: String?
    public let description: String

    public init(
        name: String,
        fields: [String: FactFieldDef],
        primaryTimeField: String? = nil,
        description: String = ""
    ) throws {
        if let pt = primaryTimeField {
            guard let def = fields[pt] else {
                throw FactSchemaError.primaryTimeFieldMissing(pt)
            }
            guard def.type == .datetime else {
                throw FactSchemaError.primaryTimeFieldWrongType(
                    field: pt, got: def.type.rawValue
                )
            }
        }
        self.name = name
        self.fields = fields
        self.primaryTimeField = primaryTimeField
        self.description = description
    }

    /// Validates a payload against this schema.
    public func validatePayload(_ payload: [String: Any]) throws {
        let unknown = Set(payload.keys).subtracting(fields.keys)
        if !unknown.isEmpty {
            throw FactSchemaError.unknownFields(
                schema: name, fields: Array(unknown).sorted()
            )
        }
        for (fieldName, fieldDef) in fields {
            do {
                try fieldDef.validate(payload[fieldName])
            } catch FactSchemaError.requiredFieldMissing {
                throw FactSchemaError.requiredFieldNamed(fieldName)
            } catch FactSchemaError.typeMismatch(let expected, let got) {
                throw FactSchemaError.fieldTypeMismatch(
                    field: fieldName, expected: expected, got: got
                )
            }
        }
    }

    /// Returns indexed field names in stable order.
    public func indexedFields() -> [String] {
        fields.filter { $0.value.indexed }.map { $0.key }.sorted()
    }
}

public enum FactSchemaError: Error, Equatable {
    case requiredFieldMissing
    case requiredFieldNamed(String)
    case typeMismatch(expected: String, got: String)
    case fieldTypeMismatch(field: String, expected: String, got: String)
    case unknownFields(schema: String, fields: [String])
    case primaryTimeFieldMissing(String)
    case primaryTimeFieldWrongType(field: String, got: String)
    case schemaNotRegistered(String)
}

public enum BuiltInFactSchemas {
    public static let financeExpense: FactSchema = {
        try! FactSchema(
            name: "finance.expense",
            fields: [
                "amount": FactFieldDef(
                    type: .float, required: true, indexed: true,
                    description: "消费金额（本币）"
                ),
                "merchant": FactFieldDef(
                    type: .str, indexed: true,
                    description: "商家名称（如'必胜客'、'邻几便利'）"
                ),
                "category": FactFieldDef(
                    type: .str, indexed: true,
                    description: "消费类别（如'餐饮'、'Transport'、'午餐'）"
                ),
                "time": FactFieldDef(
                    type: .datetime, required: true, indexed: true,
                    description: "消费时间（Unix ms）"
                ),
                "location": FactFieldDef(
                    type: .str, indexed: true,
                    description: "消费地点"
                ),
                "description": FactFieldDef(
                    type: .str, semantic: true,
                    description: "消费描述（自由文本，参与语义检索）"
                ),
            ],
            primaryTimeField: "time",
            description: "个人消费记录（记账条目）"
        )
    }()
}

public final class FactSchemaRegistry: @unchecked Sendable {
    public static let shared: FactSchemaRegistry = {
        let r = FactSchemaRegistry()
        r.register(BuiltInFactSchemas.financeExpense)
        return r
    }()

    private let lock = NSLock()
    private var schemas: [String: FactSchema] = [:]

    /// Registers or replaces a schema.
    public func register(_ schema: FactSchema) {
        lock.lock(); defer { lock.unlock() }
        schemas[schema.name] = schema
    }

    /// Returns a registered schema.
    public func get(_ name: String) throws -> FactSchema {
        lock.lock(); defer { lock.unlock() }
        guard let s = schemas[name] else {
            throw FactSchemaError.schemaNotRegistered(name)
        }
        return s
    }

    /// Returns registered schema names.
    public func registeredNames() -> [String] {
        lock.lock(); defer { lock.unlock() }
        return schemas.keys.sorted()
    }
}
