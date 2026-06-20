// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import SwiftUI
import Foundation
import EdgeData

#if os(iOS)

public struct ClassificationCorrectionSheet: View {

    let fact: Fact
    let onComplete: () -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var selectedSchema: String = ""
    @State private var stringValues: [String: String] = [:]
    @State private var dateValues: [String: Date] = [:]
    @State private var isSubmitting: Bool = false
    @State private var errorMessage: String?

    public init(fact: Fact, onComplete: @escaping () -> Void) {
        self.fact = fact
        self.onComplete = onComplete
    }

    private var candidateSchemas: [String] {
        Edge.registeredSchemaNames()
    }

    private var currentSchema: SchemaDef? {
        Edge.schema(selectedSchema)
    }

    public var body: some View {
        NavigationStack {
            Form {
                if !fact.payload.isEmpty {
                    rawPayloadSection
                }
                if candidateSchemas.count >= 2 {
                    schemaSection
                }
                if let schema = currentSchema {
                    fieldsSection(schema: schema)
                } else {
                    Section {
                        Text("No schema registered. Cannot correct fact.")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }
                if let err = errorMessage {
                    Section {
                        Text(err)
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }
                metaSection
            }
            .navigationTitle("Correct Classification")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                        .disabled(isSubmitting)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if isSubmitting {
                        ProgressView()
                    } else {
                        Button("Submit") { submit() }
                            .disabled(currentSchema == nil)
                            .bold()
                    }
                }
            }
            .onAppear { initState() }
        }
    }

    private var rawPayloadSection: some View {
        Section(header: Text("Raw payload (AI saw this)")) {
            ForEach(fact.payload.keys.sorted(), id: \.self) { key in
                HStack(alignment: .top, spacing: 8) {
                    Text(key)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .frame(minWidth: 100, alignment: .leading)
                    Text(displayValue(fact.payload[key]))
                        .font(.caption.monospaced())
                        .multilineTextAlignment(.trailing)
                        .lineLimit(3)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }
        }
    }

    private var schemaSection: some View {
        Section(header: Text("Schema")) {
            Picker("Schema", selection: $selectedSchema) {
                ForEach(candidateSchemas, id: \.self) { name in
                    Text(shortSchemaLabel(name)).tag(name)
                }
            }
            .pickerStyle(.segmented)
        }
    }

    @ViewBuilder
    private func fieldsSection(schema: SchemaDef) -> some View {
        Section(header: Text("Normalized payload")) {
            ForEach(schema.fields, id: \.name) { field in
                fieldEditor(field: field)
            }
        }
    }

    private var metaSection: some View {
        Section(header: Text("Fact meta")) {
            metaRow(label: "id", value: shortId(fact.id))
            metaRow(label: "status", value: fact.status.rawValue)
            if let conf = fact.classificationConfidence {
                metaRow(label: "confidence", value: String(format: "%.2f", conf))
            }
            if let model = fact.classificationModelVer {
                metaRow(label: "model_ver", value: model)
            }
            metaRow(label: "current schema", value: fact.schema)
        }
    }

    private func metaRow(label: String, value: String) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption.monospaced()).foregroundStyle(.primary)
        }
    }

    @ViewBuilder
    private func fieldEditor(field: FieldDef) -> some View {
        switch field.type {
        case .numeric:
            HStack {
                fieldLabel(field: field)
                Spacer()
                TextField("0.00", text: bindingFor(field.name))
                    .keyboardType(.decimalPad)
                    .multilineTextAlignment(.trailing)
                    .frame(maxWidth: 160)
            }

        case .text:
            VStack(alignment: .leading, spacing: 6) {
                fieldLabel(field: field)
                TextField("…", text: bindingFor(field.name), axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
            }

        case .entity:
            HStack {
                fieldLabel(field: field)
                Spacer()
                TextField("…", text: bindingFor(field.name))
                    .multilineTextAlignment(.trailing)
            }

        case .geohash:
            HStack {
                fieldLabel(field: field)
                Spacer()
                TextField("geohash8", text: bindingFor(field.name))
                    .multilineTextAlignment(.trailing)
                    .autocapitalization(.none)
            }

        case .categorical(let values):
            Picker(selection: bindingFor(field.name)) {
                Text("(none)").tag("")
                ForEach(values, id: \.self) { v in
                    Text(v).tag(v)
                }
            } label: {
                fieldLabel(field: field)
            }
            .pickerStyle(.menu)

        case .timestamp:
            DatePicker(
                selection: dateBindingFor(field.name),
                displayedComponents: [.date, .hourAndMinute]
            ) {
                fieldLabel(field: field)
            }
        }
    }

    private func fieldLabel(field: FieldDef) -> some View {
        HStack(spacing: 4) {
            Text(field.name)
                .font(.subheadline)
            if field.required {
                Text("*").foregroundStyle(.red)
            }
        }
    }

    private func initState() {
        let registered = Set(candidateSchemas)
        if registered.contains(fact.schema) {
            selectedSchema = fact.schema
        } else {
            selectedSchema = candidateSchemas.first ?? ""
        }

        for (key, value) in fact.payload {
            stringValues[key] = stringRepresentation(of: value)
            if let date = parseTimestamp(value) {
                dateValues[key] = date
            }
        }
    }

    private func bindingFor(_ key: String) -> Binding<String> {
        Binding(
            get: { stringValues[key, default: ""] },
            set: { stringValues[key] = $0 }
        )
    }

    private func dateBindingFor(_ key: String) -> Binding<Date> {
        Binding(
            get: {
                if let d = dateValues[key] { return d }
                if let raw = stringValues[key], let d = parseTimestampString(raw) {
                    return d
                }
                return Date()
            },
            set: { dateValues[key] = $0 }
        )
    }

    private func submit() {
        guard let schema = currentSchema else { return }
        errorMessage = nil

        var newPayload: [String: Any] = [:]
        for field in schema.fields {
            do {
                if let value = try parseField(field) {
                    newPayload[field.name] = value
                } else if field.required {
                    errorMessage = "Field '\(field.name)' is required."
                    return
                }
            } catch let parseError as ValidationError {
                errorMessage = parseError.message
                return
            } catch {
                errorMessage = error.localizedDescription
                return
            }
        }

        isSubmitting = true
        let factId = fact.id
        let targetSchema = selectedSchema
        let payload = newPayload

        Task {
            do {
                try await Edge.correctClassification(
                    factId: factId,
                    newSchema: targetSchema,
                    newPayload: payload
                )
                await MainActor.run {
                    isSubmitting = false
                    onComplete()
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    isSubmitting = false
                    errorMessage = "Submit failed: \(error.localizedDescription)"
                    NSLog("[ClassificationCorrectionSheet] submit ❌ \(error.localizedDescription)")
                }
            }
        }
    }

    private struct ValidationError: Error {
        let message: String
    }

    private func parseField(_ field: FieldDef) throws -> Any? {
        switch field.type {
        case .numeric:
            let raw = stringValues[field.name, default: ""].trimmingCharacters(in: .whitespaces)
            if raw.isEmpty { return nil }
            if let d = Double(raw) { return d }
            throw ValidationError(message: "Field '\(field.name)' must be a number, got '\(raw)'.")

        case .text, .entity, .geohash:
            let raw = stringValues[field.name, default: ""].trimmingCharacters(in: .whitespacesAndNewlines)
            return raw.isEmpty ? nil : raw

        case .categorical(let values):
            let raw = stringValues[field.name, default: ""]
            if raw.isEmpty { return nil }
            if values.contains(raw) { return raw }
            throw ValidationError(message: "Field '\(field.name)' must be one of \(values), got '\(raw)'.")

        case .timestamp:
            let date = dateBindingFor(field.name).wrappedValue
            return Int64(date.timeIntervalSince1970 * 1000)
        }
    }

    private func stringRepresentation(of value: Any) -> String {
        switch value {
        case let s as String: return s
        case let n as Int: return String(n)
        case let n as Int64: return String(n)
        case let d as Double:
            if d == d.rounded() { return String(format: "%.0f", d) }
            return String(format: "%.2f", d)
        case let b as Bool: return b ? "true" : "false"
        case is NSNull: return ""
        default: return String(describing: value)
        }
    }

    private func displayValue(_ value: Any?) -> String {
        guard let v = value else { return "(nil)" }
        let s = stringRepresentation(of: v)
        return s.isEmpty ? "(empty)" : s
    }

    private func parseTimestamp(_ value: Any) -> Date? {
        switch value {
        case let n as Int64:
            return Date(timeIntervalSince1970: TimeInterval(n) / 1000)
        case let n as Int:
            return Date(timeIntervalSince1970: TimeInterval(n) / 1000)
        case let n as Double:
            return Date(timeIntervalSince1970: TimeInterval(n) / 1000)
        case let s as String:
            return parseTimestampString(s)
        default:
            return nil
        }
    }

    private func parseTimestampString(_ raw: String) -> Date? {
        if let ms = Int64(raw) {
            return Date(timeIntervalSince1970: TimeInterval(ms) / 1000)
        }
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = iso.date(from: raw) { return d }
        iso.formatOptions = [.withInternetDateTime]
        return iso.date(from: raw)
    }

    private func shortSchemaLabel(_ name: String) -> String {
        if let dot = name.lastIndex(of: ".") {
            return String(name[name.index(after: dot)...])
        }
        return name
    }

    private func shortId(_ id: String) -> String {
        guard id.count > 16 else { return id }
        let prefix = id.prefix(8)
        let suffix = id.suffix(6)
        return "\(prefix)…\(suffix)"
    }
}

#endif
