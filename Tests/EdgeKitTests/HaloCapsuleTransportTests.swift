// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import XCTest
@testable import EdgeMesh

final class HaloCapsuleTransportTests: XCTestCase {
    func test_haloCapsulePackageTransfer_roundTripsPackageFramesToDisk() throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: source)
        let frames = try HaloCapsulePackageTransfer.packageFrames(
            message: message,
            packageDirectory: source,
            chunkSize: 7
        )
        let destination = try Self.makeTemporaryDirectory()
        let receiver = HaloCapsulePackageReceiver(configuration: .init(
            destinationDirectory: destination,
            expectedBaseModelID: "Qwen3.5-9B-4bit",
            expectedModelFamily: "qwen3_5",
            expectedHiddenSize: 4_096,
            expectedLayerCount: 32,
            expectedToolSchemaSHA256: Self.hash("tools"),
            currentRuntimeVersion: "1.0.0"
        ))

        var completed: HaloCapsuleMeshMessage?
        var sawOfferAck = false
        var chunkWrites = 0
        for frame in frames {
            let event = try receiver.receive(frame)
            switch event {
            case .offerAccepted(let ack):
                sawOfferAck = true
                XCTAssertTrue(ack.accepted)
                XCTAssertEqual(ack.transferID, message.transferID)
                XCTAssertEqual(ack.canonicalSHA256, try message.canonicalSHA256())
            case .chunkWritten:
                chunkWrites += 1
            case .completed(let received, let ack):
                completed = received
                XCTAssertTrue(ack.accepted)
                XCTAssertEqual(ack.canonicalSHA256, try message.canonicalSHA256())
            case .chunkHeaderAccepted, .offerRejected:
                break
            }
        }

        XCTAssertTrue(sawOfferAck)
        XCTAssertGreaterThan(chunkWrites, 1)
        XCTAssertEqual(completed, message)
        XCTAssertEqual(
            try Data(contentsOf: source.appendingPathComponent("manifest.json")),
            try Data(contentsOf: destination.appendingPathComponent("manifest.json"))
        )
        XCTAssertEqual(
            try Data(contentsOf: source.appendingPathComponent("halo.safetensors")),
            try Data(contentsOf: destination.appendingPathComponent("halo.safetensors"))
        )
    }

    func test_haloCapsuleInboundTransferSession_emitsAcksAndCompletedPackage() throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: source)
        let destination = try Self.makeTemporaryDirectory()
        let session = HaloCapsuleInboundTransferSession { offer in
            XCTAssertEqual(offer.transferID, message.transferID)
            return HaloCapsulePackageReceiver.Configuration(
                destinationDirectory: destination,
                expectedBaseModelID: "Qwen3.5-9B-4bit",
                expectedModelFamily: "qwen3_5",
                expectedHiddenSize: 4_096,
                expectedLayerCount: 32,
                expectedToolSchemaSHA256: Self.hash("tools"),
                currentRuntimeVersion: "1.0.0"
            )
        }
        let frames = try HaloCapsulePackageTransfer.packageFrames(
            message: message,
            packageDirectory: source,
            chunkSize: 7
        )

        var outgoingFrames: [Data] = []
        var completed: HaloCapsuleInboundTransferSession.CompletedPackage?
        for frame in frames {
            let result = try session.receive(frame)
            XCTAssertTrue(result.handled)
            outgoingFrames.append(contentsOf: result.outgoingFrames)
            if let completedPackage = result.completedPackage {
                completed = completedPackage
            }
        }

        XCTAssertFalse(session.hasActiveTransfer)
        XCTAssertEqual(completed?.message, message)
        XCTAssertEqual(completed?.packageDirectory, destination)
        XCTAssertEqual(outgoingFrames.count, 2)
        guard case .offerAck(let offerAck) = try HaloCapsulePackageTransfer.decodeFrame(outgoingFrames[0]) else {
            return XCTFail("expected offer ack")
        }
        XCTAssertTrue(offerAck.accepted)
        XCTAssertEqual(offerAck.transferID, message.transferID)
        guard case .completeAck(let completeAck) = try HaloCapsulePackageTransfer.decodeFrame(outgoingFrames[1]) else {
            return XCTFail("expected complete ack")
        }
        XCTAssertTrue(completeAck.accepted)
        XCTAssertEqual(completeAck.transferID, message.transferID)
        XCTAssertEqual(
            try Data(contentsOf: source.appendingPathComponent("manifest.json")),
            try Data(contentsOf: destination.appendingPathComponent("manifest.json"))
        )
    }

    func test_haloCapsuleInboundTransferSession_ignoresUnrelatedFrames() throws {
        let session = HaloCapsuleInboundTransferSession { _ in
            XCTFail("configuration provider should not be called for unrelated frames")
            return HaloCapsulePackageReceiver.Configuration(
                destinationDirectory: try Self.makeTemporaryDirectory()
            )
        }
        let unrelated = try JSONSerialization.data(
            withJSONObject: ["op": "training_available", "payload": [:]]
        )

        XCTAssertEqual(try session.receive(unrelated), .ignored)
        XCTAssertEqual(try session.receive(Data("not json".utf8)), .ignored)
        XCTAssertFalse(session.hasActiveTransfer)
    }

    func test_haloCapsuleApplyStatusMessageUsesStableWireShape() throws {
        let payload = HaloCapsuleApplyStatusPayload(
            transferID: "transfer-1",
            capsuleID: "capsule-1",
            status: .applied,
            artifactSHA256: Self.hash("artifact"),
            canonicalSHA256: Self.hash("canonical"),
            runtimeVersion: "1.0.0-rc52",
            prefixTokenCount: 2423,
            appliedAtUnixSeconds: 1_779_612_014.8
        )
        let message = HaloCapsuleApplyStatusMessage(payload: payload)

        let data = try JSONEncoder().encode(message)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["op"] as? String, "halo_capsule_apply_status")

        let body = try XCTUnwrap(object["payload"] as? [String: Any])
        XCTAssertEqual(body["schema_version"] as? String, "edgestudio.halo_capsule_apply_status.v1")
        XCTAssertEqual(body["transfer_id"] as? String, "transfer-1")
        XCTAssertEqual(body["capsule_id"] as? String, "capsule-1")
        XCTAssertEqual(body["status"] as? String, "applied")
        XCTAssertEqual(body["artifact_sha256"] as? String, Self.hash("artifact"))
        XCTAssertEqual(body["canonical_sha256"] as? String, Self.hash("canonical"))
        XCTAssertEqual(body["runtime_version"] as? String, "1.0.0-rc52")
        XCTAssertEqual(body["prefix_token_count"] as? Int, 2423)
        XCTAssertEqual(body["applied_at_unix_seconds"] as? Double, 1_779_612_014.8)

        let decoded = try JSONDecoder().decode(HaloCapsuleApplyStatusMessage.self, from: data)
        XCTAssertEqual(decoded, message)
    }

    func test_haloCapsulePackageTransfer_decodesApplyStatusFrame() throws {
        let status = HaloCapsuleApplyStatusPayload(
            transferID: "transfer-1",
            capsuleID: "capsule-1",
            status: .failed,
            runtimeVersion: "1.0.0-rc52",
            errorCode: "restore_failed",
            errorMessage: "runtime rejected cache"
        )

        let frame = try HaloCapsulePackageTransfer.makeApplyStatusFrame(status)

        switch try HaloCapsulePackageTransfer.decodeFrame(frame) {
        case .applyStatus(let decoded):
            XCTAssertEqual(decoded, status)
        default:
            XCTFail("expected applyStatus frame")
        }
    }

    func test_haloCapsuleAutoRestoreCoordinator_sendsReceivedThenApplied() async throws {
        let packageDirectory = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: packageDirectory)
        let completed = try Self.completedPackage(
            message: message,
            packageDirectory: packageDirectory
        )
        let recorder = AutoRestoreRecorder(
            restoreResult: .init(prefixTokenCount: 2_423)
        )
        let coordinator = HaloCapsuleAutoRestoreCoordinator(
            configuration: .init(
                runtimeVersion: "1.0.0-test",
                now: { Date(timeIntervalSince1970: 1_779_612_014.8) }
            ),
            restorePackage: { completed in
                try await recorder.restore(completed)
            },
            sendApplyStatus: { payload in
                try await recorder.send(payload)
            }
        )

        let result = try await coordinator.apply(completed)

        let restored = await recorder.restoredDirectories()
        XCTAssertEqual(restored, [packageDirectory])
        let sent = await recorder.sentPayloads()
        XCTAssertEqual(sent.count, 2)
        XCTAssertEqual(sent[0].status, .received)
        XCTAssertEqual(sent[0].transferID, message.transferID)
        XCTAssertEqual(sent[0].capsuleID, message.capsule.capsuleID)
        XCTAssertEqual(sent[0].artifactSHA256, message.capsule.artifact.sha256)
        XCTAssertEqual(sent[0].canonicalSHA256, try message.canonicalSHA256())
        XCTAssertEqual(sent[0].runtimeVersion, "1.0.0-test")
        XCTAssertEqual(sent[0].appliedAtUnixSeconds, 1_779_612_014.8)
        XCTAssertNil(sent[0].prefixTokenCount)
        XCTAssertNil(sent[0].errorCode)
        XCTAssertNil(sent[0].errorMessage)

        XCTAssertEqual(sent[1].status, .applied)
        XCTAssertEqual(sent[1].prefixTokenCount, 2_423)
        XCTAssertEqual(sent[1].canonicalSHA256, try message.canonicalSHA256())
        XCTAssertNil(sent[1].errorCode)
        XCTAssertNil(sent[1].errorMessage)

        XCTAssertEqual(result.receivedPayload, sent[0])
        XCTAssertEqual(result.terminalPayload, sent[1])
        XCTAssertEqual(result.outcome, .applied(.init(prefixTokenCount: 2_423)))
    }

    func test_haloCapsuleAutoRestoreCoordinator_reportsRestoreFailure() async throws {
        let packageDirectory = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: packageDirectory)
        let completed = try Self.completedPackage(
            message: message,
            packageDirectory: packageDirectory
        )
        let recorder = AutoRestoreRecorder(
            restoreError: AutoRestoreTestError(message: "restore failed with a long diagnostic")
        )
        let coordinator = HaloCapsuleAutoRestoreCoordinator(
            configuration: .init(
                runtimeVersion: "1.0.0-test",
                now: { Date(timeIntervalSince1970: 42) },
                maxErrorMessageLength: 14
            ),
            restorePackage: { completed in
                try await recorder.restore(completed)
            },
            sendApplyStatus: { payload in
                try await recorder.send(payload)
            }
        )

        let result = try await coordinator.apply(completed)

        let restored = await recorder.restoredDirectories()
        XCTAssertEqual(restored, [packageDirectory])
        let sent = await recorder.sentPayloads()
        XCTAssertEqual(sent.map(\.status), [.received, .failed])
        XCTAssertEqual(sent[1].errorCode, "restore_failed")
        XCTAssertEqual(sent[1].errorMessage, "restore failed")
        XCTAssertEqual(sent[1].canonicalSHA256, try message.canonicalSHA256())
        XCTAssertNil(sent[1].prefixTokenCount)
        XCTAssertEqual(
            result.outcome,
            .failed(errorCode: "restore_failed", errorMessage: "restore failed")
        )
        XCTAssertEqual(result.terminalPayload, sent[1])
    }

    func test_haloCapsuleAutoRestoreCoordinator_throwsApplyStatusSendFailure() async throws {
        let packageDirectory = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: packageDirectory)
        let completed = try Self.completedPackage(
            message: message,
            packageDirectory: packageDirectory
        )
        let recorder = AutoRestoreRecorder(
            restoreResult: .init(prefixTokenCount: 128),
            sendErrorStatus: .applied
        )
        let coordinator = HaloCapsuleAutoRestoreCoordinator(
            configuration: .init(
                runtimeVersion: "1.0.0-test",
                now: { Date(timeIntervalSince1970: 42) }
            ),
            restorePackage: { completed in
                try await recorder.restore(completed)
            },
            sendApplyStatus: { payload in
                try await recorder.send(payload)
            }
        )

        do {
            _ = try await coordinator.apply(completed)
            XCTFail("expected apply-status send failure")
        } catch let error as AutoRestoreTestError {
            XCTAssertEqual(error.message, "apply-status send failed")
        }

        let restored = await recorder.restoredDirectories()
        XCTAssertEqual(restored, [packageDirectory])
        let sent = await recorder.sentPayloads()
        XCTAssertEqual(sent.map(\.status), [.received])
    }

    func test_haloCapsulePackageReceiverRejectsApplyStatusAsInboundPackageFrame() throws {
        let status = HaloCapsuleApplyStatusPayload(
            transferID: "transfer-1",
            capsuleID: "capsule-1",
            status: .applied
        )
        let receiver = HaloCapsulePackageReceiver(configuration: .init(
            destinationDirectory: try Self.makeTemporaryDirectory()
        ))

        XCTAssertThrowsError(try receiver.receive(
            HaloCapsulePackageTransfer.makeApplyStatusFrame(status)
        )) { error in
            XCTAssertEqual(error as? HaloCapsuleTransportError, .unexpectedTransferFrame)
        }
    }

    func test_haloCapsulePackageTransfer_rejectsTamperedBinaryChunk() throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: source)
        var frames = try HaloCapsulePackageTransfer.packageFrames(
            message: message,
            packageDirectory: source,
            chunkSize: 64
        )
        guard let binaryIndex = frames.firstIndex(where: { frame in
            if case .binary = try? HaloCapsulePackageTransfer.decodeFrame(frame) {
                return true
            }
            return false
        }) else {
            return XCTFail("expected binary chunk frame")
        }
        frames[binaryIndex][0] ^= 0xff

        let receiver = HaloCapsulePackageReceiver(configuration: .init(
            destinationDirectory: try Self.makeTemporaryDirectory()
        ))

        for (idx, frame) in frames.enumerated() {
            if idx == binaryIndex {
                XCTAssertThrowsError(try receiver.receive(frame)) { error in
                    guard case HaloCapsuleTransportError.chunkHashMismatch = error else {
                        return XCTFail("expected chunkHashMismatch, got \(error)")
                    }
                }
                return
            }
            _ = try receiver.receive(frame)
        }
        XCTFail("expected tampered chunk to throw")
    }

    func test_haloCapsulePackageTransfer_rejectsUnsafeArtifactFileName() throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(
            directory: source,
            files: [
                HaloCapsuleArtifactFile(name: "../halo.safetensors", byteCount: 1, sha256: Self.hash("bad"))
            ],
            artifactBytes: 1,
            artifactSHA256: Self.hash("artifact")
        )

        XCTAssertThrowsError(try HaloCapsulePackageTransfer.packageFrames(
            message: message,
            packageDirectory: source
        )) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .unsafeArtifactFileName("../halo.safetensors")
            )
        }
    }

    func test_haloCapsuleMeshMessage_roundTripsAndValidates() throws {
        let message = try Self.message()
        let data = try JSONEncoder().encode(message)

        let decoded = try JSONDecoder().decode(HaloCapsuleMeshMessage.self, from: data)

        XCTAssertEqual(decoded, message)
        XCTAssertNoThrow(try decoded.validate(
            expectedBaseModelID: "Qwen3.5-9B-4bit",
            expectedToolSchemaSHA256: Self.hash("tools"),
            currentRuntimeVersion: "1.0.0"
        ))

        let json = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(json.contains("schema_version"))
        XCTAssertTrue(json.contains("transfer_id"))
        XCTAssertTrue(json.contains("requirements_sha256"))
        XCTAssertTrue(json.contains("cache_backend_version"))
        XCTAssertTrue(json.contains("tool_schema_sha256"))
        XCTAssertTrue(json.contains("model_family"))
        XCTAssertTrue(json.contains("hidden_size"))
        XCTAssertTrue(json.contains("layer_count"))
    }

    func test_haloCapsuleMeshMessage_canonicalHashIgnoresFileAndTensorOrder() throws {
        let messageA = try Self.message(
            files: [
                HaloCapsuleArtifactFile(name: "manifest.json", byteCount: 10, sha256: Self.hash("manifest")),
                HaloCapsuleArtifactFile(name: "halo.safetensors", byteCount: 90, sha256: Self.hash("halo")),
            ],
            tensors: [
                HaloCacheTensorDescriptor(name: "z", shape: [1], dtype: "float16", byteCount: 2, sha256: Self.hash("z")),
                HaloCacheTensorDescriptor(name: "a", shape: [1], dtype: "float16", byteCount: 2, sha256: Self.hash("a")),
            ]
        )
        let messageB = try Self.message(
            files: [
                HaloCapsuleArtifactFile(name: "halo.safetensors", byteCount: 90, sha256: Self.hash("halo")),
                HaloCapsuleArtifactFile(name: "manifest.json", byteCount: 10, sha256: Self.hash("manifest")),
            ],
            tensors: [
                HaloCacheTensorDescriptor(name: "a", shape: [1], dtype: "float16", byteCount: 2, sha256: Self.hash("a")),
                HaloCacheTensorDescriptor(name: "z", shape: [1], dtype: "float16", byteCount: 2, sha256: Self.hash("z")),
            ]
        )

        XCTAssertEqual(try messageA.canonicalSHA256(), try messageB.canonicalSHA256())
        XCTAssertEqual(
            String(data: try messageA.canonicalData(), encoding: .utf8),
            String(data: try messageB.canonicalData(), encoding: .utf8)
        )
    }

    func test_haloCapsuleMeshMessage_preservesDownloadURLsButIgnoresThemInCanonicalHash() throws {
        let base = try Self.message()
        let withDownloads = try Self.message(files: [
            HaloCapsuleArtifactFile(
                name: "manifest.json",
                byteCount: 10,
                sha256: Self.hash("manifest"),
                downloadURL: URL(string: "http://127.0.0.1:18842/api/mesh/halo_capsules/download/transfer-a/manifest.json")
            ),
            HaloCapsuleArtifactFile(
                name: "halo.safetensors",
                byteCount: 90,
                sha256: Self.hash("halo"),
                downloadURL: URL(string: "http://127.0.0.1:18842/api/mesh/halo_capsules/download/transfer-a/halo.safetensors")
            ),
        ])

        let encoded = try JSONEncoder().encode(withDownloads)
        let decoded = try JSONDecoder().decode(HaloCapsuleMeshMessage.self, from: encoded)

        XCTAssertEqual(decoded.capsule.artifact.files[0].downloadURL?.host, "127.0.0.1")
        XCTAssertEqual(try decoded.canonicalSHA256(), try base.canonicalSHA256())
        XCTAssertFalse(
            String(data: try decoded.canonicalData(), encoding: .utf8)?.contains("download_url") ?? true
        )
    }

    func test_haloCapsuleDownloadTransport_downloadsAndValidatesPackage() async throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackageWithDownloadURLs(directory: source)
        let destination = try Self.makeTemporaryDirectory()
        let transport = HaloCapsuleDownloadTransport { url in
            let tempDirectory = try Self.makeTemporaryDirectory()
            let tempURL = tempDirectory.appendingPathComponent(url.lastPathComponent)
            try FileManager.default.copyItem(at: url, to: tempURL)
            return HaloCapsuleDownloadedFile(temporaryURL: tempURL, httpStatusCode: 200)
        }

        XCTAssertTrue(HaloCapsuleDownloadTransport.usesDownloadTransport(message))
        let completed = try await transport.downloadPackage(message: message, to: destination)

        XCTAssertEqual(completed.message, message)
        XCTAssertEqual(completed.packageDirectory, destination)
        XCTAssertTrue(completed.completeAck.accepted)
        XCTAssertEqual(completed.completeAck.canonicalSHA256, try message.canonicalSHA256())
        XCTAssertEqual(
            try Data(contentsOf: source.appendingPathComponent("manifest.json")),
            try Data(contentsOf: destination.appendingPathComponent("manifest.json"))
        )
        XCTAssertEqual(
            try Data(contentsOf: source.appendingPathComponent("halo.safetensors")),
            try Data(contentsOf: destination.appendingPathComponent("halo.safetensors"))
        )
    }

    func test_haloCapsuleDownloadTransport_rejectsMissingDownloadURL() async throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackage(directory: source)
        let transport = HaloCapsuleDownloadTransport { _ in
            XCTFail("downloader should not be called without download_url")
            return HaloCapsuleDownloadedFile(temporaryURL: try Self.makeTemporaryDirectory())
        }

        do {
            _ = try await transport.downloadPackage(
                message: message,
                to: Self.makeTemporaryDirectory()
            )
            XCTFail("expected missingDownloadURL")
        } catch let error as HaloCapsuleTransportError {
            XCTAssertEqual(error, .missingDownloadURL(name: "manifest.json"))
        }
    }

    func test_haloCapsuleDownloadTransport_rejectsHTTPFailure() async throws {
        let source = try Self.makePackageDirectory()
        let message = try Self.messageForPackageWithDownloadURLs(directory: source)
        let transport = HaloCapsuleDownloadTransport { url in
            let tempDirectory = try Self.makeTemporaryDirectory()
            let tempURL = tempDirectory.appendingPathComponent(url.lastPathComponent)
            try Data().write(to: tempURL)
            return HaloCapsuleDownloadedFile(temporaryURL: tempURL, httpStatusCode: 503)
        }

        do {
            _ = try await transport.downloadPackage(
                message: message,
                to: Self.makeTemporaryDirectory()
            )
            XCTFail("expected downloadHTTPStatus")
        } catch let error as HaloCapsuleTransportError {
            XCTAssertEqual(error, .downloadHTTPStatus(name: "manifest.json", statusCode: 503))
        }
    }

    func test_haloCapsuleDescriptor_rejectsRequirementsHashMismatch() throws {
        var message = try Self.message()
        message.capsule.requirementsSHA256 = Self.hash("wrong")

        XCTAssertThrowsError(try message.validate()) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsHashMismatch(
                    expected: Self.hash("wrong"),
                    actual: try! message.capsule.requirements.canonicalSHA256()
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsToolSchemaMismatch() throws {
        let message = try Self.message()

        XCTAssertThrowsError(try message.validate(
            expectedToolSchemaSHA256: Self.hash("other-tools")
        )) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsMismatch(
                    key: "tool_schema_sha256",
                    expected: Self.hash("other-tools"),
                    actual: Self.hash("tools")
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsModelFamilyMismatch() throws {
        let message = try Self.message()

        XCTAssertThrowsError(try message.validate(
            expectedModelFamily: "llama"
        )) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsMismatch(
                    key: "model_family",
                    expected: "llama",
                    actual: "qwen3_5"
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsHiddenSizeMismatch() throws {
        let message = try Self.message()

        XCTAssertThrowsError(try message.validate(
            expectedHiddenSize: 2_560
        )) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsMismatch(
                    key: "hidden_size",
                    expected: "2560",
                    actual: "4096"
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsLayerCountMismatch() throws {
        let message = try Self.message()

        XCTAssertThrowsError(try message.validate(
            expectedLayerCount: 40
        )) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsMismatch(
                    key: "layer_count",
                    expected: "40",
                    actual: "32"
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsMissingModelShapeFields() throws {
        var message = try Self.message()
        message.capsule.requirements.modelFamily = ""
        message.capsule.requirements.hiddenSize = 0
        message.capsule.requirements.layerCount = 0
        message.capsule.requirementsSHA256 = try message.capsule.requirements.canonicalSHA256()

        XCTAssertThrowsError(try message.validate()) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .requirementsMismatch(
                    key: "model_family",
                    expected: "non_empty",
                    actual: ""
                )
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsEmptyRequirementHashField() throws {
        var message = try Self.message()
        message.capsule.requirements.toolSchemaSHA256 = ""
        message.capsule.requirementsSHA256 = try message.capsule.requirements.canonicalSHA256()

        XCTAssertThrowsError(try message.validate()) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .emptySHA256(field: "tool_schema_sha256")
            )
        }
    }

    func test_haloCapsuleDescriptor_rejectsArtifactByteMismatch() throws {
        var message = try Self.message()
        message.capsule.artifact.totalBytes = 101

        XCTAssertThrowsError(try message.validate()) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .invalidArtifactByteCount(expected: 101, actual: 100)
            )
        }
    }

    func test_haloCapsuleMeshMessage_rejectsUnsupportedRuntimeVersion() throws {
        let message = try Self.message(minRuntimeVersion: "1.1.0")

        XCTAssertThrowsError(try message.validate(currentRuntimeVersion: "1.0.9")) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .runtimeVersionUnsupported(minimum: "1.1.0", actual: "1.0.9")
            )
        }
    }

    func test_haloCapsuleMeshMessage_rejectsOlderRuntimeReleaseCandidate() throws {
        let message = try Self.message(minRuntimeVersion: "1.0.0-rc123")

        XCTAssertThrowsError(try message.validate(currentRuntimeVersion: "1.0.0-rc122")) { error in
            XCTAssertEqual(
                error as? HaloCapsuleTransportError,
                .runtimeVersionUnsupported(minimum: "1.0.0-rc123", actual: "1.0.0-rc122")
            )
        }
    }

    func test_haloCapsuleMeshMessage_acceptsNewerRuntimeReleaseCandidate() throws {
        let message = try Self.message(minRuntimeVersion: "1.0.0-rc123")

        XCTAssertNoThrow(try message.validate(currentRuntimeVersion: "1.0.0-rc124"))
        XCTAssertNoThrow(try message.validate(currentRuntimeVersion: "1.0.0"))
    }

    private static func message(
        minRuntimeVersion: String = "1.0.0",
        files: [HaloCapsuleArtifactFile] = [
            HaloCapsuleArtifactFile(name: "manifest.json", byteCount: 10, sha256: hash("manifest")),
            HaloCapsuleArtifactFile(name: "halo.safetensors", byteCount: 90, sha256: hash("halo")),
        ],
        tensors: [HaloCacheTensorDescriptor] = [
            HaloCacheTensorDescriptor(name: "k0", shape: [1, 2], dtype: "float16", byteCount: 4, sha256: hash("k0")),
            HaloCacheTensorDescriptor(name: "v0", shape: [1, 2], dtype: "float16", byteCount: 4, sha256: hash("v0")),
        ]
    ) throws -> HaloCapsuleMeshMessage {
        let requirements = HaloCapsuleRequirementsDescriptor(
            modelConfigSHA256: hash("config"),
            modelWeightsSHA256: hash("weights"),
            tokenizerJSONSHA256: hash("tokenizer-json"),
            tokenizerConfigSHA256: hash("tokenizer-config"),
            chatTemplateSHA256: hash("chat-template"),
            systemPromptSHA256: hash("system-prompt"),
            renderedPrefixSHA256: hash("rendered-prefix"),
            prefixTokenCount: 128,
            toolSchemaSHA256: hash("tools"),
            profileBodySHA256: hash("profile"),
            enableThinking: false,
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc123",
            cacheTopologySHA256: hash("topology"),
            modelFamily: "qwen3_5",
            hiddenSize: 4_096,
            layerCount: 32
        )
        let cacheSnapshot = HaloCacheSnapshotDescriptor(
            snapshotID: "snapshot-a",
            createdAt: date,
            tokenCount: 128,
            tokenIDsSHA256: hash("tokens"),
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc123",
            tensors: tensors
        )
        let artifact = HaloCapsuleArtifactDescriptor(
            artifactID: "artifact-a",
            totalBytes: 100,
            sha256: hash("artifact"),
            files: files
        )
        let descriptor = try HaloCapsuleDescriptor.make(
            capsuleID: "capsule-a",
            createdAt: date,
            baseModelID: "Qwen3.5-9B-4bit",
            minRuntimeVersion: minRuntimeVersion,
            requirements: requirements,
            cacheSnapshot: cacheSnapshot,
            artifact: artifact
        )
        return HaloCapsuleMeshMessage(
            transferID: "transfer-a",
            capsule: descriptor
        )
    }

    private static let date = Date(timeIntervalSince1970: 1_778_880_000)

    private static func hash(_ value: String) -> String {
        String(repeating: value, count: 64 / max(value.count, 1) + 1).prefix(64).description
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("halo_capsule_transfer_tests")
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func makePackageDirectory() throws -> URL {
        let directory = try makeTemporaryDirectory()
        try Data(#"{"schema":"manifest"}"#.utf8)
            .write(to: directory.appendingPathComponent("manifest.json"))
        try Data((0..<257).map { UInt8($0 % 251) })
            .write(to: directory.appendingPathComponent("halo.safetensors"))
        return directory
    }

    private static func messageForPackage(
        directory: URL,
        files explicitFiles: [HaloCapsuleArtifactFile]? = nil,
        artifactBytes explicitArtifactBytes: Int? = nil,
        artifactSHA256 explicitArtifactSHA256: String? = nil
    ) throws -> HaloCapsuleMeshMessage {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let haloURL = directory.appendingPathComponent("halo.safetensors")
        let manifestData = try Data(contentsOf: manifestURL)
        let haloData = try Data(contentsOf: haloURL)
        let files = explicitFiles ?? [
            HaloCapsuleArtifactFile(
                name: "manifest.json",
                byteCount: manifestData.count,
                sha256: HaloCapsulePackageTransfer.sha256Hex(manifestData)
            ),
            HaloCapsuleArtifactFile(
                name: "halo.safetensors",
                byteCount: haloData.count,
                sha256: HaloCapsulePackageTransfer.sha256Hex(haloData)
            ),
        ]
        let totalBytes = explicitArtifactBytes ?? files.reduce(0) { $0 + $1.byteCount }
        let artifactSHA256 = explicitArtifactSHA256 ?? HaloCapsulePackageTransfer.sha256Hex(
            manifestData + haloData
        )
        let requirements = HaloCapsuleRequirementsDescriptor(
            modelConfigSHA256: hash("config"),
            modelWeightsSHA256: hash("weights"),
            tokenizerJSONSHA256: hash("tokenizer-json"),
            tokenizerConfigSHA256: hash("tokenizer-config"),
            chatTemplateSHA256: hash("chat-template"),
            systemPromptSHA256: hash("system-prompt"),
            renderedPrefixSHA256: hash("rendered-prefix"),
            prefixTokenCount: 128,
            toolSchemaSHA256: hash("tools"),
            profileBodySHA256: hash("profile"),
            enableThinking: false,
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc123",
            cacheTopologySHA256: hash("topology"),
            modelFamily: "qwen3_5",
            hiddenSize: 4_096,
            layerCount: 32
        )
        let cacheSnapshot = HaloCacheSnapshotDescriptor(
            snapshotID: "snapshot-package",
            createdAt: date,
            tokenCount: 128,
            tokenIDsSHA256: hash("tokens"),
            cacheBackend: "cmlx",
            cacheBackendVersion: "edge-engine-rc123",
            tensors: [
                HaloCacheTensorDescriptor(
                    name: "k0",
                    shape: [1, 2],
                    dtype: "float16",
                    byteCount: 4,
                    sha256: hash("k0")
                )
            ]
        )
        let artifact = HaloCapsuleArtifactDescriptor(
            artifactID: "artifact-package",
            totalBytes: totalBytes,
            sha256: artifactSHA256,
            files: files
        )
        let descriptor = try HaloCapsuleDescriptor.make(
            capsuleID: "capsule-package",
            createdAt: date,
            baseModelID: "Qwen3.5-9B-4bit",
            minRuntimeVersion: "1.0.0",
            requirements: requirements,
            cacheSnapshot: cacheSnapshot,
            artifact: artifact
        )
        return HaloCapsuleMeshMessage(
            transferID: "transfer-package",
            capsule: descriptor
        )
    }

    private static func messageForPackageWithDownloadURLs(directory: URL) throws -> HaloCapsuleMeshMessage {
        let manifestURL = directory.appendingPathComponent("manifest.json")
        let haloURL = directory.appendingPathComponent("halo.safetensors")
        let manifestData = try Data(contentsOf: manifestURL)
        let haloData = try Data(contentsOf: haloURL)
        return try messageForPackage(
            directory: directory,
            files: [
                HaloCapsuleArtifactFile(
                    name: "manifest.json",
                    byteCount: manifestData.count,
                    sha256: HaloCapsulePackageTransfer.sha256Hex(manifestData),
                    downloadURL: manifestURL
                ),
                HaloCapsuleArtifactFile(
                    name: "halo.safetensors",
                    byteCount: haloData.count,
                    sha256: HaloCapsulePackageTransfer.sha256Hex(haloData),
                    downloadURL: haloURL
                ),
            ]
        )
    }

    private static func completedPackage(
        message: HaloCapsuleMeshMessage,
        packageDirectory: URL
    ) throws -> HaloCapsuleInboundTransferSession.CompletedPackage {
        HaloCapsuleInboundTransferSession.CompletedPackage(
            message: message,
            packageDirectory: packageDirectory,
            completeAck: HaloCapsuleTransferAck(
                transferID: message.transferID,
                accepted: true,
                canonicalSHA256: try message.canonicalSHA256()
            )
        )
    }

    private struct AutoRestoreTestError: Error, CustomStringConvertible {
        let message: String

        var description: String {
            message
        }
    }

    private actor AutoRestoreRecorder {
        private let restoreResult: HaloCapsuleAutoRestoreCoordinator.RestoreResult
        private let restoreError: Error?
        private let sendErrorStatus: HaloCapsuleApplyStatusValue?
        private var restored: [URL] = []
        private var sent: [HaloCapsuleApplyStatusPayload] = []

        init(
            restoreResult: HaloCapsuleAutoRestoreCoordinator.RestoreResult = .init(),
            restoreError: Error? = nil,
            sendErrorStatus: HaloCapsuleApplyStatusValue? = nil
        ) {
            self.restoreResult = restoreResult
            self.restoreError = restoreError
            self.sendErrorStatus = sendErrorStatus
        }

        func restore(
            _ completed: HaloCapsuleInboundTransferSession.CompletedPackage
        ) async throws -> HaloCapsuleAutoRestoreCoordinator.RestoreResult {
            restored.append(completed.packageDirectory)
            if let restoreError {
                throw restoreError
            }
            return restoreResult
        }

        func send(_ payload: HaloCapsuleApplyStatusPayload) async throws {
            if payload.status == sendErrorStatus {
                throw AutoRestoreTestError(message: "apply-status send failed")
            }
            sent.append(payload)
        }

        func restoredDirectories() -> [URL] {
            restored
        }

        func sentPayloads() -> [HaloCapsuleApplyStatusPayload] {
            sent
        }
    }
}
