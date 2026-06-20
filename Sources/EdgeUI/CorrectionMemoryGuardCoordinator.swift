// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public struct CorrectionMemoryGuardModelState: Codable, Equatable, Sendable {
    public var isLoaded: Bool
    public var isBusy: Bool

    public init(isLoaded: Bool, isBusy: Bool) {
        self.isLoaded = isLoaded
        self.isBusy = isBusy
    }
}

public enum CorrectionMemoryGuardAction: String, Codable, Equatable, Sendable {
    case unloadModelBeforeCorrection = "unload_model_before_correction"
    case skipUnloadModelBusy = "skip_unload_model_busy"
    case noModelLoaded = "no_model_loaded"
}

public struct CorrectionMemoryGuardReceipt: Codable, Equatable, Sendable {
    public var surface: String
    public var action: CorrectionMemoryGuardAction
    public var modelLoadedBefore: Bool
    public var modelBusyBefore: Bool
    public var modelLoadedAfter: Bool

    public init(
        surface: String,
        action: CorrectionMemoryGuardAction,
        modelLoadedBefore: Bool,
        modelBusyBefore: Bool,
        modelLoadedAfter: Bool
    ) {
        self.surface = surface
        self.action = action
        self.modelLoadedBefore = modelLoadedBefore
        self.modelBusyBefore = modelBusyBefore
        self.modelLoadedAfter = modelLoadedAfter
    }
}

@MainActor
public struct CorrectionMemoryGuardCoordinator {
    public typealias StateReader = () -> CorrectionMemoryGuardModelState
    public typealias ModelUnloader = () -> Void

    private let readState: StateReader
    private let unloadModel: ModelUnloader

    public init(
        readState: @escaping StateReader,
        unloadModel: @escaping ModelUnloader
    ) {
        self.readState = readState
        self.unloadModel = unloadModel
    }

    public func prepareForCorrection(surface: String) -> CorrectionMemoryGuardReceipt {
        let before = readState()
        let action: CorrectionMemoryGuardAction

        if before.isLoaded && !before.isBusy {
            unloadModel()
            action = .unloadModelBeforeCorrection
        } else if before.isBusy {
            action = .skipUnloadModelBusy
        } else {
            action = .noModelLoaded
        }

        let after = readState()
        return CorrectionMemoryGuardReceipt(
            surface: surface,
            action: action,
            modelLoadedBefore: before.isLoaded,
            modelBusyBefore: before.isBusy,
            modelLoadedAfter: after.isLoaded
        )
    }
}
