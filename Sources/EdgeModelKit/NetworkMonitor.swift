// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Combine
import Foundation

#if canImport(Network)
import Network
#endif

@MainActor
public final class NetworkMonitor: ObservableObject {
    public static let shared = NetworkMonitor()

    @Published public private(set) var isConnected: Bool
    @Published public private(set) var isCellular: Bool
    @Published public private(set) var isExpensive: Bool

    #if canImport(Network)
    private let monitor: NWPathMonitor?
    private let queue = DispatchQueue(label: "com.atomgradient.networkmonitor")
    #endif

    public init() {
        self.isConnected = true
        self.isCellular = false
        self.isExpensive = false

        #if canImport(Network)
        let monitor = NWPathMonitor()
        self.monitor = monitor
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor [weak self] in
                self?.isConnected = path.status == .satisfied
                self?.isCellular = path.usesInterfaceType(.cellular)
                self?.isExpensive = path.isExpensive
            }
        }
        monitor.start(queue: queue)
        #endif
    }

    deinit {
        #if canImport(Network)
        monitor?.cancel()
        #endif
    }
}
