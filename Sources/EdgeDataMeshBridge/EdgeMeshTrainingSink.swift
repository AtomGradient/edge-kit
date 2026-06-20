// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import EdgeData
import EdgeMesh
import Foundation

@available(iOS 17.0, macOS 14.0, *)
public final class EdgeMeshTrainingSink: EdgeTrainingDataSink {
    public let collector: DataCollector
    public let store: EventStore

    public init(storeURL: URL = EventStore.defaultURL()) throws {
        let store = try EventStore(url: storeURL)
        self.store = store
        self.collector = DataCollector(store: store)
    }

    public init(store: EventStore) {
        self.store = store
        self.collector = DataCollector(store: store)
    }

    public func collectTrainingSample(
        appId: String,
        eventType: String,
        payload: Data,
        tags: [String]
    ) throws {
        let meshTags = Set(tags.compactMap(EventTag.init(rawValue:)))
        _ = try collector.appendEvent(
            appId: appId,
            eventType: eventType,
            payload: payload,
            tags: meshTags
        )
    }

    public func eventStoreCount() -> Int? {
        try? store.count()
    }
}
