// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation

public enum ToolPermission: String, Sendable, Codable, Equatable, Hashable, CaseIterable {
    case readFacts = "read_facts"
    case writeFacts = "write_facts"
    case readEvents = "read_events"
    case writeEvents = "write_events"
    case readProfile = "read_profile"
    case writeProfile = "write_profile"
    case network
    case fileRead = "file_read"
    case fileWrite = "file_write"
    case appAction = "app_action"

    public var isReadOnly: Bool {
        switch self {
        case .readFacts, .readEvents, .readProfile, .fileRead:
            return true
        case .writeFacts, .writeEvents, .writeProfile, .network, .fileWrite, .appAction:
            return false
        }
    }
}
