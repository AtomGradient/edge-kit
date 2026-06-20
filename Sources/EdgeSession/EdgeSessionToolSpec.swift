// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/// Tool specification shape used by EdgeSession without taking a direct
/// dependency on Tokenizers.
///
/// This is intentionally layout-compatible with `Tokenizers.ToolSpec`, whose
/// public definition is also `[String: any Sendable]`.
public typealias EdgeSessionToolSpec = [String: any Sendable]
