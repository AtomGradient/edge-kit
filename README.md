# AtomGradient EdgeKit

> On-device AI kit for Apple platforms — Swift Package

[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![iOS 17+](https://img.shields.io/badge/iOS-17+-blue.svg)](https://developer.apple.com/ios/)
[![macOS 14+](https://img.shields.io/badge/macOS-14+-blue.svg)](https://developer.apple.com/macos/)

## Why EdgeKit?

| | llama.cpp | mlc-llm | Core ML | **EdgeKit** |
|---|---|---|---|---|
| Apple Silicon optimized | yes | yes | yes | **yes** |
| Swift-native API | no | no | partial | **yes** |
| Auto device adaptation | no | no | no | **yes** |
| ODR model distribution | no | no | no | **yes** |
| VLM (Vision) support | partial | yes | no | **yes** |
| TTS (Speech) support | no | no | no | **yes** |

## Quick Start

### LLM — Text Chat

```swift
import EdgeKit

let runtime = EdgeRuntime()
let engine = try await runtime.loadRecommendedModel()

for try await chunk in engine.generate(messages: [.user("What is edge computing?")]) {
    print(chunk.text, terminator: "")
}
```

### VLM — Vision + Language

```swift
import EdgeKit
import CoreImage

let engine = VLMEngine()
try await engine.loadLocal(directory: modelURL)

let image = CIImage(contentsOf: photoURL)!
for try await chunk in engine.generate(
    messages: [.user("Describe this image.")],
    ciImages: [image]
) {
    print(chunk.text, terminator: "")
}
```

### TTS — Text to Speech

```swift
import EdgeKit

let engine = TTSEngine()
try await engine.loadLocal(directory: ttsModelURL)

print(engine.availableSpeakers)  // ["serena", "vivian", "ryan", ...]
let audio = try await engine.speak("Hello world", voice: "serena")
// audio.samples: [Float], audio.sampleRate: 24000, audio.duration: 2.3s
```

### Auto-detect Model Type

```swift
let runtime = EdgeRuntime()
let engine = try await runtime.loadLocal(directory: anyModelURL)

switch engine.category {
case .llm: engine.llm!.generate(messages: [...])
case .vlm: engine.vlm!.generate(messages: [...], ciImages: [...])
case .tts: try await engine.tts!.speak("Hello", voice: "serena")
}
```

## Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/AtomGradient/edge-kit.git", exact: "1.0.0-rc103")
]
```

After updating EdgeKit or its native `EdgeEngine` dependency, clean stale SwiftPM/Xcode build artifacts before rebuilding:

```bash
swift package update
swift package clean
swift test --filter NativeRuntimeBridgeTests
```

In Xcode, use **Product > Clean Build Folder** after the package update.

## Runtime State Lifecycle

`LLMEngine` has two native decode backends with separate cache state:

- Greedy decode session: Swift-owned prompt-cache bookkeeping around
  `QwenGreedyDecodeSession`.
- CMLX lazy decode session: resident backend session with its own KV/cache
  state, DSR policy bookkeeping, and command-buffer limits.

Lifecycle terms are used consistently in the engine:

- **reset** calls the backend session reset API on an existing session.
- **nil/drop** removes Swift references and local bookkeeping without implying
  the backend was reset first.
- **release** resets the backend session first, then drops Swift references and
  bookkeeping.

`clearPromptCache(resetTurnCounter:)` is the public conversation-reset entry
point. It clears the logical prompt cache, releases the CMLX lazy session, and
clears the greedy decode session. Private helpers are intentionally scoped by
backend name; `clearNativeGreedyDecodeSession()` does not release CMLX state.

## Release Notes

### 1.0.0-rc20

- Adds the `EdgeSession` product and re-exports it from `EdgeKit`.
- Implements shared chat/session infrastructure: `ChatSessionController`,
  `ConversationStore`, `HistoryCompactor`, and `InferenceRequestQueue`.
- Breaking API note: `EdgeGenerationClient` is now `@MainActor` and
  class-bound (`AnyObject`) instead of `Sendable`, so app inference owners
  should conform from their main-actor runtime/controller objects.

## Supported Models

### LLM (Text Generation)

| Model | Size | Min RAM | Tier |
|-------|------|---------|------|
| Qwen3.5-0.8B | 1.6GB | 3GB | Standard |
| Qwen3.5-2B-8bit | 2.5GB | 4GB | Standard |
| Qwen3-4B-4bit | 2.1GB | 4GB | Standard |
| Qwen3-8B-4bit | 4.3GB | 6GB | Max |
| Qwen3.5-9B-8bit | 9.7GB | 12GB | Ultra |
| Gemma-3-4B-4bit | 2.8GB | 4GB | Standard |

### VLM (Vision + Language)

| Model | Size | Min RAM |
|-------|------|---------|
| Qwen3.5-2B (VLM) | 2.5GB | 4GB |
| Qwen3-VL-4B-4bit | 2.9GB | 5GB |
| Gemma-3-4B-4bit | 2.8GB | 4GB |

Supports all VLM architectures in mlx-swift-lm: Qwen3VL, Qwen3.5, Gemma3, Mistral3, Paligemma, Pixtral, etc.

### TTS (Text to Speech)

| Model | Size | Speakers |
|-------|------|----------|
| Qwen3-TTS-0.6B CustomVoice | ~1.2GB | 9 built-in speakers |

## Architecture

```
EdgeKit (Swift Package)
├── EdgeInference       — Core inference engines
│   ├── LLMEngine       — Text generation (mlx-swift-lm / MLXLLM)
│   ├── VLMEngine       — Vision + Language (mlx-swift-lm / MLXVLM)
│   ├── TTSEngine       — Text to Speech (embedded Qwen3TTS)
│   ├── ModelCategory   — Auto-detect: LLM / VLM / TTS
│   ├── EdgeRuntime     — Unified entry point + AnyEngine
│   └── DeviceProfile   — Hardware detection + auto memory tuning
├── EdgeModelKit        — Model management (Tier / Cache / ODR / HF download)
├── EdgeVoice           — Audio recording + Whisper STT (skeleton)
└── EdgeMesh            — Multi-device mesh inference
```

## License

MIT — AtomGradient
