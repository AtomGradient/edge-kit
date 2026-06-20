// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import AVFoundation

/// Records microphone audio to a local WAV file.
/// AVAudioSession is iOS-only; macOS uses AVAudioEngine directly.
@MainActor
public final class AudioRecorder: NSObject, ObservableObject {

    @Published public private(set) var isRecording = false
    @Published public private(set) var currentLevel: Float = 0.0

    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    private var outputFile: AVAudioFile?

    private var outputURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("EdgeVoice/recording_\(UUID().uuidString).wav")
    }

    public override init() {}

    public func startRecording() async throws -> URL {
#if os(iOS)
        try AVAudioSession.sharedInstance().setCategory(.record, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)
#endif

        let engine = AVAudioEngine()
        audioEngine = engine
        inputNode = engine.inputNode

        let format = engine.inputNode.outputFormat(forBus: 0)
        let url = outputURL
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        outputFile = file

        engine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            try? file.write(from: buffer)
            let level = self.calculateLevel(buffer: buffer)
            Task { @MainActor in self.currentLevel = level }
        }

        try engine.start()
        isRecording = true
        return url
    }

    public func stopRecording() -> URL? {
        guard isRecording, let url = outputFile?.url else { return nil }
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        outputFile = nil
        isRecording = false
#if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false)
#endif
        return url
    }

    private func calculateLevel(buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData?[0] else { return 0 }
        let frameCount = Int(buffer.frameLength)
        let sum = (0..<frameCount).reduce(Float(0)) { $0 + abs(data[$1]) }
        return sum / Float(frameCount)
    }
}
