// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import Foundation
import Metal

extension DeviceBenchmark {
    public struct MetalHardwareProbeResult: Sendable {
        public let deviceName: String
        public let supportedFamilies: [String]
        public let medianGFLOPS: Double?
        public let diagnosticLines: [String]
        public let error: String?
    }

    public static func runMetalHardwareProbe() -> MetalHardwareProbeResult {
        DeviceBenchmarkMetalHardwareProbe.run()
    }
}

private enum DeviceBenchmarkMetalHardwareProbe {
    private static let aluThreadCount = 1_048_576
    private static let aluIterations = 10_000
    private static let aluMeasuredRuns = 10
    private static let warmupRuns = 2
    private static let pointerChaseEntryCount = 4_194_304
    private static let pointerChaseSteps = 100_000
    private static let memoryMeasuredRuns = 5
    private static let sequentialBufferBytes = 64 * 1_048_576
    private static let sequentialThreadCount = 4_096

    static func run() -> DeviceBenchmark.MetalHardwareProbeResult {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return DeviceBenchmark.MetalHardwareProbeResult(
                deviceName: "unavailable",
                supportedFamilies: [],
                medianGFLOPS: nil,
                diagnosticLines: ["metal_hw_probe_unavailable reason=no_default_device"],
                error: "no_default_device"
            )
        }

        let families = supportedAppleFamilies(device)
        var lines = [
            "metal_device name=\"\(device.name)\" families=\(families.joined(separator: ",")) hasUnifiedMemory=\(device.hasUnifiedMemory)",
            "metal_device_limits maxThreadsPerThreadgroup=\(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)x\(device.maxThreadsPerThreadgroup.depth) maxThreadgroupMemory=\(device.maxThreadgroupMemoryLength) maxBufferLength=\(device.maxBufferLength) recommendedMaxWorkingSet=\(device.recommendedMaxWorkingSetSize)"
        ]

        var medianGFLOPS: Double?
        var errors: [String] = []
        do {
            let result = try runALUBenchmark(device: device)
            medianGFLOPS = result.gflops
            lines.append(
                "metal_alu_benchmark threads=\(aluThreadCount) iterations=\(aluIterations) runs=\(aluMeasuredRuns) medianMs=\(String(format: "%.3f", result.medianSeconds * 1000)) gflops=\(String(format: "%.1f", result.gflops))"
            )
        } catch {
            lines.append("metal_alu_benchmark_failed error=\"\(String(describing: error))\"")
            errors.append("alu=\(String(describing: error))")
        }

        do {
            let result = try runPointerChaseBenchmark(device: device)
            lines.append(
                "metal_memory_latency_benchmark chainSize=\(pointerChaseEntryCount) steps=\(pointerChaseSteps) runs=\(memoryMeasuredRuns) medianMs=\(String(format: "%.3f", result.medianSeconds * 1000)) latencyNs=\(String(format: "%.2f", result.latencyNs))"
            )
        } catch {
            lines.append("metal_memory_latency_benchmark_failed error=\"\(String(describing: error))\"")
            errors.append("memory_latency=\(String(describing: error))")
        }

        do {
            let result = try runSequentialReadBenchmark(device: device)
            lines.append(
                "metal_sequential_read_benchmark bufferMB=\(sequentialBufferBytes / 1_048_576) runs=\(memoryMeasuredRuns) medianMs=\(String(format: "%.3f", result.medianSeconds * 1000)) bandwidthGBs=\(String(format: "%.1f", result.bandwidthGBs))"
            )
        } catch {
            lines.append("metal_sequential_read_benchmark_failed error=\"\(String(describing: error))\"")
            errors.append("sequential_read=\(String(describing: error))")
        }

        return DeviceBenchmark.MetalHardwareProbeResult(
            deviceName: device.name,
            supportedFamilies: families,
            medianGFLOPS: medianGFLOPS,
            diagnosticLines: lines,
            error: errors.isEmpty ? nil : errors.joined(separator: "; ")
        )
    }

    private static func supportedAppleFamilies(_ device: MTLDevice) -> [String] {
        let candidates: [(String, MTLGPUFamily)] = [
            ("apple7", .apple7),
            ("apple8", .apple8),
            ("apple9", .apple9),
            ("apple10", .apple10)
        ]
        return candidates.compactMap { name, family in
            device.supportsFamily(family) ? name : nil
        }
    }

    private static func runALUBenchmark(device: MTLDevice) throws -> (medianSeconds: Double, gflops: Double) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void pure_alu_benchmark(
            device float* output [[buffer(0)]],
            uint tid [[thread_position_in_grid]]
        ) {
            float x = float(tid) * 0.001f;
            for (int i = 0; i < \(aluIterations); ++i) {
                x = fma(x, 1.00001f, 0.00001f);
            }
            output[tid] = x;
        }
        """

        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "pure_alu_benchmark") else {
            throw ProbeError.missingFunction
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw ProbeError.noCommandQueue
        }
        guard let output = device.makeBuffer(
            length: aluThreadCount * MemoryLayout<Float>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noOutputBuffer
        }

        let threadsPerGroup = MTLSize(
            width: max(1, min(256, pipeline.maxTotalThreadsPerThreadgroup)),
            height: 1,
            depth: 1
        )
        let threads = MTLSize(width: aluThreadCount, height: 1, depth: 1)

        func runOnce() throws -> Double {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ProbeError.commandBufferCreationFailed
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(output, offset: 0, index: 0)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()

            let wallStart = CFAbsoluteTimeGetCurrent()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let wallSeconds = CFAbsoluteTimeGetCurrent() - wallStart

            if let error = commandBuffer.error {
                throw error
            }
            let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            return gpuSeconds > 0 ? gpuSeconds : wallSeconds
        }

        for _ in 0..<warmupRuns {
            _ = try runOnce()
        }

        var samples: [Double] = []
        samples.reserveCapacity(aluMeasuredRuns)
        for _ in 0..<aluMeasuredRuns {
            samples.append(try runOnce())
        }
        let medianSeconds = samples.sorted()[samples.count / 2]
        let operations = Double(aluThreadCount) * Double(aluIterations) * 2.0
        return (medianSeconds, operations / medianSeconds / 1_000_000_000.0)
    }

    private static func runPointerChaseBenchmark(device: MTLDevice) throws -> (medianSeconds: Double, latencyNs: Double) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void pointer_chase(
            device const uint* chain [[buffer(0)]],
            device uint* output [[buffer(1)]],
            constant uint& steps [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            uint idx = tid;
            for (uint i = 0; i < steps; ++i) {
                idx = chain[idx];
            }
            output[tid] = idx;
        }
        """

        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "pointer_chase") else {
            throw ProbeError.missingFunction
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw ProbeError.noCommandQueue
        }
        guard let chain = device.makeBuffer(
            length: pointerChaseEntryCount * MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noInputBuffer
        }
        guard let output = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noOutputBuffer
        }
        var steps = UInt32(pointerChaseSteps)
        guard let stepsBuffer = device.makeBuffer(
            bytes: &steps,
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noParameterBuffer
        }

        buildRandomPointerChain(into: chain, count: pointerChaseEntryCount)

        let threadsPerGroup = MTLSize(width: 1, height: 1, depth: 1)
        let threads = MTLSize(width: 1, height: 1, depth: 1)

        func runOnce() throws -> Double {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ProbeError.commandBufferCreationFailed
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(chain, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBuffer(stepsBuffer, offset: 0, index: 2)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
            let seconds = try commitAndMeasure(commandBuffer)
            _ = output.contents().load(as: UInt32.self)
            return seconds
        }

        for _ in 0..<warmupRuns {
            _ = try runOnce()
        }

        var samples: [Double] = []
        samples.reserveCapacity(memoryMeasuredRuns)
        for _ in 0..<memoryMeasuredRuns {
            samples.append(try runOnce())
        }
        let medianSeconds = samples.sorted()[samples.count / 2]
        return (medianSeconds, medianSeconds * 1_000_000_000.0 / Double(pointerChaseSteps))
    }

    private static func runSequentialReadBenchmark(device: MTLDevice) throws -> (medianSeconds: Double, bandwidthGBs: Double) {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void sequential_read(
            device const float4* data [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant uint& length [[buffer(2)]],
            constant uint& stride [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            float4 sum = float4(0.0f);
            for (uint i = tid; i < length; i += stride) {
                sum += data[i];
            }
            output[tid] = sum;
        }
        """

        let library = try device.makeLibrary(source: source, options: nil)
        guard let function = library.makeFunction(name: "sequential_read") else {
            throw ProbeError.missingFunction
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue() else {
            throw ProbeError.noCommandQueue
        }
        let vectorCount = sequentialBufferBytes / MemoryLayout<SIMD4<Float>>.stride
        guard let input = device.makeBuffer(length: sequentialBufferBytes, options: [.storageModeShared]) else {
            throw ProbeError.noInputBuffer
        }
        guard let output = device.makeBuffer(
            length: sequentialThreadCount * MemoryLayout<SIMD4<Float>>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noOutputBuffer
        }
        initializeSequentialInput(input, vectorCount: vectorCount)
        var length = UInt32(vectorCount)
        var stride = UInt32(sequentialThreadCount)
        guard let lengthBuffer = device.makeBuffer(
            bytes: &length,
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noParameterBuffer
        }
        guard let strideBuffer = device.makeBuffer(
            bytes: &stride,
            length: MemoryLayout<UInt32>.stride,
            options: [.storageModeShared]
        ) else {
            throw ProbeError.noParameterBuffer
        }

        let threadsPerGroup = MTLSize(
            width: max(1, min(256, pipeline.maxTotalThreadsPerThreadgroup)),
            height: 1,
            depth: 1
        )
        let threads = MTLSize(width: sequentialThreadCount, height: 1, depth: 1)

        func runOnce() throws -> Double {
            guard let commandBuffer = queue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw ProbeError.commandBufferCreationFailed
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            encoder.setBuffer(lengthBuffer, offset: 0, index: 2)
            encoder.setBuffer(strideBuffer, offset: 0, index: 3)
            encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
            let seconds = try commitAndMeasure(commandBuffer)
            _ = output.contents().load(as: SIMD4<Float>.self)
            return seconds
        }

        for _ in 0..<warmupRuns {
            _ = try runOnce()
        }

        var samples: [Double] = []
        samples.reserveCapacity(memoryMeasuredRuns)
        for _ in 0..<memoryMeasuredRuns {
            samples.append(try runOnce())
        }
        let medianSeconds = samples.sorted()[samples.count / 2]
        return (medianSeconds, Double(sequentialBufferBytes) / medianSeconds / 1_000_000_000.0)
    }

    private static func commitAndMeasure(_ commandBuffer: MTLCommandBuffer) throws -> Double {
        let wallStart = CFAbsoluteTimeGetCurrent()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallSeconds = CFAbsoluteTimeGetCurrent() - wallStart

        if let error = commandBuffer.error {
            throw error
        }
        let gpuSeconds = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        return gpuSeconds > 0 ? gpuSeconds : wallSeconds
    }

    private static func buildRandomPointerChain(into buffer: MTLBuffer, count: Int) {
        var order = (0..<count).map(UInt32.init)
        var rng: UInt64 = 0x4d595df4d0f33173
        if count > 1 {
            for i in stride(from: count - 1, through: 1, by: -1) {
                rng = rng &* 2862933555777941757 &+ 3037000493
                let j = Int(rng % UInt64(i + 1))
                order.swapAt(i, j)
            }
        }

        let chain = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count {
            chain[Int(order[i])] = order[(i + 1) % count]
        }
    }

    private static func initializeSequentialInput(_ buffer: MTLBuffer, vectorCount: Int) {
        let data = buffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: vectorCount)
        for i in 0..<vectorCount {
            let value = Float(i % 251) * 0.001
            data[i] = SIMD4<Float>(value, value + 1, value + 2, value + 3)
        }
    }

    private enum ProbeError: Error {
        case missingFunction
        case noCommandQueue
        case noInputBuffer
        case noOutputBuffer
        case noParameterBuffer
        case commandBufferCreationFailed
    }
}
