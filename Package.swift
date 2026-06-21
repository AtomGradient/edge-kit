// swift-tools-version: 5.9
// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import PackageDescription

let package = Package(
    name: "edge-kit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "EdgeKit",
            targets: ["EdgeKit"]
        ),
        .library(name: "EdgeInference", targets: ["EdgeInference"]),
        .library(name: "EdgeModelKit", targets: ["EdgeModelKit"]),
        .library(name: "EdgeVoice",     targets: ["EdgeVoice"]),
        .library(name: "EdgeMesh",      targets: ["EdgeMesh"]),
        .library(name: "EdgeData",      targets: ["EdgeData"]),
        .library(name: "EdgeDataMeshBridge", targets: ["EdgeDataMeshBridge"]),
        .library(name: "EdgeUI",        targets: ["EdgeUI"]),
        .library(name: "EdgeSession",   targets: ["EdgeSession"]),
    ],
    dependencies: [
        .package(url: "https://github.com/AtomGradient/edge-engine.git", exact: "1.0.0-rc140"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.2"),
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.10.0"),
    ],
    targets: [
        .target(
            name: "EdgeKit",
            dependencies: [
                "EdgeInference",
                "EdgeMesh",
                "EdgeData",
                "EdgeDataMeshBridge",
                "EdgeUI",
                "EdgeVoice",
                "EdgeModelKit",
                "EdgeSession",
            ],
            path: "Sources/EdgeKit"
        ),

        .target(
            name: "EdgeInference",
            dependencies: [
                .product(name: "EdgeEngine", package: "edge-engine"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/EdgeInference"
        ),

        .target(
            name: "EdgeModelKit",
            dependencies: [
                "EdgeInference",
            ],
            path: "Sources/EdgeModelKit"
        ),

        .target(
            name: "EdgeSession",
            dependencies: [
                "EdgeInference",
            ],
            path: "Sources/EdgeSession"
        ),

        .target(
            name: "EdgeVoice",
            dependencies: [],
            path: "Sources/EdgeVoice"
        ),

        .target(
            name: "EdgeMesh",
            dependencies: [
                "EdgeInference",
            ],
            path: "Sources/EdgeMesh"
        ),

        .target(
            name: "EdgeData",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/EdgeData"
        ),

        .target(
            name: "EdgeDataMeshBridge",
            dependencies: [
                "EdgeData",
                "EdgeMesh",
            ],
            path: "Sources/EdgeDataMeshBridge"
        ),

        .target(
            name: "EdgeUI",
            dependencies: [
                "EdgeData",
            ],
            path: "Sources/EdgeUI"
        ),

        .testTarget(
            name: "EdgeKitTests",
            dependencies: [
                "EdgeKit",
                "EdgeInference",
                "EdgeModelKit",
                "EdgeVoice",
                "EdgeMesh",
                "EdgeData",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Tests/EdgeKitTests",
            resources: [
                .copy("Fixtures/route_router_manifest_v0.json"),
                .copy("Fixtures/route_matrix_input_sha256_cases.json"),
            ]
        ),

        .testTarget(
            name: "EdgeSessionTests",
            dependencies: [
                "EdgeSession",
            ],
            path: "Tests/EdgeSessionTests"
        ),

        .testTarget(
            name: "EdgeDataMeshBridgeTests",
            dependencies: [
                "EdgeDataMeshBridge",
                "EdgeMesh",
            ],
            path: "Tests/EdgeDataMeshBridgeTests"
        ),
    ]
)
