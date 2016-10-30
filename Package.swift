import PackageDescription

let package = Package(
    name: "CUDABenchmark",
    dependencies: [
        .Package(url: "https://github.com/rxwei/cuda-swift", majorVersion: 1)
    ]
)
