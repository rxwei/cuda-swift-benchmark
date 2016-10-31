import NVRTC
import CUDARuntime
import Foundation
import Dispatch
import CuBLAS

guard let device = Device.current else {
    print("No CUDA device available")
    exit(1)
}

/// Iterations
let iterationCount = 50
let n: Int = 1 << 21
print("Iteration count: \(iterationCount)")
print("Vector size: n = \(n)")

/// Load cuBLAS
let blasLoadingStart = DispatchTime.now().uptimeNanoseconds
_ = BLAS.main
let blasLoadingTime = DispatchTime.now().uptimeNanoseconds - blasLoadingStart
print("cuBLAS loading time: \(blasLoadingTime)")

/// Compile kernel
let compileStart = DispatchTime.now().uptimeNanoseconds
let daxpySource =
    "extern \"C\" __global__ void daxpy(size_t n, double a, double *x, double *y, double *z) {"
  + "    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (tid < n) z[tid] = a * x[tid] + y[tid];"
  + "}"
let module = try Module(source: daxpySource, compileOptions: [
    .computeCapability(device.computeCapability),
    .useFastMath
])
let compileTime = DispatchTime.now().uptimeNanoseconds - compileStart
let daxpy = module.function(named: "daxpy")!
print("Kernel compile+load time: \(compileTime)")

/// Host data
let a: Double = 5.1
var hostX = Array(repeating: 1.0, count: n)
var hostY = Array(sequence(first: 0.0, next: {$0+1}).prefix(n))
var hostResult = Array<Double>(repeating: 0, count: n)

/// Device data
let h2dStart = DispatchTime.now().uptimeNanoseconds
let deviceA = DeviceValue(a)
var x = DeviceArray<Double>(fromHost: hostX)
var y = DeviceArray<Double>(fromHost: hostY)
var result = DeviceArray<Double>(capacity: n)
let h2dTime = DispatchTime.now().uptimeNanoseconds - h2dStart
print("Host-to-device memcpy time: \(h2dTime)")

/// Run on CPU
print("Running daxpy on CPU...")

let cpuStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    for i in 0..<n {
        hostResult[i] = a * hostX[i] + hostY[i]
    }
}
let cpuTime = DispatchTime.now().uptimeNanoseconds - cpuStart
print("Time: \(cpuTime)")

/// Copy Y to Z for BLAS
result = y
/// Run on BLAS
print("Running daxpy on GPU using cuBLAS...")
let blasStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    BLAS.main.add(x, multipliedBy: deviceA, onto: &result)
}
let blasTime = DispatchTime.now().uptimeNanoseconds - blasStart
print("Time: \(blasTime)")


/// Run on GPU
/// Add arguments to a list
print("Running daxpy on GPU using compiled kernel...")
var args = ArgumentList()
args.append(Int32(n))    /// count
args.append(a)           /// a
args.append(&x)          /// X
args.append(&y)          /// Y
args.append(&result)     /// Z
let gpuStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    try daxpy<<<(n/256, 256)>>>(args)
}
let gpuTime = DispatchTime.now().uptimeNanoseconds - gpuStart
print("Time: \(gpuTime)")

print("Compiled GPU kernel is \(Float(cpuTime)/Float(gpuTime))x faster than CPU, and \(Float(blasTime)/Float(gpuTime))x faster than cuBLAS")
