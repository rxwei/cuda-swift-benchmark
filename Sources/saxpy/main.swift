import NVRTC
import CUDADriver
import Foundation
import Dispatch
import CuBLAS
import Warp

let device = Device.current
print(device.computeCapability)

/// Iterations
let iterationCount = 100
let n: Int = 1 << 21
print("Iteration count: \(iterationCount)")
print("Vector size: n = \(n)")

/// Load cuBLAS
let blasLoadingStart = DispatchTime.now().uptimeNanoseconds
let blas = BLAS.global(on: device)
let blasLoadingTime = DispatchTime.now().uptimeNanoseconds - blasLoadingStart
print("cuBLAS loading time: \(blasLoadingTime)")

/// Compile kernel
let compileStart = DispatchTime.now().uptimeNanoseconds
let daxpySource =
    "extern \"C\" __global__ void daxpy(long long n, double a, double *x, double *y, double *z) {"
  + "    long long tid = blockIdx.x * blockDim.x + threadIdx.x;"
  + "    if (tid < n) { z[tid] = a * x[tid] + y[tid]; }"
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
var x = DeviceArray(hostX)
var y = DeviceArray(hostY)
var result = DeviceArray<Double>(capacity: n)
let h2dTime = DispatchTime.now().uptimeNanoseconds - h2dStart
print("Host-to-device memcpy time: \(h2dTime)")

print("Preparing Warp kernel...")
result.assign(from: .addition,
              left: x, multipliedBy: a,
              right: y)

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
result = DeviceArray(y)
/// Run on BLAS
print("Running daxpy on GPU using cuBLAS...")
let blasStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    blas.axpy(alpha: a,
              x: x.unsafeDevicePointer, stride: 1,
              y: result.unsafeMutableDevicePointer, stride: 1,
              count: Int32(x.count))
}
let blasTime = DispatchTime.now().uptimeNanoseconds - blasStart
print("Time: \(blasTime)")


/// Run on GPU
/// Add arguments to a list
print("Running daxpy on GPU using compiled kernel...")
let gpuStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    try daxpy<<<(n/256, 256)>>>[
        .longLong(Int64(n)),         // count
        .double(a),           // alpha
        .constPointer(to: x), // &x
        .constPointer(to: y), // &y
        .pointer(to: &result) // &result
    ]
}
let gpuTime = DispatchTime.now().uptimeNanoseconds - gpuStart
print("Time: \(gpuTime)")

/// Run using Warp library (which is based on compiled kernels)
print("Running daxpy on GPU using Warp library")
let warpStart = DispatchTime.now().uptimeNanoseconds
for _ in 0..<iterationCount {
    result.assign(from: .addition,
                  left: x, multipliedBy: a,
                  right: y)
}
let warpTime = DispatchTime.now().uptimeNanoseconds - warpStart
print("Time: \(warpTime)")

print("Compiled GPU kernel is \(Float(cpuTime)/Float(gpuTime))x faster than CPU, and \(Float(blasTime)/Float(gpuTime))x faster than cuBLAS, \(Float(warpTime)/Float(gpuTime))x faster than Warp.")
