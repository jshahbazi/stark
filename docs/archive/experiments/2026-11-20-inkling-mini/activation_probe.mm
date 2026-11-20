// Compiles the upstream Ollama shader supplied by the caller with its width constants.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstring>

extern "C" int probe_activation(const char * shader, int width, int rows, const uint16_t * input,
                                float * inner, float * maxima, double * wall, double * gpu,
                                int iterations, int groups) {
    @autoreleasepool {
        if (groups < 1 || rows % groups != 0) return -5;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -1;
        NSError * error = nil;
        MTLCompileOptions * options = [MTLCompileOptions new];
        // Match the pinned MLX compile options and the release host's Metal 4 language.
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:shader]
                                                 options:options error:&error];
        if (!lib) { fprintf(stderr, "activation shader compilation failed\n"); return -2; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
            [lib newFunctionWithName:@"quantize_activation"] error:&error];
        if (!pipeline) return -3;
        const size_t n = size_t(rows)*width;
        id<MTLBuffer> x = [device newBufferWithBytes:input length:n*2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> values = [device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> amax = [device newBufferWithLength:rows*4 options:MTLResourceStorageModeShared];
        auto queue = [device newCommandQueue];
        const int threads = ((width/16 + 31)/32)*32;
        for (int i=-2; i<iterations; ++i) {
            auto begin = std::chrono::steady_clock::now();
            auto cmd = [queue commandBuffer];
            auto enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            const int rows_per_group = rows / groups;
            for (int group = 0; group < groups; ++group) {
                const size_t first_row = size_t(rows_per_group)*group;
                [enc setBuffer:x offset:first_row*width*2 atIndex:0];
                [enc setBuffer:values offset:first_row*width*4 atIndex:1];
                [enc setBuffer:amax offset:first_row*4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1,rows_per_group,1)
                    threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            }
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
            if (i >= 0) {
                wall[i] = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                gpu[i] = cmd.GPUEndTime - cmd.GPUStartTime;
            }
        }
        std::memcpy(inner, values.contents, n*4);
        std::memcpy(maxima, amax.contents, rows*4);
    }
    return 0;
}
