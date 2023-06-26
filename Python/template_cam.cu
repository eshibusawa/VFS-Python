// This file is part of VFS-Python.
// Copyright (c) 2023, Eijiro SHIBUSAWA <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

extern "C" __global__ void getRays(float3 *output,
	int height, int width)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexU >= width) || (indexV >= height))
	{
		return;
	}

	const int indexOutput = indexU + indexV * width;
	output[indexOutput] =  CameraModel::unproject(make_float2(indexU, indexV));
}

extern "C" __global__ void getThetas(float3 *output,
	const float3 * __restrict__ rays,
	int height, int width)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexU >= width) || (indexV >= height))
	{
		return;
	}

	const int index = indexU + indexV * width;
	output[index].x = atan2f(rays[index].x, rays[index].z);
	output[index].y = atan2f(rays[index].y, rays[index].z);
	output[index].z = atan2f(hypotf(rays[index].x, rays[index].y), rays[index].z);
}

extern "C" __global__ void getUVFromXYZ(
	float2* output,
    const float3* __restrict__ xyz,
	int height,
	int width)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexU >= width) || (indexV >= height))
	{
		return;
	}
	const int index = indexU + indexV * width;
	output[index] = CameraModel::project(xyz[index]);
}

extern "C" __global__ void getXYZFromUV(
	float3* output,
    const float2* __restrict__ uv,
	int height,
	int width)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexU >= width) || (indexV >= height))
	{
		return;
	}
	const int index = indexU + indexV * width;
	output[index] = CameraModel::unproject(uv[index]);
}

extern "C" __global__ void getXYZFromDistance(
	float3* output,
    const float3* __restrict__ rays,
    const float* __restrict__ distance,
  	cudaTextureObject_t texMask,
	int height,
	int width)
{
	const int indexU = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexV = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexU >= width) || (indexV >= height))
	{
		return;
	}
	const int index = indexU + indexV * width;
    if (tex2D<unsigned char>(texMask, indexU, indexV) == 0)
    {
    	output[index] =  make_float3(0, 0, -1);
        return;
    }

	output[index].x =  distance[index] * rays[index].x;
	output[index].y =  distance[index] * rays[index].y;
	output[index].z =  distance[index] * rays[index].z;
}
