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

#ifndef EQUIDISTANT_CAM_CUH_
#define EQUIDISTANT_CAM_CUH_

__constant__ float EquidistantCameraIntrinsics[3]; // f, u0, v0

struct EquidistantCamera
{
inline __device__ static float2 project(const float3 &X)
{
	float R = hypotf(X.x, X.y);
	float theta = atan2f(R, X.z);
	float phi = atan2f(X.y, X.x);
	float2 uv;
	sincosf(phi, &(uv.y), &(uv.x));
	uv.x = fmaf(EquidistantCameraIntrinsics[0] * theta, uv.x, EquidistantCameraIntrinsics[1]);
	uv.y = fmaf(EquidistantCameraIntrinsics[0] * theta, uv.y, EquidistantCameraIntrinsics[2]);

	return uv;
}

inline __device__ static float3 unproject(const float2 &x)
{
	float2 xy = make_float2(x.x - EquidistantCameraIntrinsics[1], x.y - EquidistantCameraIntrinsics[2]);
	float theta = hypotf(xy.x, xy.y) / EquidistantCameraIntrinsics[0];
	float phi = atan2f(xy.y, xy.x);
	float3 X;
	sincosf(phi, &(X.y), &(X.x));
	float2 sc;
	sincosf(theta, &(sc.x), &(sc.y));
	X.x *= sc.x;
	X.y *= sc.x;
	X.z = sc.y;

	return X;
}
};
#endif // EQUIDISTANT_CAM_CUH_
