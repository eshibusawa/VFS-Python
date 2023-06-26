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

inline __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator*(float a, float2 b)
{
	return make_float2(a * b.x, a * b.y);
}

template<typename TYPE>
struct Bilinear
{
	__device__ static inline TYPE get(cudaTextureObject_t input, int X, int Y, int height, int width)
	{
		float dx = 1.0f / width;
		float dy = 1.0f / height;
		const float x = (X + 0.5f) * dx;
		const float y = (Y + 0.5f) * dy;
		dx *= 0.25f;
		dy *= 0.25f;
		return 0.25f * (tex2D<TYPE>(input, x - dx, y) + tex2D<TYPE>(input, x + dx, y) +
			tex2D<TYPE>(input, x, y - dy) + tex2D<TYPE>(input, x, y + dy));
	}
};

template<typename TYPE>
struct NearestNeighbor
{
	__device__ static inline TYPE get(cudaTextureObject_t input, int X, int Y, int height, int width)
	{
		const float x = (X + 0.5f) / width;
		const float y = (Y + 0.5f) / height;
		return tex2D<TYPE>(input, x, y);
	}

	__device__ static inline TYPE get(cudaTextureObject_t input, int X, int Y, float warpX, float warpY, int height, int width)
	{
		const float x = (X + warpX + 0.5f) / width;
		const float y = (Y + warpY + 0.5f) / height;
		return tex2D<TYPE>(input, x, y);
	}
};

template<typename TYPE>
struct NearestNeighborScaled
{
	__device__ static inline TYPE get(cudaTextureObject_t input, int X, int Y, int height, int width)
	{
		static const float scale(VFS_FSCALE);
		const float x = (X + 0.5f) / width;
		const float y = (Y + 0.5f) / height;
		return scale * tex2D<TYPE>(input, x, y);
	}
};

template<typename TYPE>
struct Gaussian
{
	__device__ static inline TYPE get(cudaTextureObject_t input, int X, int Y, int height, int width)
	{
		const float dx = 1.f / width;
		const float dy = 1.f / height;
		const float x = (X + 0.5f) * dx;
		const float y = (Y + 0.5f) * dy;
		TYPE val0 = tex2D<TYPE>(input, x, y);
		TYPE val1 = tex2D<TYPE>(input, x - dx, y - dy);
		val1 += tex2D<TYPE>(input, x - dx, y + dy);
		val1 += tex2D<TYPE>(input, x + dx, y - dy);
		val1 += tex2D<TYPE>(input, x + dx, y + dy);
		TYPE val2 = tex2D<TYPE>(input, x - dx, y);
		val2 += tex2D<TYPE>(input, x + dx, y);
		val2 += tex2D<TYPE>(input, x, y - dy);
		val2 += tex2D<TYPE>(input, x, y + dy);

		return (.25f * val0) + (.0625 * val1) + (.125f * val2);
	}
};

struct Derivative
{
	__device__ static inline float2 get(cudaTextureObject_t input, int X, int Y, int height, int width)
	{
		const float dx = 1.f / width;
		const float dy = 1.f / height;
		const float x = (X + 0.5f) * dx;
		const float y = (Y + 0.5f) * dy;
		float val0 = tex2D<float>(input, x + dx, y + dy);
		val0 -= tex2D<float>(input, x, y + dy);
		float val1 = tex2D<float>(input, x + dx, y + dy);
		val1 -= tex2D<float>(input, x + dx, y);

		return make_float2(val0, val1);
	}

	__device__ static inline float2 get(cudaTextureObject_t inputRef, cudaTextureObject_t inputOther,
		cudaTextureObject_t inputTranslation,
		int X, int Y, int height, int width)
	{
		const float x = (X + 0.5f) / width;
		const float y = (Y + 0.5f) / height;
		const float2 vec = tex2D<float2>(inputTranslation, x, y);
		const float norm = hypotf(vec.x, vec.y);
		const float dx = vec.x / norm / width;
		const float dy = vec.y / norm / height;
		float val0 = tex2D<float>(inputRef, x - 2 * dx, y - 2 * dy);
		val0 -= tex2D<float>(inputRef, x - dx, y - dy) * 8;
		val0 += tex2D<float>(inputRef, x + dx, y + dy) * 8;
		val0 -= tex2D<float>(inputRef, x + 2 * dx, y + 2 * dy);
		float val1 = tex2D<float>(inputOther, x - 2 * dx, y - 2 * dy);
		val1 -= tex2D<float>(inputOther, x - dx, y - dy) * 8;
		val1 += tex2D<float>(inputOther, x + dx, y + dy) * 8;
		val1 -= tex2D<float>(inputOther, x + 2 * dx, y + 2 * dy);
		float val2 = tex2D<float>(inputRef, x, y);
		float val3 = tex2D<float>(inputOther, x, y);

		return make_float2((val0 + val1)/24, val3 - val2);
	}
};

extern "C" __global__ void castAndScale(float *output, unsigned char * __restrict__ input, int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
			return;
	}

	const int index = indexY * width + indexX;
	output[index] = static_cast<float>(input[index]) / static_cast<float>(256);
}

template<typename TYPE, typename METHOD>
__device__ void sampling_(TYPE *output, cudaTextureObject_t input, int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
			return;
	}
	const int index = indexY * width + indexX;
	output[index] = METHOD::get(input, indexX, indexY, height, width);
}

extern "C" __global__ void downSamplingImage(float *output, cudaTextureObject_t input, int height, int width)
{
	sampling_<float, Bilinear<float> >(output, input, height, width);
}

extern "C" __global__ void downSamplingMask(unsigned char *output, cudaTextureObject_t input, int height, int width)
{
	sampling_<unsigned char, NearestNeighbor<unsigned char> >(output, input, height, width);
}

extern "C" __global__ void downSamplingVectorField(float2 *output, cudaTextureObject_t input, int height, int width)
{
	sampling_<float2, Bilinear<float2> >(output, input, height, width);
}

template<typename TYPE, typename METHOD>
__device__ void warping_(TYPE *output, const float2 * __restrict__ warp, cudaTextureObject_t input, int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
			return;
	}
	const int index = indexY * width + indexX;
	output[index] = METHOD::get(input, indexX, indexY, warp[index].x, warp[index].y, height, width);
}

extern "C" __global__ void warpingImage(float *output, const float2 * __restrict__ warp, cudaTextureObject_t input, int height, int width)
{
	warping_<float, NearestNeighbor<float> >(output, warp, input, height, width);
}

template<typename TYPE, typename METHOD, typename MEHTOD_MASK>
__device__ void samplingMasked_(TYPE *output, cudaTextureObject_t input, cudaTextureObject_t inputMask,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (MEHTOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		output[index] = 0;
		return;
	}
	output[index] = METHOD::get(input, indexX, indexY, height, width);
}

extern "C" __global__ void gaussianMasked(float *output, cudaTextureObject_t input, cudaTextureObject_t inputMask, int height, int width)
{
	samplingMasked_<float, Gaussian<float>, NearestNeighbor<unsigned char> >(output, input, inputMask, height, width);
}

__device__ inline float3 computeEta(float a, float b, float c)
{
	static const float a004(4 * (VFS_ALPHA0) * (VFS_ALPHA0));
	static const float a11((VFS_ALPHA1) * (VFS_ALPHA1));
	return make_float3(
		fmaf(a11, (fmaf(a, a, b * b) + 2 * c*c), fmaf(a11, fmaf(a + c, a + c, fmaf((b + c), (b + c), 0)), 0)),
		fmaf(a11, fmaf(b, b, c * c), a004),
		fmaf(a11, fmaf(a, a, c * c), a004)
	);
}

template<typename METHOD, typename MEHTOD_MASK>
__device__ void computeTensorEtaMasked_(float3 *outputTensor, float3 *outputEta, cudaTextureObject_t input, cudaTextureObject_t inputMask,
	int height, int width)
{
	static const float beta(VFS_BETA);
	static const float gamma(VFS_GAMMA);
	static const float minNornalizedGrad(VFS_MIN_NORMALIZED_GRAD);
	static const float minTensorVal(VFS_MIN_TESOR_VALUE);

	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (MEHTOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		outputTensor[index] = outputEta[index] = make_float3(0, 0, 0);
		return;
	}
	float2 grad = METHOD::get(input, indexX, indexY, height, width);
	const float normGrad = hypotf(grad.x, grad.y);
	float2 normalizedGrad = make_float2(grad.x / normGrad, grad.y / normGrad);
	if (normGrad < minNornalizedGrad)
	{
		normalizedGrad = make_float2(1, 0);
	}

	float2 normalizedGradT = make_float2(normalizedGrad.y, -normalizedGrad.x);

	float weight = fmaxf(expf(-beta * powf(normGrad, gamma)), minTensorVal);
	float a = weight * normalizedGrad.x * normalizedGrad.x + normalizedGradT.x * normalizedGradT.x;
	float b = weight * normalizedGrad.y * normalizedGrad.y + normalizedGradT.y * normalizedGradT.y;
	float c = weight * normalizedGrad.x * normalizedGrad.y + normalizedGradT.x * normalizedGradT.y;
	outputTensor[index] = make_float3(a, b, c);
	outputEta[index] = computeEta(a, b, c);
}

extern "C" __global__ void computeTensorEtaMasked(float3 *outputTensor, float3 *outputEta, cudaTextureObject_t input, cudaTextureObject_t inputMask, int height, int width)
{
	computeTensorEtaMasked_<Derivative, NearestNeighbor<unsigned char> >(outputTensor, outputEta, input, inputMask, height, width);
}

template<typename MEHTOD>
__device__ void findWarpingVector_(float2 *output,
	const float2 * __restrict__ warpUV, cudaTextureObject_t input,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	output[index] = MEHTOD::get(input, indexX, indexY, warpUV[index].x, warpUV[index].y, height, width);
}

extern "C" __global__ void findWarpingVector(float2 *output,
	const float2 * __restrict__ warpUV, cudaTextureObject_t input, int height, int width)
{
	findWarpingVector_<NearestNeighbor<float2> >(output, warpUV, input, height, width);
}

template<typename MEHTOD>
__device__ void computeDerivatives_(float2 *output,
	cudaTextureObject_t inputRef, cudaTextureObject_t inputOther, cudaTextureObject_t inputTranslation,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	output[index] = MEHTOD::get(inputRef, inputOther, inputTranslation, indexX, indexY, height, width);
}

extern "C" __global__ void computeDerivatives(float2 *output,
	cudaTextureObject_t inputRef, cudaTextureObject_t inputOther, cudaTextureObject_t inputTranslation,
	int height, int width)
{
	computeDerivatives_<Derivative>(output, inputRef, inputOther, inputTranslation, height, width);
}

template<typename METHOD_MASK>
__device__ void updateDualVariables_(float2 *outputP, float2 *outputTp, float4 *outputQ,
	float sigma,
	const float * __restrict__ u, const float2 * __restrict__ v,
	const float3 * __restrict__ abc,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	static const float alpha0(VFS_ALPHA0);
	static const float alpha1(VFS_ALPHA1);
	static const float etaP(VFS_ETA_P);
	static const float etaQ(VFS_ETA_Q);
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		outputP[index] = outputTp[index] = make_float2(0, 0);
		outputQ[index] = make_float4(0, 0, 0, 0);
		return;
	}

	// 4-neighbor
	const int indexE = indexY * width + (indexX + 1);
	const int indexW = indexY * width + (indexX - 1);
	const int indexS = (indexY + 1) * width + indexX;
	const int indexN = (indexY - 1) * width + indexX;
	float2 gradU = make_float2(0, 0);
	float4 gradV = make_float4(0, 0, 0, 0);
	if (METHOD_MASK::get(inputMask, indexX + 1, indexY, height, width) != 0) // E
	{
		gradU.x = u[indexE] - u[index] - v[index].x;
		gradV.x = v[indexE].x - v[index].x;
		gradV.w = v[indexE].y - v[index].y;
	}
	else // W
	{
		gradU.x = u[index] - u[indexW] - v[index].x;
		gradV.x = v[index].x - v[indexW].x;
		gradV.w = v[index].y - v[indexW].y;
	}
	if (METHOD_MASK::get(inputMask, indexX, indexY + 1, height, width) != 0) // S
	{
		gradU.y = u[indexS] - u[index] - v[index].y;
		gradV.y = v[indexS].y - v[index].y;
		gradV.z = v[indexS].x - v[index].x;
	}
	else // N
	{
		gradU.y = u[index] - u[indexN] - v[index].y;
		gradV.y = v[index].y - v[indexN].y;
		gradV.z = v[index].x - v[indexN].x;
	}
	// update P, TP
	float2 tu = make_float2(
		fmaf(abc[index].x, gradU.x, abc[index].z * gradU.y),
		fmaf(abc[index].z, gradU.x, abc[index].y * gradU.y));
	float scale = alpha1 * sigma / etaP;
	float2 updatedP = make_float2(
		fmaf(scale, tu.x, outputP[index].x),
		fmaf(scale, tu.y, outputP[index].y));
	float norm = fmaxf(hypotf(updatedP.x, updatedP.y), 1.0f);
	outputP[index] = make_float2(updatedP.x / norm, updatedP.y / norm);
	outputTp[index] = make_float2(
		fmaf(abc[index].x, outputP[index].x, abc[index].z * outputP[index].y),
		fmaf(abc[index].z, outputP[index].x, abc[index].y * outputP[index].y));
	// update Q
	scale = alpha0 * sigma / etaQ;
	float4 updatedQ = make_float4(
		fmaf(scale, gradV.x, outputQ[index].x),
		fmaf(scale, gradV.y, outputQ[index].y),
		fmaf(scale, gradV.z, outputQ[index].z),
		fmaf(scale, gradV.w, outputQ[index].w));
	norm = fmaxf(norm4df(updatedQ.x, updatedQ.y, updatedQ.z, updatedQ.w), 1.0f);
	outputQ[index] = make_float4(
		updatedQ.x / norm,
		updatedQ.y / norm,
		updatedQ.z / norm,
		updatedQ.w / norm);
}

extern "C" __global__ void updateDualVariables(float2 *outputP, float2 *outputTp, float4 *outputQ,
	float sigma,
	const float * __restrict__ u, const float2 * __restrict__ v,
	const float3 * __restrict__ abc,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	updateDualVariables_<NearestNeighbor<unsigned char> >(outputP, outputTp, outputQ, sigma, u, v, abc, inputMask, height, width);
}

template<typename METHOD_MASK>
__device__ void l1Thresholding_(float *output,
	const float tau, const float2 * __restrict__ tp,
	const float * __restrict__ u, const float * __restrict__ u_,
	const float2 * __restrict__ deriv_uz, const float3 * __restrict__ eta,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	static const float lambda(VFS_LAMBDA);
	static const float eps(1E-7);
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		output[index] = 0;
		return;
	}

	// 4-neighbor
	const int indexW = indexY * width + (indexX - 1);
	const int indexN = (indexY - 1) * width + indexX;
	float2 gradTp = make_float2(0, 0);
	if ((METHOD_MASK::get(inputMask, indexX + 1, indexY, height, width) != 0) && // E-W
		(METHOD_MASK::get(inputMask, indexX - 1, indexY, height, width) != 0))
	{
		gradTp.x = tp[index].x - tp[indexW].x;
	}
	else if (METHOD_MASK::get(inputMask, indexX + 1, indexY, height, width) == 0)
	{
		gradTp.x = -tp[indexW].x;
	}
	else
	{
		gradTp.x = tp[index].x;
	}

	if ((METHOD_MASK::get(inputMask, indexX, indexY + 1, height, width) != 0) && // S-N
		(METHOD_MASK::get(inputMask, indexX, indexY - 1, height, width) != 0))
	{
		gradTp.y = tp[index].y - tp[indexN].y;
	}
	else if (METHOD_MASK::get(inputMask, indexX, indexY + 1, height, width) == 0)
	{
		gradTp.y = -tp[indexN].y;
	}
	else
	{
		gradTp.y = tp[index].y;
	}

	const float &dIu = deriv_uz[index].x, &dIz = deriv_uz[index].y;
	float tauEtaU = (fabsf(eta[index].x) < eps) ? tau : tau / (eta[index].x);
	float uTauGtp = u_[index] + tauEtaU * (gradTp.x + gradTp.y);
	float rho = dIu * (uTauGtp - u[index]) + dIz;
	const float bound = lambda * tauEtaU * dIu * dIu;
	float du = 0.f;
	if ((rho <= bound) && (rho >= -bound))
	{
		du = (fabsf(dIu) < eps) ? ((uTauGtp - u[index])) : ((uTauGtp - u[index]) - rho / dIu);
	}
	else if (rho < -bound)
	{
		du = (uTauGtp - u[index]) + lambda * tauEtaU * dIu;
	}
	else if (rho > bound)
	{
		du = (uTauGtp - u[index]) - lambda * tauEtaU * dIu;
	}
	output[index] = u[index] + du;
}

extern "C" __global__ void l1Thresholding(float *output,
	const float tau, const float2 * __restrict__ tp,
	const float * __restrict__ u, 	const float * __restrict__ u_,
	const float2 * __restrict__ deriv_uz, const float3 * __restrict__ eta,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	l1Thresholding_<NearestNeighbor<unsigned char> >(output, tau, tp, u, u_, deriv_uz, eta, inputMask, height, width);
}

template<typename METHOD_MASK>
__device__ void updatePrimalVariables_(float *outputU, float2 *outputV,
	const float tau, const float mu,
	const float2 * __restrict__ p, const float4 * __restrict__ q,
	const float * __restrict__ u, const float * __restrict__ u_, const float2 * __restrict__ v_,
	const float3 * __restrict__ abc, const float3 * __restrict__ eta,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	static const float alpha0(VFS_ALPHA0);
	static const float alpha1(VFS_ALPHA1);
	static const float eps(1E-7);
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		outputU[index] = 0;
		outputV[index] = make_float2(0, 0);
		return;
	}

	// 4-neighbor
	const int indexW = indexY * width + (indexX - 1);
	const int indexN = (indexY - 1) * width + indexX;
	float4 gradQ = make_float4(0, 0, 0, 0);
	if (METHOD_MASK::get(inputMask, indexX + 1, indexY, height, width) != 0) // E
	{
		gradQ.x = q[index].x;
		gradQ.w = q[index].w;
	}
	if (METHOD_MASK::get(inputMask, indexX - 1, indexY, height, width) != 0) // W
	{
		gradQ.x -= q[indexW].x;
		gradQ.w -= q[indexW].w;
	}

	if (METHOD_MASK::get(inputMask, indexX, indexY + 1, height, width) != 0) // S
	{
		gradQ.y = q[index].y;
		gradQ.z = q[index].z;
	}
	if (METHOD_MASK::get(inputMask, indexX, indexY - 1, height, width) != 0) // N
	{
		gradQ.y -= q[indexN].y;
		gradQ.z -= q[indexN].z;
	}
	const float2 tu = make_float2(
		fmaf(abc[index].x, p[index].x, abc[index].z * p[index].y),
		fmaf(abc[index].z, p[index].x, abc[index].y * p[index].y));

	const float tauEta1 = (fabsf(eta[index].y) < eps) ? tau : (tau / eta[index].y);
	const float tauEta2 = (fabsf(eta[index].z) < eps) ? tau : (tau / eta[index].z);
	const float2 updatedV = make_float2(
		fmaf(tauEta1, fmaf(alpha1, tu.x, alpha0 * (gradQ.x + gradQ.z)), v_[index].x),
		fmaf(tauEta2, fmaf(alpha1, tu.y, alpha0 * (gradQ.w + gradQ.y)), v_[index].y));
	outputU[index] = fmaf(mu, u[index] - u_[index], u[index]);
	outputV[index].x = fmaf(mu, updatedV.x - v_[index].x, updatedV.x);
	outputV[index].y = fmaf(mu, updatedV.y - v_[index].y, updatedV.y);
}

extern "C" __global__ void updatePrimalVariables(float *outputU, float2 *outputV,
	const float tau, const float mu,
	const float2 * __restrict__ p, const float4 * __restrict__ q,
	const float * __restrict__ u, const float * __restrict__ u_, const float2 * __restrict__ v_,
	const float3 * __restrict__ abc, const float3 * __restrict__ eta,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	updatePrimalVariables_<NearestNeighbor<unsigned char> >(outputU, outputV, tau, mu, p, q, u, u_, v_, abc, eta, inputMask, height, width);
}

template<typename METHOD_MASK, int KERNEL_SIZE>
__device__ void medianFilter_(float *output,
	const float * __restrict__ input,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	static const int NUM_VALUES(KERNEL_SIZE * KERNEL_SIZE); // 9 or 25
	static const int MEDIAN_POSITION(NUM_VALUES/2); // 4 or 12
	static const int HALF_KERNEL_SIZE(KERNEL_SIZE/2); // 1 or 2
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		output[index] = input[index];
		return;
	}

	float valuesSorted[NUM_VALUES] = {};
	for (int j = 0, k = 0; j < KERNEL_SIZE; j++)
	{
		int indexY2 = min(max(indexY + j - HALF_KERNEL_SIZE, 0), height - 1);
		for (int i = 0; i < KERNEL_SIZE; i++, k++)
		{
			int indexX2 = min(max(indexX + i - HALF_KERNEL_SIZE, 0), width - 1);
			valuesSorted[k] = input[indexY2 * width + indexX2];
		}
	}
	float tmpValue = 0.f;
	// only median value is needed so full range sorting is unnecessary
	for (int j = 0; j < (MEDIAN_POSITION + 1); j++)
	{
		for (int i = j + 1; i < NUM_VALUES; i++)
		{
			if (valuesSorted[j] > valuesSorted[i])
			{
				tmpValue = valuesSorted[j];
				valuesSorted[j] = valuesSorted[i];
				valuesSorted[i] = tmpValue;
			}
		}
	}

	output[index] = valuesSorted[MEDIAN_POSITION];
}

extern "C" __global__ void medianFilter(float *output,
	const float * __restrict__ input,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	medianFilter_<NearestNeighbor<unsigned char>, (VFS_MEDIAN_FILTER_KERNEL_SIZE)>(output, input, inputMask, height, width);
}

template<typename METHOD_MASK>
__device__ void computeOpticalFlowWithClamp_(float2 *outputOF, float *outputClamped,
	const float * __restrict__ input,
	const float2 * __restrict__ inputTranslation,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		return;
	}

	// evaluate delta and clamp
	float delta = fminf(input[index] - outputClamped[index], (VFS_UPPER_LIMIT));
	outputClamped[index] += delta;
	// update optical flow by using delta
	outputOF[index].x = fmaf(delta, inputTranslation[index].x, outputOF[index].x);
	outputOF[index].y = fmaf(delta, inputTranslation[index].y, outputOF[index].y);
}

extern "C" __global__ void computeOpticalFlowWithClamp(float2 *outputOF, float *outputClamped,
	const float * __restrict__ input,
	const float2 * __restrict__ inputTranslation,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	// outputOF and outputClamped contain old value and overwritten by using clamped delta
	computeOpticalFlowWithClamp_<NearestNeighbor<unsigned char> >(outputOF, outputClamped, input, inputTranslation, inputMask, height, width);
}

template<typename METHOD, typename VALUE>
__device__ void upSamplingVectorField_(VALUE *output,
	cudaTextureObject_t input,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	output[index] = METHOD::get(input, indexX, indexY, height, width);
}

extern "C" __global__ void upSamplingVectorField(float2 *output,
	cudaTextureObject_t input,
	int height, int width)
{
	upSamplingVectorField_<NearestNeighborScaled<float2>, float2>(output, input, height, width);
}

extern "C" __global__ void upSamplingVectorField1(float *output,
	cudaTextureObject_t input,
	int height, int width)
{
	upSamplingVectorField_<NearestNeighborScaled<float>, float>(output, input, height, width);
}

template<typename METHOD_MASK>
__device__ void triangulateRays_(float3 *output,
	const float3 * __restrict__ inputRaysRef,
	const float3 * __restrict__ inputRaysOther,
	const float3 * __restrict__ inputTranslation,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
	const int indexY = blockIdx.y * blockDim.y + threadIdx.y;
	if ((indexX >= width) || (indexY >= height))
	{
		return;
	}
	const int index = indexY * width + indexX;
	if (METHOD_MASK::get(inputMask, indexX, indexY, height, width) == 0)
	{
		output[index] = make_float3(0, 0, -1);
		return;
	}

	// input rays satisfy epipolar geometry so only X-coordinate is used
	float2 xyRef = make_float2(inputRaysRef[index].x / inputRaysRef[index].z,
		inputRaysRef[index].y / inputRaysRef[index].z);
	float2 xyOther = make_float2(inputRaysOther[index].x / inputRaysOther[index].z,
		inputRaysOther[index].y / inputRaysOther[index].z);
	float ZRef = fmaf(-inputTranslation[0].z, xyOther.x, inputTranslation[0].x) / fmaf(-1, xyRef.x, xyOther.x);
	output[index] = make_float3(ZRef * xyRef.x, ZRef * xyRef.y, ZRef);
}

extern "C" __global__ void triangulateRays(float3 *output,
	const float3 * __restrict__ inputRaysRef,
	const float3 * __restrict__ inputRaysOther,
	const float3 * __restrict__ inputTranslation,
	cudaTextureObject_t inputMask,
	int height, int width)
{
	triangulateRays_<NearestNeighbor<unsigned char> >(output, inputRaysRef, inputRaysOther, inputTranslation, inputMask, height, width);
}
