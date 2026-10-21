from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from data import build_byte_batch_iterator
from runtime import (barrier, cleanup, init_distributed, unwrap_model,
                     JsonlLogger, count_parameters, nats_to_bpb,
                     build_adamw, cosine_lr, set_optimizer_lr)


# Exact Continuum Attention operator. NVIDIA CUDA kernels with FP32 exponential moments.
# This single Python module compiles its embedded CUDA kernel once in the node's
# torch extension cache. No CuPy, Triton, separate .cu file or extra architecture.
import functools

_CUDA_CPP = r'''
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> ca_forward_cuda(torch::Tensor scores, int64_t start);
torch::Tensor ca_backward_cuda(torch::Tensor scores, torch::Tensor weights,
    torch::Tensor stats, torch::Tensor grad_weights, torch::Tensor means, int64_t start);
std::vector<torch::Tensor> ca_backward_fused_cuda(torch::Tensor scores,
    torch::Tensor grad_weights, int64_t start);
torch::Tensor ca_direct_output_cuda(torch::Tensor scores, torch::Tensor values, int64_t start);
torch::Tensor ca_forward_half_cuda(torch::Tensor scores, int64_t start);
std::vector<torch::Tensor> ca_backward_half_cuda(torch::Tensor scores,
    torch::Tensor grad_weights, torch::Tensor scale_per_head, int64_t start);
std::vector<torch::Tensor> ca_backward_half_gw16_cuda(torch::Tensor scores,
    torch::Tensor grad_weights_half, torch::Tensor scale_per_head, int64_t start);
torch::Tensor ca_scale_scores_half_cuda(torch::Tensor raw_half, torch::Tensor scale_per_head);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_fused", &ca_backward_fused_cuda, "Combined exact weights and analytic score VJP");
    m.def("forward_half", &ca_forward_half_cuda, "Exact CA weights written directly in FP16");
    m.def("scale_scores_half", &ca_scale_scores_half_cuda,
        "Fuse half QK score cast with per-head FP32 gain scaling");
    m.def("backward_half_gw16", &ca_backward_half_gw16_cuda,
        "Analytic CA VJP reads FP16 dYVT directly instead of copying it to FP32");
    m.def("backward_half", &ca_backward_half_cuda,
        "Exact analytical score VJP with fused FP16 output, gain reduction, and scaling");
    m.def("direct_output", &ca_direct_output_cuda, "Fused exact segment integration and value accumulation");
    m.def("forward", &ca_forward_cuda, "CA segment-integrated weight matrix");
    m.def("backward", &ca_backward_cuda, "CA analytic score VJP");
}
'''

_CUDA_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <vector>

// All loops below are parallel CUDA tile/reduction computations.
// NO recurrent state, numerical quadrature or iterative integral solver.

// v6: replace TWO whole-matrix PyTorch ops per query tile
//  raw_half.float() and (FP32_raw * per_head_scale)
// with ONE scale-and-cast CUDA pass. QK GEMM and CA score definition unchanged.
__global__ void ca_scale_scores_half_kernel(
    const __half* __restrict__ raw_half,
    const float* __restrict__ scale_per_head,
    float* __restrict__ scores,
    int Q,int K,int H,int total) {
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=total)return;
    int head=(id/(Q*K))%H;
    scores[id]=__half2float(raw_half[id])*scale_per_head[head];
}

__device__ __forceinline__ float ca_gw_float(float x){return x;}
__device__ __forceinline__ float ca_gw_float(__half x){return __half2float(x);}

__device__ __forceinline__ float seg_L(float t) {
    if (t < 0.1f) {
        float t2=t*t,t3=t2*t,t4=t2*t2,t5=t4*t,t6=t3*t3;
        return 1.f-0.5f*t+t2/6.f-t3/24.f+t4/120.f-t5/720.f+t6/5040.f;
    }
    return -expm1f(-t)/t;
}
__device__ __forceinline__ float seg_mu(float d) {
    float t=fabsf(d);
    if (t < 0.1f) {
        float d2=d*d;
        return 0.5f+d/12.f-d*d2/720.f+d2*d2*d/30240.f;
    }
    float p=1.f/(-expm1f(-t))-1.f/t;
    return d>=0.f ? p : 1.f-p;
}
__device__ __forceinline__ float seg_mass(float a,float b,float m){
    return expf(fmaxf(a,b)-m)*seg_L(fabsf(b-a));
}

__device__ __forceinline__ float block_max(float v,float* scratch){
    int tid=threadIdx.x;
    scratch[tid]=v;
    __syncthreads();
    for(int step=128;step>=1;step>>=1){
        if(tid<step) scratch[tid]=fmaxf(scratch[tid],scratch[tid+step]);
        __syncthreads();
    }
    return scratch[0];
}
__device__ __forceinline__ float block_sum(float v,float* scratch){
    int tid=threadIdx.x;
    scratch[tid]=v;
    __syncthreads();
    for(int step=128;step>=1;step>>=1){
        if(tid<step) scratch[tid]+=scratch[tid+step];
        __syncthreads();
    }
    return scratch[0];
}

__global__ void ca_weights_kernel(const float* __restrict__ scores,
                                   float* __restrict__ weights,
                                   float* __restrict__ stats,
                                   int T,int Q,int first_query){
    const int row=blockIdx.x, lane=threadIdx.x;
    const int query=first_query+row%Q;
    const int offset=row*T;
    __shared__ float scratch[256];
    float localmax=-INFINITY;
    for(int j=lane;j<=query;j+=256) localmax=fmaxf(localmax,scores[offset+j]);
    const float m=block_max(localmax,scratch);
    float localZ=0.f;
    for(int j=lane;j<T;j+=256){
        float w=0.f;
        if(j<=query){
            float s=scores[offset+j];
            if(j==0) w+=0.5f*expf(s-m);
            if(j==query) w+=0.5f*expf(s-m);
            if(j>0){float a=scores[offset+j-1];w+=seg_mass(a,s,m)*seg_mu(s-a);}
            if(j<query){float b=scores[offset+j+1];w+=seg_mass(s,b,m)*(1.f-seg_mu(b-s));}
        }
        weights[offset+j]=w;
        localZ+=w;
    }
    const float Z=block_sum(localZ,scratch);
    if(lane==0){stats[2*row]=m;stats[2*row+1]=1.f/Z;}
    for(int j=lane;j<T;j+=256) weights[offset+j]*=1.f/Z;
}

struct Moments{float m0,m1,m2;};
__device__ __forceinline__ Moments moments(float delta){
    float t=fabsf(delta), a,b,c;
    if(t<1.f){
        // The Taylor expansion represents the *analytical* exponential moments.
        // Small-value cancellation avoided. Closed-form values used elsewhere.
        a=1.f+t*(-0.5f+t*(1.f/6.f+t*(-1.f/24.f+t*(1.f/120.f+
          t*(-1.f/720.f+t*(1.f/5040.f+t*(-1.f/40320.f+t*(1.f/362880.f+t*(-1.f/3628800.f)))))))));
        b=0.5f+t*(-1.f/3.f+t*(1.f/8.f+t*(-1.f/30.f+t*(1.f/144.f+
          t*(-1.f/840.f+t*(1.f/5760.f+t*(-1.f/45360.f+t*(1.f/403200.f+t*(-1.f/3991680.f)))))))));
        c=1.f/3.f+t*(-1.f/4.f+t*(1.f/10.f+t*(-1.f/36.f+t*(1.f/168.f+
          t*(-1.f/960.f+t*(1.f/6480.f+t*(-1.f/50400.f+t*(1.f/443520.f+t*(-1.f/4536000.f)))))))));
        if(delta>0.f) { b=a-b; c=a-2.f*(a-b)+c; }
    }else{
        float x=expf(-t),inv=1.f/t,inv2=inv*inv;
        a=(1.f-x)*inv;
        if(delta>=0.f){
           b=(t-1.f+x)*inv2;
           c=(t*t-2.f*t+2.f-2.f*x)*inv2*inv;
        }else{
           b=(1.f-(t+1.f)*x)*inv2;
           c=(2.f-(t*t+2.f*t+2.f)*x)*inv2*inv;
        }
    }
    return {a,b,c};
}

__global__ void ca_scores_backward_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ grad_weights,
    const float* __restrict__ stats,
    const float* __restrict__ means,
    float* __restrict__ ds,
    int T,int Q,int first_query,int total){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=total) return;
    int row=id/T, col=id%T;
    int query=first_query+row%Q;
    if(col>query){ds[id]=0.f;return;}
    int off=row*T;
    float s=scores[id],m=stats[2*row],invZ=stats[2*row+1];
    float g=grad_weights[id]-means[row];
    float res=0.f;
    if(col==0)res+=0.5f*expf(s-m)*g;
    if(col==query)res+=0.5f*expf(s-m)*g;
    if(col>0){
        float a=scores[id-1];Moments J=moments(s-a);
        float e=expf(fmaxf(a,s)-m);
        float gl=grad_weights[id-1]-means[row];
        res+=e*(gl*(J.m1-J.m2)+g*J.m2);
    }
    if(col<query){
        float b=scores[id+1];Moments J=moments(b-s);
        float e=expf(fmaxf(b,s)-m);
        float gr=grad_weights[id+1]-means[row];
        res+=e*(g*(J.m0-2.f*J.m1+J.m2)+gr*(J.m1-J.m2));
    }
    ds[id]=res*invZ;
}


// Fused analytic backward: obtains score normalization and the weighted
// mean of dL/dw in the same CTA, then directly evaluates the exact J0/J1/J2
// derivatives. This removes the independent weight normalization + means
// normalization and score derivative kernel launches.
__global__ void ca_backward_fused_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ gw,
    float* __restrict__ weights,
    float* __restrict__ ds,
    int K,int Q,int first_query) {
    const int row=blockIdx.x, lane=threadIdx.x;
    const int query=first_query+row%Q;
    const int off=row*K;
    __shared__ float scratch[256];
    __shared__ float s_m,s_invz,s_mean;
    float localmax=-INFINITY;
    for(int j=lane;j<=query;j+=256)
        localmax=fmaxf(localmax,scores[off+j]);
    float maxval=block_max(localmax,scratch);
    if(lane==0)s_m=maxval;
    __syncthreads();
    float localz=0.f, localmean=0.f;
    for(int j=lane;j<=query;j+=256){
        float a=scores[off+j], w=0.f;
        if(j==0) w+=0.5f*expf(a-s_m);
        if(j==query) w+=0.5f*expf(a-s_m);
        if(j>0) {
            float left=scores[off+j-1];
            float d=a-left;
            w+=seg_mass(left,a,s_m)*seg_mu(d);
        }
        if(j<query){
            float right=scores[off+j+1], d=right-a;
            w+=seg_mass(a,right,s_m)*(1.f-seg_mu(d));
        }
        weights[off+j]=w;
        localz+=w;
        localmean+=w*gw[off+j];
    }
    float Z=block_sum(localz,scratch);
    if(lane==0)s_invz=1.f/Z;
    __syncthreads();
    float totalGW=block_sum(localmean,scratch);
    if(lane==0)s_mean=totalGW*s_invz;
    __syncthreads();
    for(int j=lane;j<K;j+=256){
        if(j>query){weights[off+j]=0.f;ds[off+j]=0.f;continue;}
        weights[off+j]*=s_invz;
        float sj=scores[off+j], g=gw[off+j]-s_mean, res=0.f;
        if(j==0)res+=0.5f*expf(sj-s_m)*g;
        if(j==query)res+=0.5f*expf(sj-s_m)*g;
        if(j>0){
            float sa=scores[off+j-1];
            Moments J=moments(sj-sa);
            float e=expf(fmaxf(sa,sj)-s_m);
            float gl=gw[off+j-1]-s_mean;
            res+=e*(gl*(J.m1-J.m2)+g*J.m2);
        }
        if(j<query){
            float sb=scores[off+j+1];
            Moments J=moments(sb-sj);
            float e=expf(fmaxf(sb,sj)-s_m);
            float gr=gw[off+j+1]-s_mean;
            res+=e*(g*(J.m0-2.f*J.m1+J.m2)+gr*(J.m1-J.m2));
        }
        ds[off+j]=res*s_invz;
    }
}


// v5: preserve FP32 score normalization and exact segment moments. Return
// normalized half weights directly, skipping a float-to-half torch copy per
// query tile; identical conversion as prefix w.to(torch.float16).
// The shared row buffer holds unnormalized FP32 CA weights.
__global__ void ca_forward_half_kernel(
    const float* __restrict__ scores, __half* __restrict__ out,
    int K,int Q,int first_query) {
    const int row=blockIdx.x,lane=threadIdx.x;
    const int query=first_query+row%Q;
    const int off=row*K;
    __shared__ float scratch[256], wcache[2048];
    float localmax=-INFINITY;
    for(int j=lane;j<=query;j+=256)
        localmax=fmaxf(localmax,scores[off+j]);
    float m=block_max(localmax,scratch);
    float localz=0.f;
    for(int j=lane;j<K;j+=256){
        float w=0.f;
        if(j<=query){
            float a=scores[off+j];
            if(j==0) w+=0.5f*expf(a-m);
            if(j==query)w+=0.5f*expf(a-m);
            if(j>0){float left=scores[off+j-1];w+=seg_mass(left,a,m)*seg_mu(a-left);}
            if(j<query){float right=scores[off+j+1];w+=seg_mass(a,right,m)*(1.f-seg_mu(right-a));}
        }
        wcache[j]=w;
        localz+=w;
    }
    float Z=block_sum(localz,scratch);
    float invz=1.f/Z;
    for(int j=lane;j<K;j+=256)
        out[off+j]=__float2half_rn(wcache[j]*invz);
}

// The fused analytical kernel returns three quantities, using the same
// fused CA VJP equations:
//   W_half: normalized weights rounded once to FP16 for dV GEMM.
//   dS_scaled_half: (dL/dS * gain / sqrt(D)) rounded once for dQ/dK GEMMs.
//   gain_partial: sum_j dL/dS_ij * S_ij, one FP32 number per query/head.
// This removes separate torch mul, half conversions, and the [Q,K] gain VJP
// reduction. No change to the CA integrals or analytical derivative.
template <typename GWType>
__global__ void ca_backward_half_kernel(
    const float* __restrict__ scores,
    const GWType* __restrict__ gw,
    const float* __restrict__ scale_per_head,
    __half* __restrict__ weights,
    __half* __restrict__ ds_scaled,
    float* __restrict__ gain_partial,
    int K,int Q,int H,int first_query) {
    const int row=blockIdx.x,lane=threadIdx.x;
    const int query=first_query+row%Q;
    const int off=row*K;
    const float scale=scale_per_head[(row/Q)%H];
    __shared__ float scratch[256], wcache[2048];
    __shared__ float s_m,s_invz,s_mean;
    float localmax=-INFINITY;
    for(int j=lane;j<=query;j+=256)
        localmax=fmaxf(localmax,scores[off+j]);
    float maxval=block_max(localmax,scratch);
    if(lane==0)s_m=maxval;
    __syncthreads();
    float localz=0.f,localmean=0.f;
    for(int j=lane;j<K;j+=256){
        float w=0.f;
        if(j<=query){
            float a=scores[off+j];
            if(j==0)w+=0.5f*expf(a-s_m);
            if(j==query)w+=0.5f*expf(a-s_m);
            if(j>0){float left=scores[off+j-1];w+=seg_mass(left,a,s_m)*seg_mu(a-left);}
            if(j<query){float right=scores[off+j+1];w+=seg_mass(a,right,s_m)*(1.f-seg_mu(right-a));}
        }
        wcache[j]=w;
        localz+=w;
        if(j<=query) localmean+=w*ca_gw_float(gw[off+j]);
    }
    float Z=block_sum(localz,scratch);
    if(lane==0)s_invz=1.f/Z;
    __syncthreads();
    float totalGW=block_sum(localmean,scratch);
    if(lane==0)s_mean=totalGW*s_invz;
    __syncthreads();
    float rho_sum=0.f;
    for(int j=lane;j<K;j+=256){
        weights[off+j]=__float2half_rn(wcache[j]*s_invz);
        if(j>query){ds_scaled[off+j]=__float2half_rn(0.f);continue;}
        float sj=scores[off+j],g=ca_gw_float(gw[off+j])-s_mean,res=0.f;
        if(j==0)res+=0.5f*expf(sj-s_m)*g;
        if(j==query)res+=0.5f*expf(sj-s_m)*g;
        if(j>0){
            float sa=scores[off+j-1];
            Moments J=moments(sj-sa);
            float e=expf(fmaxf(sa,sj)-s_m);
            float gl=ca_gw_float(gw[off+j-1])-s_mean;
            res+=e*(gl*(J.m1-J.m2)+g*J.m2);
        }
        if(j<query){
            float sb=scores[off+j+1];
            Moments J=moments(sb-sj);
            float e=expf(fmaxf(sb,sj)-s_m);
            float gr=ca_gw_float(gw[off+j+1])-s_mean;
            res+=e*(g*(J.m0-2.f*J.m1+J.m2)+gr*(J.m1-J.m2));
        }
        float ds=res*s_invz;
        ds_scaled[off+j]=__float2half_rn(ds*scale);
        rho_sum+=ds*sj;
    }
    float sum=block_sum(rho_sum,scratch);
    if(lane==0) gain_partial[row]=sum;
}

// Optional forward fusion. Each CTA owns a complete query row and keeps
// unnormalized exact CA weights in shared memory rather than writing an
// attention matrix to global memory. The same CTA accumulates the result.
// A tiled GEMM may be faster on Volta, so this path is benchmark-selected.
// K and D bounded to control shared memory and resource usage on sm_70.
template<typename scalar_t>
__global__ void ca_direct_output_kernel(
    const float* __restrict__ scores,
    const scalar_t* __restrict__ values,
    scalar_t* __restrict__ out,
    int K,int Q,int D,int first_query) {
    const int row=blockIdx.x, lane=threadIdx.x;
    const int query=first_query+row%Q;
    const int off=row*K;
    __shared__ float wshared[2048];
    __shared__ float scratch[256];
    __shared__ float s_m,s_invz;
    float localmax=-INFINITY;
    for(int j=lane;j<=query;j+=256)
        localmax=fmaxf(localmax,scores[off+j]);
    float m=block_max(localmax,scratch);
    if(lane==0)s_m=m;
    __syncthreads();
    float localz=0.f;
    for(int j=lane;j<=query;j+=256){
        float a=scores[off+j],w=0.f;
        if(j==0)w+=0.5f*expf(a-s_m);
        if(j==query)w+=0.5f*expf(a-s_m);
        if(j>0){
            float left=scores[off+j-1];
            w+=seg_mass(left,a,s_m)*seg_mu(a-left);
        }
        if(j<query){
            float right=scores[off+j+1];
            w+=seg_mass(a,right,s_m)*(1.f-seg_mu(right-a));
        }
        wshared[j]=w;
        localz+=w;
    }
    float Z=block_sum(localz,scratch);
    if(lane==0)s_invz=1.f/Z;
    __syncthreads();
    const scalar_t* v=values+(row/Q)*K*D;
    for(int d=lane;d<D;d+=256){
        float acc=0.f;
        for(int j=0;j<=query;j++){
            acc+=wshared[j]*static_cast<float>(v[j*D+d]);
        }
        out[row*D+d]=static_cast<scalar_t>(acc*s_invz);
    }
}


torch::Tensor ca_forward_half_cuda(torch::Tensor scores,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
        "v5 forward requires contiguous CUDA FP32 scores");
    TORCH_CHECK(scores.dim()==3,"scores must have shape [BH,Q,K]");
    int K=scores.size(2),Q=scores.size(1),rows=scores.size(0)*Q;
    TORCH_CHECK(K<=2048 && start>=0 && start+Q<=K,"v5 invalid causal prefix");
    auto out=torch::empty(scores.sizes(),scores.options().dtype(torch::kFloat16));
    ca_forward_half_kernel<<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(),reinterpret_cast<__half*>(out.data_ptr<at::Half>()),K,Q,int(start));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
torch::Tensor ca_scale_scores_half_cuda(torch::Tensor raw,torch::Tensor scale){
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type()==torch::kFloat16 && raw.is_contiguous() && raw.dim()==3,
        "v6 QK scores require contiguous FP16 [BH,Q,K] CUDA tensor");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type()==torch::kFloat32 &&
        scale.is_contiguous() && scale.dim()==1,"v6 gain scale requires FP32 [H]");
    int BH=raw.size(0),Q=raw.size(1),K=raw.size(2),H=scale.numel();
    TORCH_CHECK(BH%H==0 && K<=2048,"v6 invalid score or head shape");
    auto s=torch::empty(raw.sizes(),raw.options().dtype(torch::kFloat32));
    int total=raw.numel();
    ca_scale_scores_half_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(raw.data_ptr<at::Half>()),
        scale.data_ptr<float>(),s.data_ptr<float>(),Q,K,H,total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return s;
}
std::vector<torch::Tensor> ca_backward_half_gw16_cuda(
    torch::Tensor scores,torch::Tensor grad_weights,torch::Tensor scale_per_head,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 &&
        scores.is_contiguous() && scores.dim()==3,"v6 scores must be contiguous FP32 [BH,Q,K]");
    TORCH_CHECK(grad_weights.is_cuda() && grad_weights.scalar_type()==torch::kFloat16 &&
        grad_weights.is_contiguous() && grad_weights.sizes()==scores.sizes(),
        "v6 GW must be contiguous FP16 [BH,Q,K]");
    TORCH_CHECK(scale_per_head.is_cuda() && scale_per_head.scalar_type()==torch::kFloat32 &&
        scale_per_head.is_contiguous() && scale_per_head.dim()==1,"v6 gain scale must be FP32 [H]");
    int K=scores.size(2),Q=scores.size(1),BH=scores.size(0),H=scale_per_head.numel(),rows=BH*Q;
    TORCH_CHECK(K<=2048 && start>=0 && start+Q<=K && BH%H==0,"v6 invalid causal prefix");
    auto opts_half=scores.options().dtype(torch::kFloat16);
    auto w=torch::empty(scores.sizes(),opts_half);
    auto ds=torch::empty(scores.sizes(),opts_half);
    auto grow=torch::empty({BH,Q},scores.options());
    ca_backward_half_kernel<__half><<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(),reinterpret_cast<const __half*>(grad_weights.data_ptr<at::Half>()),
        scale_per_head.data_ptr<float>(),
        reinterpret_cast<__half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(ds.data_ptr<at::Half>()),
        grow.data_ptr<float>(),K,Q,H,int(start));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w,ds,grow};
}

std::vector<torch::Tensor> ca_backward_half_cuda(
    torch::Tensor scores,torch::Tensor grad_weights,
    torch::Tensor scale_per_head,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
        "v5 backward requires contiguous CUDA FP32 scores");
    TORCH_CHECK(scores.dim()==3 && grad_weights.sizes()==scores.sizes() &&
        grad_weights.is_cuda() && grad_weights.is_contiguous() &&
        grad_weights.scalar_type()==torch::kFloat32,"v5 grad_weights mismatch");
    TORCH_CHECK(scale_per_head.is_cuda() && scale_per_head.scalar_type()==torch::kFloat32 &&
        scale_per_head.is_contiguous() && scale_per_head.dim()==1,"v5 scale mismatch");
    int K=scores.size(2),Q=scores.size(1),BH=scores.size(0);
    int H=scale_per_head.numel(),rows=BH*Q;
    TORCH_CHECK(K<=2048 && start>=0 && start+Q<=K && BH%H==0,"v5 invalid causal prefix");
    auto opts_half=scores.options().dtype(torch::kFloat16);
    auto w=torch::empty(scores.sizes(),opts_half);
    auto ds=torch::empty(scores.sizes(),opts_half);
    auto grow=torch::empty({BH,Q},scores.options());
    ca_backward_half_kernel<float><<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(),grad_weights.data_ptr<float>(),scale_per_head.data_ptr<float>(),
        reinterpret_cast<__half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(ds.data_ptr<at::Half>()),
        grow.data_ptr<float>(),K,Q,H,int(start));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w,ds,grow};
}

std::vector<torch::Tensor> ca_backward_fused_cuda(
    torch::Tensor scores,torch::Tensor grad_weights,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
        "Fused CA requires contiguous float32 scores");
    TORCH_CHECK(grad_weights.is_cuda() && grad_weights.is_contiguous() &&
        grad_weights.scalar_type()==torch::kFloat32 &&
        grad_weights.sizes()==scores.sizes(),"CA VJP input mismatch");
    int Q=scores.size(1),K=scores.size(2);
    TORCH_CHECK(start>=0 && start+Q<=K,"Invalid causal query prefix");
    int rows=scores.size(0)*Q;
    auto w=torch::empty_like(scores),ds=torch::empty_like(scores);
    ca_backward_fused_kernel<<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(),grad_weights.data_ptr<float>(),
        w.data_ptr<float>(),ds.data_ptr<float>(),K,Q,(int)start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w,ds};
}

torch::Tensor ca_direct_output_cuda(torch::Tensor scores,torch::Tensor values,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
        "Direct CA output requires contiguous float32 scores");
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() && values.dim()==3 &&
        values.size(0)==scores.size(0) && values.size(1)==scores.size(2),
        "CA fused value tensor must have shape [BH, K, D]");
    TORCH_CHECK(values.scalar_type()==torch::kFloat16 || values.scalar_type()==torch::kFloat32,
        "CA fused output supports FP16 and FP32 values");
    int BH=scores.size(0),Q=scores.size(1),K=scores.size(2),D=values.size(2);
    TORCH_CHECK(start>=0 && start+Q<=K,"Invalid causal query prefix");
    TORCH_CHECK(K<=2048 && D<=128,"Direct CUDA path supports K<=2048, D<=128");
    auto out=torch::empty({BH,Q,D},values.options());
    int rows=BH*Q;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(),"ca_direct_output",[&]{
        ca_direct_output_kernel<scalar_t><<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
            scores.data_ptr<float>(),values.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),
            K,Q,D,(int)start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> ca_forward_cuda(torch::Tensor scores,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
       "CUDA CA requires contiguous float32 scores");
    TORCH_CHECK(scores.dim()==3,"Expected flattened scores [batch*heads,Q,T]");
    int rows=scores.size(0)*scores.size(1),Q=scores.size(1),T=scores.size(2);
    TORCH_CHECK(start>=0 && start+Q<=T,"Invalid CA causal query tile");
    auto w=torch::empty_like(scores);
    auto stats=torch::empty({rows,2},scores.options());
    ca_weights_kernel<<<rows,256,0,at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(),w.data_ptr<float>(),stats.data_ptr<float>(),T,Q,(int)start);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {w,stats};
}

torch::Tensor ca_backward_cuda(torch::Tensor scores,torch::Tensor weights,
    torch::Tensor stats,torch::Tensor grad_weights,torch::Tensor means,int64_t start){
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.is_contiguous(),
       "CUDA CA backward requires contiguous float32 scores");
    TORCH_CHECK(weights.sizes()==scores.sizes() && grad_weights.sizes()==scores.sizes(),"CA VJP shape mismatch");
    int Q=scores.size(1),T=scores.size(2),total=scores.numel();
    TORCH_CHECK(start>=0 && start+Q<=T,"Invalid CA causal query tile");
    auto ds=torch::empty_like(scores);
    ca_scores_backward_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
         scores.data_ptr<float>(),grad_weights.data_ptr<float>(),stats.data_ptr<float>(),
         means.data_ptr<float>(),ds.data_ptr<float>(),T,Q,(int)start,total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return ds;
}
'''

@functools.lru_cache(maxsize=1)
def _ca_cuda_extension():
    import os
    from torch.utils.cpp_extension import load_inline, CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError('Compiling the CUDA extension requires a compatible CUDA toolkit (nvcc) and PyTorch build.')
    return load_inline(
        name='continuum_attention_cuda_v6_kernel',
        cpp_sources=_CUDA_CPP, cuda_sources=_CUDA_SRC,
        functions=None, with_cuda=True,
        extra_cflags=['-O2'], extra_cuda_cflags=['-O3'],
        verbose=os.environ.get('CA_CUDA_VERBOSE','0')=='1',
    )

def _ca_reference_weights(scores: torch.Tensor, start: int) -> torch.Tensor:
    # Numerically stable torch reference and CPU test oracle, including caps.
    B,Q,T=scores.shape
    queries=torch.arange(start,start+Q,device=scores.device).view(1,Q,1)
    keys=torch.arange(T,device=scores.device).view(1,1,T)
    mask=keys<=queries
    m=scores.masked_fill(~mask,float('-inf')).amax(-1,keepdim=True)
    valid=keys[...,1:]<=queries
    a=torch.where(valid,scores[...,:-1],m)
    b=torch.where(valid,scores[...,1:],m)
    d=b-a
    t=d.abs()
    t2=t*t
    Lsmall=1-.5*t+t2/6-t*t2/24+t2*t2/120-t*t2*t2/720+t2*t2*t2/5040
    safe_t=t.clamp_min(.1)
    L=torch.where(t<.1,Lsmall,-torch.expm1(-safe_t)/safe_t)
    musmall=.5+d/12-d.pow(3)/720+d.pow(5)/30240
    mup=1/(-torch.expm1(-safe_t))-1/safe_t
    mu=torch.where(t<.1,musmall,torch.where(d>=0,mup,1-mup))
    r=torch.exp(torch.maximum(a,b)-m)*L*valid
    left=F.pad(r*(1-mu),(0,1))
    right=F.pad(r*mu,(1,0))
    c0=F.pad(.5*torch.exp(scores[...,:1]-m),(0,T-1))
    ci=F.one_hot(torch.arange(start,start+Q,device=scores.device),num_classes=T).to(scores.dtype)
    end=.5*torch.exp(scores.gather(-1,queries.expand(B,-1,1))-m)*ci
    weights=(left+right+c0+end)*mask
    return weights/weights.sum(-1,keepdim=True)

class _CAExactTiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,log_gain,query_tile):
        if q.dtype!=k.dtype or q.dtype!=v.dtype:
            raise ValueError('CA Q,K,V must share dtype')
        B,H,T,D=q.shape
        gain=log_gain.detach().exp().float().view(1,H,1,1)
        scale=gain*(D**-0.5)
        ext=_ca_cuda_extension() if q.is_cuda else None
        mode=os.environ.get('CA_CUDA_MODE','prefix').lower()
        if mode not in ('dense','prefix','fused','prefix_v5','prefix_v6'):
            raise ValueError('CA_CUDA_MODE must be dense, prefix, fused, prefix_v5, or prefix_v6')
        if mode in ('prefix_v5','prefix_v6') and (not q.is_cuda or q.dtype!=torch.float16 or T>2048):
            raise ValueError('CA v5/v6 paths require CUDA FP16 Q/K/V and T<=2048')
        if mode=='fused' and (T>2048 or D>128 or v.dtype not in (torch.float16,torch.float32)):
            raise ValueError('CA fused forward supports T<=2048, D<=128, FP16/FP32 values')
        qb=q.reshape(B*H,T,D)
        kb=k.reshape(B*H,T,D)
        vb=v.reshape(B*H,T,D)
        out=[]
        for start in range(0,T,query_tile):
            end=min(T,start+query_tile)
            # Exact causal skip: no future key/value is included in any GEMM.
            # Dense score computation is retained for comparison tests.
            K=T if mode=='dense' else end
            qs=qb[:,start:end,:]
            keys=kb[:,:K,:]
            raw=torch.bmm(qs,keys.transpose(1,2))
            if mode=='prefix_v6':
                s=ext.scale_scores_half(raw.contiguous(),scale.reshape(H).contiguous())
            else:
                s=(raw.float().reshape(B,H,-1,K)*scale).reshape(B*H,-1,K).contiguous()
            if ext is not None and mode in ('prefix_v5','prefix_v6'):
                w=ext.forward_half(s,start)
                y=torch.bmm(w,vb[:,:K,:])
            elif ext is not None and mode=='fused':
                y=ext.direct_output(s,vb[:,:K,:].contiguous(),start)
            else:
                w=ext.forward(s,start)[0] if ext is not None else _ca_reference_weights(s,start)
                y=torch.bmm(w.to(v.dtype),vb[:,:K,:])
            out.append(y)
        ctx.save_for_backward(q,k,v,log_gain)
        ctx.query_tile=int(query_tile)
        ctx.mode=mode
        return torch.cat(out,dim=1).reshape(B,H,T,D)

    @staticmethod
    def backward(ctx,grad_y):
        if torch.is_grad_enabled():
            raise RuntimeError('CA Volta analytic kernel supports first-order gradients only')
        q,k,v,log_gain=ctx.saved_tensors
        B,H,T,D=q.shape
        gain=log_gain.detach().exp().float().view(1,H,1,1)
        scale=gain*(D**-0.5)
        ext=_ca_cuda_extension() if q.is_cuda else None
        qb=q.reshape(B*H,T,D)
        kb=k.reshape(B*H,T,D)
        vb=v.reshape(B*H,T,D)
        gb=grad_y.contiguous().reshape(B*H,T,D)
        gq=torch.zeros_like(q,dtype=torch.float32).reshape(B*H,T,D)
        gk=torch.zeros_like(k,dtype=torch.float32).reshape(B*H,T,D)
        gv=torch.zeros_like(v,dtype=torch.float32).reshape(B*H,T,D)
        gr=torch.zeros_like(log_gain,dtype=torch.float32)
        for start in range(0,T,ctx.query_tile):
            end=min(T,start+ctx.query_tile)
            K=T if ctx.mode=='dense' else end
            qs=qb[:,start:end,:]
            keys=kb[:,:K,:]
            vals=vb[:,:K,:]
            gys=gb[:,start:end,:]
            raw=torch.bmm(qs,keys.transpose(1,2))
            if ctx.mode=='prefix_v6':
                s=ext.scale_scores_half(raw.contiguous(),scale.reshape(H).contiguous())
            else:
                s=(raw.float().reshape(B,H,-1,K)*scale).reshape(B*H,-1,K).contiguous()
            # dW = dY V^T. Fused VJP generates W and dS together without
            # materialized norm stats or an additional mean reduction kernel.
            if ext is not None and ctx.mode=='prefix_v6':
                gw_half=torch.bmm(gys,vals.transpose(1,2))
                w,ds_scaled,gain_rows=ext.backward_half_gw16(
                    s,gw_half.contiguous(),scale.reshape(H).contiguous(),start)
            elif ext is not None and ctx.mode=='prefix_v5':
                gw=torch.bmm(gys,vals.transpose(1,2)).float().contiguous()
                w,ds_scaled,gain_rows=ext.backward_half(s,gw,scale.reshape(H).contiguous(),start)
            elif ext is not None and ctx.mode != 'dense':
                gw=torch.bmm(gys,vals.transpose(1,2)).float().contiguous()
                w,ds=ext.backward_fused(s,gw,start)
            else:
                if ext is not None:
                    w,stats=ext.forward(s,start)
                else:
                    w=_ca_reference_weights(s,start)
                    stats=None
                gw=torch.bmm(gys,vals.transpose(1,2)).float().contiguous()
                if ext is not None:
                    means=(w*gw).sum(-1).contiguous().reshape(-1)
                    ds=ext.backward(s,w,stats,gw,means,start)
                else:
                    with torch.enable_grad():
                        sr=s.detach().requires_grad_(True)
                        wr=_ca_reference_weights(sr,start)
                        ds=torch.autograd.grad(wr,sr,gw)[0]
            # dV = W^T dY. Use cublas Tensor Cores for this dense contraction.
            if ctx.mode in ('prefix_v5','prefix_v6'):
                gv[:,:K,:].add_(torch.bmm(w.transpose(1,2),gys).float())
                gq[:,start:end,:].add_(torch.bmm(ds_scaled,keys).float())
                gk[:,:K,:].add_(torch.bmm(ds_scaled.transpose(1,2),qs).float())
                gr.add_(gain_rows.reshape(B,H,-1).sum(dim=(0,2)))
            else:
                gv[:,:K,:].add_(torch.bmm(w.to(v.dtype).transpose(1,2),gys).float())
                ds4=ds.reshape(B,H,-1,K)
                scaled=(ds4*scale).reshape(B*H,-1,K)
                gq[:,start:end,:].add_(torch.bmm(scaled.to(q.dtype),keys).float())
                gk[:,:K,:].add_(torch.bmm(scaled.to(k.dtype).transpose(1,2),qs).float())
                gr.add_((ds4*s.reshape(B,H,-1,K)).sum(dim=(0,2,3)))
        return (gq.reshape_as(q).to(q.dtype),gk.reshape_as(k).to(k.dtype),
                gv.reshape_as(v).to(v.dtype),gr.to(log_gain.dtype),None)

def continuum_attention(q,k,v,log_gain,query_tile=64):
    """Exact causal Continuum Attention, with efficient causal-prefix contractions.

    CA_CUDA_MODE='dense': dense all-key reference path.
    CA_CUDA_MODE='prefix': causal-prefix GEMMs + fused analytical backward.
    CA_CUDA_MODE='prefix_v6': optimized FP16 score scaling and analytical backward.
    CA_CUDA_MODE='prefix_v5': causal-prefix implementation with FP16 conversions,
    scaled score VJP, and head-gain gradient partial reduction.
    CA_CUDA_MODE='fused': additionally fuse exact forward integration and value
    accumulation into one CUDA kernel, with no dense attention weight write.

    All modes have identical integration, half token caps, trainable gains,
    causal visibility, gradients, and parameter topology. They may differ at
    ordinary FP16/FP32 rounding precision due to execution order.
    """
    return _CAExactTiled.apply(q.contiguous(),k.contiguous(),v.contiguous(),log_gain,int(query_tile))
@dataclass
class ContinuumDecoderConfig:
    """A NORMAL decoder-only byte Transformer with exact Continuum Attention.

    Only the attention operator differs from standard causal softmax:
        integral exp(s_i(t)) V(t) dt / integral exp(s_i(t)) dt,
    where K(t),V(t) linearly interpolate adjacent causal positions and
    the first/last half-token caps have constant endpoints.
    """
    vocab_size: int = 256
    block_size: int = 2048
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    eps: float = 1e-6
    rope_theta: float = 10000.0
    grad_checkpointing: bool = False
    ca_query_tile: int = 64

    def __post_init__(self):
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if (self.d_model // self.n_heads) % 2:
            raise ValueError('RoPE needs even head dimension')
        if self.n_layers<=0 or self.block_size<2 or self.ca_query_tile<=0:
            raise ValueError('Invalid layer, context or CA query tile count')
        if self.dropout != 0:
            raise ValueError('The CA operator has no attention dropout; use --dropout 0.0')


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim))
        self.eps=eps
    def forward(self,x):
        return self.weight*x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.eps)


def apply_rope(q,k,theta):
    """Rotary position embeddings applied independently at each position."""
    T=q.shape[-2]; D=q.shape[-1]
    freq=1.0/(theta**(torch.arange(0,D,2,device=q.device,dtype=torch.float32)/D))
    pos=torch.arange(T,device=q.device,dtype=torch.float32)
    phase=torch.outer(pos,freq)
    cos=torch.repeat_interleave(phase.cos(),2,-1).to(q.dtype)[None,None,:,:]
    sin=torch.repeat_interleave(phase.sin(),2,-1).to(q.dtype)[None,None,:,:]
    def rotate(t):
        return torch.stack((-t[...,1::2],t[...,0::2]),-1).flatten(-2)
    return q*cos+rotate(q)*sin, k*cos+rotate(k)*sin


class ContinuumSelfAttention(nn.Module):
    def __init__(self, dim,heads,rope_theta,ca_query_tile):
        super().__init__()
        self.heads=heads;self.head_dim=dim//heads
        self.rope_theta=rope_theta;self.ca_query_tile=ca_query_tile
        self.qkv=nn.Linear(dim,3*dim,bias=False)
        self.out=nn.Linear(dim,dim,bias=False)
        self.out._residual_proj=True
        # S: positive trainable per-head score gain g = exp(rho), starting at 1.
        self.log_qk_gain=nn.Parameter(torch.zeros(heads))
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x).reshape(B,T,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        q,k=apply_rope(q,k,self.rope_theta)
        y=continuum_attention(q,k,v,self.log_qk_gain,self.ca_query_tile)
        return self.out(y.transpose(1,2).contiguous().reshape(B,T,C))


class GELUMLP(nn.Module):
    """Feed-forward network with tanh-approximate GELU."""
    def __init__(self,dim,ratio,dropout):
        super().__init__()
        hidden=int(dim*ratio)
        self.fc1=nn.Linear(dim,hidden,bias=False)
        self.fc2=nn.Linear(hidden,dim,bias=False)
        self.fc2._residual_proj=True
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x),approximate='tanh')))


class CADecoderBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=cfg.d_model
        self.norm1=RMSNorm(d,cfg.eps)
        self.attn=ContinuumSelfAttention(d,cfg.n_heads,cfg.rope_theta,cfg.ca_query_tile)
        self.norm2=RMSNorm(d,cfg.eps)
        self.mlp=GELUMLP(d,cfg.mlp_ratio,cfg.dropout)
    def forward(self,x):
        x=x+self.attn(self.norm1(x))
        return x+self.mlp(self.norm2(x))


class ContinuumDecoderLM(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.byte_emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([CADecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f=RMSNorm(cfg.d_model,cfg.eps)
        # The output projection is not tied to the input byte embedding.
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.residual_layers=cfg.n_layers
        self.residual_init_scale=1.0/math.sqrt(2.0*cfg.n_layers)
        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            std=.02*self.residual_init_scale if getattr(m,'_residual_proj',False) else .02
            nn.init.normal_(m.weight,mean=0,std=std)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0,std=.02)
    def forward(self,input_ids,targets=None):
        if input_ids.dim()!=2 or input_ids.shape[1]>self.cfg.block_size:
            raise ValueError('Input must be [B,T] with T<=block_size')
        x=self.drop(self.byte_emb(input_ids))
        for block in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x=torch.utils.checkpoint.checkpoint(block,x,use_reentrant=False)
            else:
                x=block(x)
        logits=self.lm_head(self.norm_f(x))
        out={'logits':logits}
        if targets is not None:
            out['loss']=F.cross_entropy(logits[:,:-1,:].contiguous().reshape(-1,self.cfg.vocab_size),
                                       targets[:,1:].contiguous().reshape(-1),reduction='mean')
        return out
    def parameter_breakdown(self):
        return {'byte_embedding':count_parameters(self.byte_emb),
                'decoder_blocks':count_parameters(self.blocks),
                'continuum_head_gains':self.cfg.n_layers*self.cfg.n_heads,
                'final_norm':count_parameters(self.norm_f),
                'lm_head':count_parameters(self.lm_head)}


def make_config(args):
    return ContinuumDecoderConfig(
        vocab_size=256,block_size=args.block_size,d_model=args.d_model,
        n_heads=args.n_heads,n_layers=args.n_layers,mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,rope_theta=args.rope_theta,
        grad_checkpointing=args.grad_checkpointing,ca_query_tile=args.ca_query_tile)


def parse_args():
    p=argparse.ArgumentParser('Exact continuum-attention normal causal decoder')
    p.add_argument('--mode',choices=['train','eval_bpb','param_count','ca_check','ca_cuda_check','causality_test'],default='train')
    p.add_argument('--output_dir',default=None)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--dataset_name',default=None)
    p.add_argument('--dataset_config',default=None)
    p.add_argument('--split',default='train')
    p.add_argument('--text_column',default='text')
    p.add_argument('--data_files',default=None)
    p.add_argument('--no_streaming',action='store_true')
    p.add_argument('--shuffle_buffer',type=int,default=20000)
    p.add_argument('--no_document_separator',action='store_true')
    p.add_argument('--block_size',type=int,default=2048)
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--grad_accum',type=int,default=8)
    p.add_argument('--max_steps',type=int,default=50000)
    p.add_argument('--max_batches',type=int,default=200)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--d_model',type=int,default=1024)
    p.add_argument('--n_heads',type=int,default=16)
    p.add_argument('--n_layers',type=int,default=8)
    p.add_argument('--mlp_ratio',type=float,default=4.0)
    p.add_argument('--rope_theta',type=float,default=10000.0)
    p.add_argument('--ca_query_tile',type=int,default=64,help='Independent query tile count, controls memory only')
    p.add_argument('--dropout',type=float,default=0.0)
    p.add_argument('--grad_checkpointing',action='store_true')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--min_lr',type=float,default=3e-5)
    p.add_argument('--warmup_steps',type=int,default=2000)
    p.add_argument('--weight_decay',type=float,default=0.1)
    p.add_argument('--beta1',type=float,default=0.9)
    p.add_argument('--beta2',type=float,default=0.95)
    p.add_argument('--grad_clip',type=float,default=1.0)
    p.add_argument('--amp_dtype',choices=['none','fp16','bf16'],default='fp16')
    p.add_argument('--compile',action='store_true')
    p.add_argument('--compile_mode',choices=['default','max-autotune-no-cudagraphs','reduce-overhead'],default='default')
    p.add_argument('--prefetch_batches',type=int,default=2)
    p.add_argument('--no_tf32',action='store_true')
    p.add_argument('--resume',default=None)
    p.add_argument('--log_every',type=int,default=10)
    p.add_argument('--save_every',type=int,default=1000)
    p.add_argument('--keep_last',type=int,default=3)
    p.add_argument('--wandb_project',default=None)
    p.add_argument('--wandb_run_name',default=None)
    p.add_argument('--output',default=None)
    p.add_argument('--atol',type=float,default=1e-5)
    return p.parse_args()
class _BatchPrefetcher:
    """Depth-k background prefetch around a batch iterator.

    Yields the exact same items in the exact same order as the wrapped
    iterator; the only effect is that the next item's production (HF stream
    parsing, shard downloads, byte packing, device transfer) overlaps with
    GPU compute instead of stalling the training loop between micro-steps.
    """

    def __init__(self, iterator: Any, depth: int):
        import queue
        import threading

        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, depth))
        self._sentinel = object()
        self._err: Optional[BaseException] = None

        def _worker() -> None:
            try:
                for item in iterator:
                    self._q.put(item)
            except BaseException as e:  # propagate to the consumer
                self._err = e
            finally:
                self._q.put(self._sentinel)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def __iter__(self) -> "_BatchPrefetcher":
        return self

    def __next__(self) -> Any:
        item = self._q.get()
        if item is self._sentinel:
            if self._err is not None:
                raise self._err
            raise StopIteration
        return item


def autocast_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def safe_latest_symlink_or_copy(src: Path, latest: Path) -> None:
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        rel = os.path.relpath(src, latest.parent)
        latest.symlink_to(rel, target_is_directory=True)
    except OSError:
        shutil.copytree(src, latest)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: ContinuumDecoderConfig,
    step: int,
    tokens_seen: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ckpt_dir = output_dir / f"step_{step:08d}"
    tmp_dir = output_dir / f".tmp_step_{step:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    torch.save(unwrap_model(model).state_dict(), tmp_dir / "model.pt", _use_new_zipfile_serialization=False)
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt", _use_new_zipfile_serialization=False)
    if scaler is not None:
        torch.save(scaler.state_dict(), tmp_dir / "scaler.pt", _use_new_zipfile_serialization=False)

    with open(tmp_dir / "continuum_decoder_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(tmp_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "tokens_seen": tokens_seen, "extra": extra or {}}, f, indent=2)

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)
    safe_latest_symlink_or_copy(ckpt_dir, output_dir / "latest")
    return ckpt_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    if model is not None:
        state = torch.load(checkpoint_dir / "model.pt", map_location=map_location)
        unwrap_model(model).load_state_dict(state)
    if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=map_location))
    if scaler is not None and (checkpoint_dir / "scaler.pt").exists():
        scaler.load_state_dict(torch.load(checkpoint_dir / "scaler.pt", map_location=map_location))
    state_path = checkpoint_dir / "trainer_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"step": 0, "tokens_seen": 0, "extra": {}}


def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted([p for p in output_dir.glob("step_*") if p.is_dir()])
    for ckpt in ckpts[:-keep_last]:
        shutil.rmtree(ckpt, ignore_errors=True)


def train(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        raise ValueError("--output_dir is required for --mode train")
    if args.dataset_name is None:
        raise ValueError("--dataset_name is required for --mode train")

    ctx = init_distributed(seed=args.seed)
    output_dir = Path(args.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(args)
    model = ContinuumDecoderLM(cfg).to(ctx.device)
    if args.compile:
        compile_mode = None if args.compile_mode == "default" else args.compile_mode
        model = torch.compile(model, mode=compile_mode)  # type: ignore[attr-defined]

    optimizer = build_adamw(model, args.lr, args.weight_decay, betas=(args.beta1, args.beta2))
    scaler = torch.cuda.amp.GradScaler(enabled=(ctx.device.type == "cuda" and args.amp_dtype == "fp16"))
    start_step = 0
    tokens_seen = 0

    if args.resume:
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler, map_location=ctx.device)
        start_step = int(state.get("step", 0))
        tokens_seen = int(state.get("tokens_seen", 0))

    if ctx.is_distributed:
        has_disabled_branches = False
        model = DDP(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=has_disabled_branches,
        )

    if ctx.is_main:
        params = count_parameters(unwrap_model(model))
        cfg_record = asdict(cfg)
        cfg_record["residual_layers"] = unwrap_model(model).residual_layers
        cfg_record["residual_init_scale"] = unwrap_model(model).residual_init_scale
        print(json.dumps({"params": params, "config": cfg_record, "breakdown": unwrap_model(model).parameter_breakdown()}, indent=2), flush=True)
        with open(output_dir / "continuum_decoder_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_record, f, indent=2)
        logger = JsonlLogger(output_dir / "train_log.jsonl")
        if args.wandb_project:
            if wandb is None:
                raise RuntimeError("wandb is not installed")
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={**vars(args), "model": cfg_record, "params": params},
            )
    else:
        logger = None

    barrier()
    iterator = build_byte_batch_iterator(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        data_files=args.data_files,
        streaming=not args.no_streaming,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed + start_step,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=ctx.rank,
        world_size=ctx.world_size,
        add_document_separator=not args.no_document_separator,
        infinite=True,
        device=ctx.device,
    )
    if args.prefetch_batches > 0:
        iterator = _BatchPrefetcher(iterator, args.prefetch_batches)

    model.train()
    running_loss = 0.0
    running_micro = 0
    last_time = time.time()
    last_tokens = tokens_seen

    for step in range(start_step, args.max_steps):
        lr_value = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
        set_optimizer_lr(optimizer, lr_value)
        optimizer.zero_grad(set_to_none=True)
        step_loss_accum = torch.zeros((), device=ctx.device)

        for micro in range(args.grad_accum):
            batch = next(iterator)
            # Skip the DDP gradient all-reduce on all but the last micro-step:
            # gradients are summed locally and reduced exactly once per
            # optimizer step (identical result, 1/grad_accum the comms).
            sync_ctx = (
                model.no_sync()
                if ctx.is_distributed and micro < args.grad_accum - 1
                else nullcontext()
            )
            with sync_ctx:
                with autocast_context(ctx.device, args.amp_dtype):
                    out = model(batch, targets=batch)
                    loss = out["loss"] / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            # accumulate on-device; the GPU->CPU sync happens only at log time
            step_loss_accum += loss.detach()

        if args.grad_clip and args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(float("nan"))

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step_tokens = args.batch_size * args.block_size * ctx.world_size * args.grad_accum
        tokens_seen += step_tokens
        running_loss += step_loss_accum
        running_micro += 1

        if (step + 1) % args.log_every == 0:
            now = time.time()
            elapsed = max(now - last_time, 1e-9)
            bytes_per_sec = (tokens_seen - last_tokens) / elapsed
            avg_loss = float(running_loss) / max(running_micro, 1)
            record = {
                "step": step + 1,
                "loss_nats": avg_loss,
                "bpb": nats_to_bpb(avg_loss),
                "lm_loss_nats": avg_loss,
                "lm_bpb": nats_to_bpb(avg_loss),
                "lr": lr_value,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_seen": tokens_seen,
                "bytes_per_sec": bytes_per_sec,
                "world_size": ctx.world_size,
                "microbatch_per_gpu": args.batch_size,
                "grad_accum": args.grad_accum,
                "block_size": args.block_size,
            }
            if ctx.is_main:
                print(json.dumps(record), flush=True)
                if logger is not None:
                    logger.write(record)
                if args.wandb_project and wandb is not None:
                    wandb.log(record, step=step + 1)
            running_loss = 0.0
            running_micro = 0
            last_time = now
            last_tokens = tokens_seen

        if ctx.is_main and ((step + 1) % args.save_every == 0 or (step + 1) == args.max_steps):
            save_checkpoint(output_dir, model, optimizer, scaler, cfg, step + 1, tokens_seen, extra={"args": vars(args)})
            prune_checkpoints(output_dir, args.keep_last)
        if (step + 1) % args.save_every == 0:
            barrier()

    if ctx.is_main and args.wandb_project and wandb is not None:
        wandb.finish()
    cleanup()


@torch.no_grad()
def eval_bpb(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode eval_bpb")
    if args.dataset_name is None:
        raise ValueError("--dataset_name is required for --mode eval_bpb")
    if args.output is None:
        raise ValueError("--output is required for --mode eval_bpb")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = Path(args.checkpoint) / "continuum_decoder_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = {k: v for k, v in data.items() if k in ContinuumDecoderConfig.__dataclass_fields__}
        cfg = ContinuumDecoderConfig(**allowed)
    else:
        cfg = make_config(args)

    model = ContinuumDecoderLM(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    iterator = build_byte_batch_iterator(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        data_files=args.data_files,
        streaming=not args.no_streaming,
        shuffle_buffer=0,
        seed=args.seed,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        add_document_separator=not args.no_document_separator,
        infinite=True,
        device=device,
    )

    total_nll = 0.0
    total_bytes = 0
    start = time.time()
    for _ in range(args.max_batches):
        batch = next(iterator)
        with autocast_context(device, args.amp_dtype):
            logits = model(batch)["logits"]
            loss_sum = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size),
                batch[:, 1:].contiguous().view(-1),
                reduction="sum",
            )
        total_nll += float(loss_sum.detach().cpu())
        total_bytes += batch.size(0) * (batch.size(1) - 1)

    nll_nats = total_nll / max(total_bytes, 1)
    result = {
        "task": "continuum_decoder_byte_lm_bpb",
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_column": args.text_column,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "nll_nats": nll_nats,
        "bpb": nats_to_bpb(nll_nats),
        "bytes_scored": total_bytes,
        "elapsed_sec": time.time() - start,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)


def param_count(args: argparse.Namespace) -> None:
    cfg = make_config(args)
    model = ContinuumDecoderLM(cfg)
    result = {"params": count_parameters(model), "breakdown": model.parameter_breakdown(),
              "config": asdict(cfg), "architecture": "normal_decoder_transformer_CA",
              "attention": "exact_piecewise_linear_continuum", "residual_layers": model.residual_layers,
              "residual_init_scale": model.residual_init_scale}
    print(json.dumps(result, indent=2), flush=True)

def ca_check(device='cpu'):
    """Exact-algebra regression, causal and gradient checks, no dataset needed."""
    torch.manual_seed(1234)
    dev=torch.device(device)
    B,H,T,D=1,2,9,8
    q=torch.randn(B,H,T,D,device=dev,dtype=torch.float32,requires_grad=True)
    k=torch.randn(B,H,T,D,device=dev,dtype=torch.float32,requires_grad=True)
    v=torch.randn(B,H,T,D,device=dev,dtype=torch.float32,requires_grad=True)
    gain=torch.zeros(H,device=dev,dtype=torch.float32,requires_grad=True)
    g=torch.randn(B,H,T,D,device=dev,dtype=torch.float32)
    out=continuum_attention(q,k,v,gain,4)
    oracle_weights=_ca_reference_weights((q @ k.transpose(-2,-1)).reshape(B*H,T,T) / math.sqrt(D),0)
    oracle=(oracle_weights.reshape(B,H,T,T)@v)
    err=float((out-oracle).abs().max().detach().cpu())
    if err>2e-5:raise AssertionError(f'CA forward mismatch {err}')
    test_grads=torch.autograd.grad((out*g).sum(),(q,k,v,gain),retain_graph=False)
    oracle_grads=torch.autograd.grad((oracle*g).sum(),(q,k,v,gain),allow_unused=True)
    for name,a,b in zip(['Q','K','V','rho'],test_grads,oracle_grads):
        if name=='rho':
            # Gain is initialized at 1. Include the trainable gain in oracle.
            s=(q@k.transpose(-2,-1))*gain.exp().view(1,H,1,1)/math.sqrt(D)
            w=_ca_reference_weights(s.reshape(B*H,T,T),0).reshape(B,H,T,T)
            b=torch.autograd.grad((w@v*g).sum(),gain)[0]
        dif=(a-b).abs().max().item()
        print('CA-gradient',name,'max_abs',round(dif,8),flush=True)
        if dif>3e-4:raise AssertionError(f'CA {name} gradient mismatch {dif}')
    # Constant-score half-cap: standard uniform causal attention.
    const=torch.full((1,T,T),3.5,device=dev)
    w=_ca_reference_weights(const,0)
    target=torch.tril(torch.ones(T,T,device=dev))
    target=target/target.sum(-1,keepdim=True)
    if not torch.allclose(w[0],target,atol=1e-6,rtol=1e-6):
        raise AssertionError('Half-cap / uniform limit incorrect')
    # No future value can influence past output.
    v2=v.detach().clone(); v2[:,:,T//2+1:]+=1000
    z=continuum_attention(q.detach(),k.detach(),v2,gain.detach(),4)
    if not torch.allclose(out.detach()[:,:,:T//2+1],z[:,:,:T//2+1],atol=1e-5,rtol=1e-5):
        raise AssertionError('Causal value leak')
    print(json.dumps({'CA_check':'passed','device':str(dev),'max_forward_error':err}),flush=True)

def ca_cuda_preflight():
    """GPU-specific 9/192/512/2048-token score-gradient oracle tests.

    Runs BEFORE DDP training. A 2048-length failure must not be hidden by a
    passing 9-token example. CPU reference uses float64 autograd, which has
    no shared CUDA kernel code with the analytic implementation.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA preflight requested without CUDA')
    ext=_ca_cuda_extension()
    torch.manual_seed(20260814)
    for T,start,Q,magnitude in [
        (9,0,9,1.0),
        (192,144,32,1.0),
        (512,480,32,2.5),
        (2048,2024,24,2.5),
        (2048,0,24,0.0),
    ]:
        S=(torch.randn(1,Q,T,device='cuda')*magnitude).contiguous()
        GW=torch.randn_like(S)
        W,stats=ext.forward(S,start)
        means=(GW*W).sum(-1).contiguous().reshape(-1)
        DS=ext.backward(S,W,stats,GW,means,start)
        with torch.enable_grad():
            ref_scores=S.detach().to('cpu').double().requires_grad_(True)
            ref_weights=_ca_reference_weights(ref_scores,start)
            ref_grad=torch.autograd.grad(
                (ref_weights*GW.detach().cpu().double()).sum(),ref_scores)[0]
        wa=(W.detach().cpu().double()-ref_weights.detach()).abs().max().item()
        ga=(DS.detach().cpu().double()-ref_grad.detach()).abs().max().item()
        print(f'CUDA CA oracle T={T} q={start}:{start+Q} magnitude={magnitude} '
              f'forward_abs={wa:.6g} grad_abs={ga:.6g}',flush=True)
        torch.testing.assert_close(W.detach().cpu().double(),ref_weights.detach(),
                                   atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(DS.detach().cpu().double(),ref_grad.detach(),
                                   atol=3e-4,rtol=3e-3)
    # Direct comparison of the fused CUDA weight/score VJP against the
    # independent double precision oracle, including late causal tiles.
    for T,start,Q,magnitude in [
        (9,0,9,1.0),
        (192,160,32,2.5),
        (512,480,32,2.5),
        (2048,2024,24,2.5),
        (2048,0,24,0.0),
    ]:
        K=start+Q  # The optimized path skips all future score columns.
        S=(torch.randn(1,Q,K,device='cuda')*magnitude).contiguous()
        GW=torch.randn_like(S)
        W,DS=ext.backward_fused(S,GW,start)
        with torch.enable_grad():
            rs=S.detach().cpu().double().requires_grad_(True)
            rw=_ca_reference_weights(rs,start)
            rd=torch.autograd.grad((rw*GW.detach().cpu().double()).sum(),rs)[0]
        torch.testing.assert_close(W.cpu().double(),rw.detach(),atol=3e-5,rtol=3e-5)
        torch.testing.assert_close(DS.cpu().double(),rd,atol=3e-4,rtol=3e-3)
        V=torch.randn(1,K,64,device='cuda',dtype=torch.float16)
        out=ext.direct_output(S,V,start)
        baseline=torch.bmm(W.to(V.dtype),V)
        torch.testing.assert_close(out.float(),baseline.float(),atol=8e-3,rtol=8e-3)
        print(f'FUSED CUDA oracle T={T} start={start} Q={Q}: PASSED',flush=True)
    # Complete forward and backward gradients, including dQ/dK/dV/dgain:
    # Compare dense, prefix and fused paths on the same inputs.
    import os as _os
    old_mode=_os.environ.get('CA_CUDA_MODE')
    try:
        runs={}
        for mode in ('dense','prefix','fused'):
            _os.environ['CA_CUDA_MODE']=mode
            torch.manual_seed(341)
            Qv=torch.randn(1,2,17,16,device='cuda',dtype=torch.float32,requires_grad=True)
            Kv=torch.randn_like(Qv,requires_grad=True)
            Vv=torch.randn_like(Qv,requires_grad=True)
            rh=torch.tensor([0.0,0.3],device='cuda',requires_grad=True)
            gg=torch.randn_like(Qv)
            yy=continuum_attention(Qv,Kv,Vv,rh,7)
            deriv=torch.autograd.grad((yy*gg).sum(),(Qv,Kv,Vv,rh))
            runs[mode]=(yy.detach(),*[x.detach() for x in deriv])
        for mode in ('prefix','fused'):
            for name,before,after in zip(('Y','dQ','dK','dV','dRho'),runs['dense'],runs[mode]):
                torch.testing.assert_close(after,before,atol=1e-3,rtol=1e-3,
                                           msg=lambda x:f'CUDA {mode} mismatch in {name}: {x}')
            print(f'END TO END CUDA CA {mode} against dense reference: PASSED',flush=True)
    finally:
        if old_mode is None:_os.environ.pop('CA_CUDA_MODE',None)
        else:_os.environ['CA_CUDA_MODE']=old_mode
    print('CUDA CA analytic forward/backward independent preflight: PASSED',flush=True)

@torch.no_grad()
def causality_test(args):
    cfg=make_config(args)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ContinuumDecoderLM(cfg).to(device).eval()
    T=min(cfg.block_size,32)
    x=torch.randint(0,256,(1,T),device=device)
    x2=x.clone();x2[:,T//2+1:]=torch.randint(0,256,(1,T-T//2-1),device=device)
    a=model(x)['logits'];b=model(x2)['logits']
    diff=(a[:,:T//2+1]-b[:,:T//2+1]).abs().max().item()
    print(json.dumps({'causality':'passed' if diff<=args.atol else 'FAILED',
                      'max_abs':diff,'T':T}),flush=True)
    if diff>args.atol:raise RuntimeError('CA decoder causal future leak')

def main():
    args=parse_args()
    if torch.cuda.is_available() and not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    if args.mode=='param_count':
        param_count(args)
    elif args.mode in ('ca_check','ca_cuda_check'):
        (ca_cuda_preflight() if args.mode=='ca_cuda_check' else ca_check('cpu'))
    elif args.mode=='causality_test':
        causality_test(args)
    elif args.mode=='train':
        train(args)
    elif args.mode=='eval_bpb':
        eval_bpb(args)
    else:
        raise ValueError(args.mode)

if __name__=='__main__':
    main()
