#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <dwmapi.h>
#include <directxmath.h>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <algorithm>
#include <cstdio>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx11.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dwmapi.lib")

#ifndef WDA_NONE
#define WDA_NONE 0
#endif
#ifndef WDA_MONITOR
#define WDA_MONITOR 1
#endif
#ifndef WDA_EXCLUDEFROMCAPTURE
#define WDA_EXCLUDEFROMCAPTURE 2
#endif

template <typename T>
void SafeRelease(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pd3dDeviceContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

bool g_pausedMinimized = false;
bool isClickThrough = true;
bool g_showDepthPreview = false;

float g_varianceRadius   = 4.0f;   // bilateral spatial spread (texels)
float g_structureWeight  = 0.5f;   // surface flatten: 0=crisp edges, 1=kill text floaters
float g_perspectiveWeight = 0.6f;  // ground-plane prior weight
float g_depthScale       = 1.0f;
float g_depthOffset      = 0.0f;
float g_edgeTolerance    = 0.12f;  // bilateral color range sigma (edge snap)
float g_hazeWeight       = 1.0f;   // aerial-perspective cue weight
float g_normalSmooth     = 0.0f;   // Gaussian depth smoothing for SSR normal stability
bool  g_invertDepth      = false;

HWND g_hwnd = nullptr;
HWND g_targetHwnd = nullptr;

int g_swapWidth = 0;
int g_swapHeight = 0;

IDXGIOutputDuplication* g_deskDup = nullptr;
ID3D11Texture2D* g_capturedColorTex = nullptr;
ID3D11ShaderResourceView* g_capturedColorSRV = nullptr;

ID3D11DepthStencilView* g_localDSV = nullptr;
ID3D11Texture2D* g_localDepthTex = nullptr;
ID3D11DepthStencilState* g_dss = nullptr;
ID3D11SamplerState* g_sampler = nullptr;
ID3D11BlendState* g_blendState = nullptr;
ID3D11Buffer* g_cbPreview = nullptr;

ID3D11VertexShader* g_vs = nullptr;
ID3D11PixelShader* g_psEstimate = nullptr;   // pass 1: raw monocular depth
ID3D11PixelShader* g_psDespeckle = nullptr;  // pass 2: adaptive despeckle -> SV_Depth

// Intermediate raw-depth target (pass 1 output -> pass 2 input).
ID3D11Texture2D* g_rawDepthTex = nullptr;
ID3D11RenderTargetView* g_rawDepthRTV = nullptr;
ID3D11ShaderResourceView* g_rawDepthSRV = nullptr;

const char* VS_SRC = R"(
struct VS_OUT {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};
VS_OUT main(uint id : SV_VertexID) {
    VS_OUT o;
    o.uv  = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
)";

// Shared HLSL prelude: constant buffer + helper functions used by both passes.
const char* HLSL_COMMON = R"(
cbuffer PreviewBuffer : register(b0) {
    int   previewActive;
    int   invertDepth;
    float varianceRadius;
    float structureWeight;
    float perspectiveWeight;
    float depthScale;
    float depthOffset;
    float edgeTolerance;
    float hazeWeight;
    float normalSmooth;   // Gaussian depth smoothing for stable normal reconstruction (SSR)
    float _pad1;
    float _pad2;
};

SamplerState smp : register(s0);

float getLuminance(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Perceptual color distance in a roughly luma-weighted space.
float colorDistance(float3 a, float3 b) {
    float3 d = a - b;
    return sqrt(dot(d * d, float3(0.5, 0.7, 0.3)));
}

// HSV-ish saturation without the full conversion.
float getSaturation(float3 c) {
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    return (mx - mn) / max(mx, 1e-4);
}

// ReShade applies a hyperbolic linearization curve to depth buffers.
// Our depth is linear (0=near, 1=far), so encode to hyperbolic and ReShade's
// internal linearizer decodes it back, preserving the gradient for RTGI.
float linear_to_hyperbolic(float linear_norm) {
    const float n = 1.0;
    const float f = 1000.0;
    float L = lerp(n, f, linear_norm);
    return (f - (n * f) / L) / (f - n);
}
)";

// ---------------------------------------------------------------------------
// PASS 1 - ESTIMATE
// Pure per-pixel monocular depth. No spatial filtering here: this pass only
// decides "how far is THIS pixel" from aerial perspective + a ground prior +
// sky handling, and writes linear depth (0=near, 1=far) to an R32_FLOAT target.
// Keeping it unfiltered means the despeckle pass gets a clean signal to work
// from, instead of us trying to smooth and estimate in the same step.
// ---------------------------------------------------------------------------
const char* PS_ESTIMATE_SRC = R"(
Texture2D<float4> gameTex : register(t0);

float estimateDepth(float3 col, float2 uv) {
    float lum = getLuminance(col);
    float sat = getSaturation(col);

    // Aerial perspective: distance grows as saturation falls and luma rises.
    float haze = saturate((1.0 - sat) * 0.65 + lum * 0.35);
    haze = pow(haze, 1.3);

    // Weak ground-plane prior: bottom near, top far.
    float ground = pow(saturate(1.0 - uv.y), 1.2);

    float wH = hazeWeight;
    float wP = perspectiveWeight;
    float total = wH + wP + 1e-4;
    float depth = (haze * wH + ground * wP) / total;

    // Sky -> far plane.
    float skyFactor = saturate(lum * 1.4) * saturate(1.0 - sat * 3.0) * saturate(uv.y * 2.0);
    depth = lerp(depth, 1.0, skyFactor * 0.85);

    return saturate(depth);
}

float main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_Target0 {
    float3 c = gameTex.Sample(smp, uv).rgb;
    return estimateDepth(c, uv);
}
)";

// ---------------------------------------------------------------------------
// PASS 2 - DESPECKLE
// Reads the raw depth (t0) and the color image (t1) and produces the final
// depth buffer. The key idea that fixes floaters around text/HUD is an
// ADAPTIVE, per-pixel coverage test rather than one global crispness slider:
//
//   For the center pixel we scan the neighborhood and split neighbors into
//   "similar color" vs "different color". A real surface silhouette has a
//   LARGE similar-color region on the center's side (high coverage) -> we
//   trust the edge and keep depth crisp. A thin feature like a text stroke or
//   a wire has only a FEW similar-color neighbors (low coverage) -> it is
//   detail painted onto a surface, so we replace its depth with the median-ish
//   depth of the DIFFERENT-color (background) neighbors, snapping it onto the
//   surface behind it. No floater, and true silhouettes stay sharp.
//
//   varianceRadius  -> neighborhood radius (texels)
//   edgeTolerance   -> color similarity threshold
//   structureWeight -> coverage threshold: how "thin" counts as detail
//                      (0 = flatten almost nothing, 1 = flatten aggressively)
// ---------------------------------------------------------------------------
const char* PS_DESPECKLE_SRC = R"(
Texture2D<float>  rawDepth : register(t0);
Texture2D<float4> gameTex  : register(t1);

struct PS_OUT {
    float4 col   : SV_Target0;
    float  depth : SV_Depth;
};

PS_OUT main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) {
    PS_OUT o;

    uint width, height;
    gameTex.GetDimensions(width, height);
    float2 texelSize = float2(1.0 / width, 1.0 / height);

    float3 c0 = gameTex.Sample(smp, uv).rgb;
    float  d0 = rawDepth.Sample(smp, uv).r;

    float radius = clamp(varianceRadius, 1.0, 15.0);
    float colorThresh = max(edgeTolerance, 0.01);
    int   iRadius = (int)ceil(radius);

    float spatialSigma = radius * 0.5;
    float twoSpatialSq = 2.0 * spatialSigma * spatialSigma;

    // Accumulators.
    float simDepthSum = 0.0;   // depth of same-surface (similar color) neighbors
    float simCount    = 0.0;
    float bgDepthSum  = 0.0;   // depth of background (different color) neighbors
    float bgCount     = 0.0;
    float bgWeightSum = 0.0;   // color-distance weighted, favors nearest bg color
    float bgDepthW    = 0.0;
    float total       = 0.0;
    float gaussDepth  = 0.0;   // pure Gaussian depth (color-blind) for smooth normals
    float gaussWeight = 0.0;

    [loop]
    for (int dy = -iRadius; dy <= iRadius; ++dy) {
        [loop]
        for (int dx = -iRadius; dx <= iRadius; ++dx) {
            float2 off = float2(dx, dy);
            float r2 = dot(off, off);
            if (r2 > radius * radius) continue;

            float2 suv = uv + off * texelSize;
            float3 c = gameTex.Sample(smp, suv).rgb;
            float  d = rawDepth.Sample(smp, suv).r;

            // Color-blind Gaussian: a C1-smooth depth field. SSR reconstructs
            // per-pixel normals by differentiating depth, so it needs this to
            // avoid normalize(0)=NaN (the white pixels) and structured noise on
            // flat-colored regions. Weighted by distance only, never clipped.
            float gW = exp(-r2 / twoSpatialSq);
            gaussDepth  += d * gW;
            gaussWeight += gW;

            float cd = colorDistance(c, c0);
            total += 1.0;

            if (cd <= colorThresh) {
                // Same surface as the center pixel.
                simDepthSum += d;
                simCount    += 1.0;
            } else {
                // Belongs to a differently-colored region (potential background).
                float w = 1.0 / (cd + 0.05); // nearer colors weigh more
                bgDepthSum  += d;
                bgCount     += 1.0;
                bgDepthW    += d * w;
                bgWeightSum += w;
            }
        }
    }

    // Coverage = fraction of the neighborhood that shares the center's color.
    // Low coverage => the center is a thin feature painted onto something else.
    float coverage = simCount / max(total, 1.0);

    // Depth of the surrounding surface (color-weighted so we snap to the
    // dominant nearby background rather than an average of everything).
    float surfaceDepth = (bgWeightSum > 1e-4) ? (bgDepthW / bgWeightSum) : d0;

    // Same-surface smoothed depth (removes raw-estimate noise on real surfaces).
    float surfSmooth = (simCount > 0.0) ? (simDepthSum / simCount) : d0;

    // How aggressively we treat low-coverage pixels as detail to flatten.
    // structureWeight in [0,1] maps to a coverage cutoff in ~[0.05, 0.6].
    float cutoff = lerp(0.05, 0.6, saturate(structureWeight));

    // If coverage is below the cutoff, this pixel is thin detail: pull it to
    // the surface behind it. Smoothstep gives a soft transition so we don't
    // get a hard on/off boundary that itself becomes an edge.
    float flatten = 1.0 - smoothstep(cutoff * 0.5, cutoff, coverage);
    float finalDepth = lerp(surfSmooth, surfaceDepth, flatten);

    // Normal Smoothing: blend toward the color-blind Gaussian. At 0 the buffer
    // keeps crisp silhouettes (best for RTGI/MXAO). Raising it feeds SSR a
    // differentiable depth field so its reconstructed normals stop producing
    // NaN white specks and interference patterns on flat surfaces. Silhouettes
    // soften as it rises, so use the lowest value that stabilizes reflections.
    float gauss = (gaussWeight > 1e-4) ? (gaussDepth / gaussWeight) : finalDepth;
    finalDepth = lerp(finalDepth, gauss, saturate(normalSmooth));

    // User trims.
    finalDepth = saturate(finalDepth * depthScale + depthOffset);
    if (invertDepth) finalDepth = 1.0 - finalDepth;
    finalDepth = clamp(finalDepth, 0.01, 0.99);

    o.depth = linear_to_hyperbolic(finalDepth);
    o.col   = previewActive ? float4(finalDepth.xxx, 1.0) : float4(c0, 1.0);
    return o;
}
)";

bool SetOverlayCaptureExclusion(HWND hwnd) {
    typedef BOOL (WINAPI *SWDA_t)(HWND, DWORD);
    SWDA_t fn = (SWDA_t)GetProcAddress(GetModuleHandleA("user32.dll"), "SetWindowDisplayAffinity");
    if (!fn) return false;
    if (fn(hwnd, WDA_EXCLUDEFROMCAPTURE)) return true;
    if (fn(hwnd, WDA_MONITOR)) return true;
    return false;
}

bool CreateDepthBuffer(int width, int height) {
    SafeRelease(g_localDepthTex);
    SafeRelease(g_localDSV);

    D3D11_TEXTURE2D_DESC dd = {};
    dd.Width = width;
    dd.Height = height;
    dd.MipLevels = 1;
    dd.ArraySize = 1;
    dd.Format = DXGI_FORMAT_R32_TYPELESS;
    dd.SampleDesc.Count = 1;
    dd.Usage = D3D11_USAGE_DEFAULT;
    dd.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
    
    HRESULT hr = g_pd3dDevice->CreateTexture2D(&dd, nullptr, &g_localDepthTex);
    if (FAILED(hr) || !g_localDepthTex) {
        std::cerr << "Failed to create depth texture! HR: 0x" << std::hex << hr << std::endl;
        return false;
    }

    D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
    dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    dsvDesc.Texture2D.MipSlice = 0;
    
    hr = g_pd3dDevice->CreateDepthStencilView(g_localDepthTex, &dsvDesc, &g_localDSV);
    if (FAILED(hr) || !g_localDSV) {
        std::cerr << "Failed to create DSV! HR: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // Intermediate raw-depth target: pass 1 writes here, pass 2 samples it.
    SafeRelease(g_rawDepthTex);
    SafeRelease(g_rawDepthRTV);
    SafeRelease(g_rawDepthSRV);

    D3D11_TEXTURE2D_DESC rd = {};
    rd.Width = width;
    rd.Height = height;
    rd.MipLevels = 1;
    rd.ArraySize = 1;
    rd.Format = DXGI_FORMAT_R32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Usage = D3D11_USAGE_DEFAULT;
    rd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    hr = g_pd3dDevice->CreateTexture2D(&rd, nullptr, &g_rawDepthTex);
    if (FAILED(hr) || !g_rawDepthTex) {
        std::cerr << "Failed to create raw-depth texture! HR: 0x" << std::hex << hr << std::endl;
        return false;
    }
    hr = g_pd3dDevice->CreateRenderTargetView(g_rawDepthTex, nullptr, &g_rawDepthRTV);
    if (FAILED(hr) || !g_rawDepthRTV) {
        std::cerr << "Failed to create raw-depth RTV! HR: 0x" << std::hex << hr << std::endl;
        return false;
    }
    hr = g_pd3dDevice->CreateShaderResourceView(g_rawDepthTex, nullptr, &g_rawDepthSRV);
    if (FAILED(hr) || !g_rawDepthSRV) {
        std::cerr << "Failed to create raw-depth SRV! HR: 0x" << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

void ResizeSwapChain(int width, int height) {
    if (!g_pSwapChain || width <= 0 || height <= 0) return;
    if (g_swapWidth == width && g_swapHeight == height) return;

    SafeRelease(g_mainRenderTargetView);
    SafeRelease(g_localDepthTex);
    SafeRelease(g_localDSV);
    SafeRelease(g_rawDepthTex);
    SafeRelease(g_rawDepthRTV);
    SafeRelease(g_rawDepthSRV);

    ImGui_ImplDX11_InvalidateDeviceObjects();

    HRESULT hr = g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) {
        std::cerr << "ResizeBuffers failed! HR: 0x" << std::hex << hr << std::endl;
        return;
    }

    g_swapWidth = width;
    g_swapHeight = height;

    ID3D11Texture2D* bb = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&bb));
    if (bb) {
        g_pd3dDevice->CreateRenderTargetView(bb, nullptr, &g_mainRenderTargetView);

        D3D11_TEXTURE2D_DESC bbDesc;
        bb->GetDesc(&bbDesc);
        bb->Release();

        CreateDepthBuffer(bbDesc.Width, bbDesc.Height);
    }

    ImGui_ImplDX11_CreateDeviceObjects();
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
            break;
        case WM_SIZE:
            if (g_pd3dDevice && g_pSwapChain) {
                ResizeSwapChain(LOWORD(lParam), HIWORD(lParam));
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

bool InitDXGIDuplication() {
    SafeRelease(g_deskDup);

    IDXGIDevice* dxgiDevice = nullptr;
    g_pd3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);

    IDXGIAdapter* adapter = nullptr;
    if (dxgiDevice) dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&adapter);
    SafeRelease(dxgiDevice);
    if (!adapter) return false;

    IDXGIOutput* output = nullptr;
    adapter->EnumOutputs(0, &output);
    SafeRelease(adapter);
    if (!output) return false;

    IDXGIOutput1* output1 = nullptr;
    output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&output1);
    SafeRelease(output);
    if (!output1) return false;

    HRESULT hr = output1->DuplicateOutput(g_pd3dDevice, &g_deskDup);
    SafeRelease(output1);
    return SUCCEEDED(hr);
}

void CaptureFrame() {
    if (!g_deskDup) {
        InitDXGIDuplication();
        if (!g_deskDup) return;
    }

    DXGI_OUTDUPL_FRAME_INFO frameInfo = {};
    IDXGIResource* desktopRes = nullptr;
    HRESULT hr = g_deskDup->AcquireNextFrame(100, &frameInfo, &desktopRes);

    if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_ACCESS_DENIED) {
        SafeRelease(desktopRes);
        InitDXGIDuplication();
        return;
    }
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        SafeRelease(desktopRes);
        return;
    }
    if (FAILED(hr)) {
        SafeRelease(desktopRes);
        return;
    }

    ID3D11Texture2D* desktopTex = nullptr;
    if (desktopRes) {
        desktopRes->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&desktopTex);
    }

    RECT rc = {};
    if (g_targetHwnd && IsWindow(g_targetHwnd) && IsWindowVisible(g_targetHwnd)) {
        GetWindowRect(g_targetHwnd, &rc);
    } else {
        rc = { 0, 0, (LONG)GetSystemMetrics(SM_CXSCREEN), (LONG)GetSystemMetrics(SM_CYSCREEN) };
    }

    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;

    if (w > 0 && h > 0 && desktopTex) {
        D3D11_TEXTURE2D_DESC srcDesc;
        desktopTex->GetDesc(&srcDesc);

        D3D11_BOX box;
        box.left   = std::max<LONG>(0, rc.left);
        box.top    = std::max<LONG>(0, rc.top);
        box.right  = std::min<LONG>((LONG)srcDesc.Width, rc.right);
        box.bottom = std::min<LONG>((LONG)srcDesc.Height, rc.bottom);
        box.front  = 0; box.back = 1;

        int copyW = box.right - box.left;
        int copyH = box.bottom - box.top;

        static int lastW = 0, lastH = 0;
        if (copyW != lastW || copyH != lastH) {
            SafeRelease(g_capturedColorTex);
            SafeRelease(g_capturedColorSRV);

            D3D11_TEXTURE2D_DESC desc = {};
            desc.Width = copyW;
            desc.Height = copyH;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            g_pd3dDevice->CreateTexture2D(&desc, nullptr, &g_capturedColorTex);
            g_pd3dDevice->CreateShaderResourceView(g_capturedColorTex, nullptr, &g_capturedColorSRV);

            lastW = copyW; lastH = copyH;
        }

        if (g_capturedColorTex) {
            g_pd3dDeviceContext->CopySubresourceRegion(g_capturedColorTex, 0, 0, 0, 0, desktopTex, 0, &box);
        }
    }

    g_deskDup->ReleaseFrame();
    SafeRelease(desktopTex);
    SafeRelease(desktopRes);
}

void OpenReshadeMenu() {
    if (!g_targetHwnd || isClickThrough) return;
    PostMessageW(g_targetHwnd, WM_KEYDOWN, VK_HOME, 0x00470001);
    PostMessageW(g_targetHwnd, WM_KEYUP,   VK_HOME, 0xC0470001);
}

void UpdateOverlayPosition() {
    RECT rc = {};
    if (g_targetHwnd && IsWindow(g_targetHwnd) && IsWindowVisible(g_targetHwnd)) {
        GetWindowRect(g_targetHwnd, &rc);
    } else {
        g_targetHwnd = nullptr;
        rc = { 0, 0, (LONG)GetSystemMetrics(SM_CXSCREEN), (LONG)GetSystemMetrics(SM_CYSCREEN) };
    }

    int w = rc.right - rc.left;
    int h = rc.bottom - rc.top;
    if (w <= 0 || h <= 0) return;

    RECT curRc = {};
    GetWindowRect(g_hwnd, &curRc);

    if (curRc.left != rc.left || curRc.top != rc.top ||
        (curRc.right - curRc.left) != w || (curRc.bottom - curRc.top) != h) {
        SetWindowPos(g_hwnd, HWND_TOPMOST, rc.left, rc.top, w, h,
                     SWP_NOACTIVATE | SWP_SHOWWINDOW);
        ResizeSwapChain(w, h);
    }
}

void RenderOverlayEngine() {
    if (!g_pd3dDeviceContext || !g_mainRenderTargetView || !g_localDSV) return;
    if (!g_rawDepthRTV || !g_rawDepthSRV) return;

    float clear[4] = { 0, 0, 0, 1.0f };
    g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear);
    g_pd3dDeviceContext->ClearDepthStencilView(g_localDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    RECT rc;
    GetClientRect(g_hwnd, &rc);
    if (rc.right == 0 || rc.bottom == 0) return;

    D3D11_VIEWPORT vp = { 0, 0, (float)rc.right, (float)rc.bottom, 0, 1 };
    g_pd3dDeviceContext->RSSetViewports(1, &vp);

    if (!g_capturedColorSRV) {
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        return;
    }

    // Shared constants for both passes.
    struct CB {
        int   previewActive;
        int   invertDepth;
        float varianceRadius;
        float structureWeight;
        float perspectiveWeight;
        float depthScale;
        float depthOffset;
        float edgeTolerance;
        float hazeWeight;
        float normalSmooth;
        float _pad1;
        float _pad2;
    } cb;

    cb.previewActive     = g_showDepthPreview ? 1 : 0;
    cb.invertDepth       = g_invertDepth ? 1 : 0;
    cb.varianceRadius    = g_varianceRadius;
    cb.structureWeight   = g_structureWeight;
    cb.perspectiveWeight = g_perspectiveWeight;
    cb.depthScale        = g_depthScale;
    cb.depthOffset       = g_depthOffset;
    cb.edgeTolerance     = g_edgeTolerance;
    cb.hazeWeight        = g_hazeWeight;
    cb.normalSmooth      = g_normalSmooth;
    cb._pad1 = cb._pad2 = 0.0f;

    g_pd3dDeviceContext->UpdateSubresource(g_cbPreview, 0, nullptr, &cb, 0, 0);
    g_pd3dDeviceContext->PSSetConstantBuffers(0, 1, &g_cbPreview);
    g_pd3dDeviceContext->PSSetSamplers(0, 1, &g_sampler);
    g_pd3dDeviceContext->VSSetShader(g_vs, nullptr, 0);
    g_pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    ID3D11ShaderResourceView* nullsrv2[2] = { nullptr, nullptr };

    // ---- Pass 1: estimate raw depth into g_rawDepthRTV (no depth test) ----
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_rawDepthRTV, nullptr);
    g_pd3dDeviceContext->OMSetDepthStencilState(nullptr, 0);
    g_pd3dDeviceContext->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
    g_pd3dDeviceContext->PSSetShaderResources(0, 1, &g_capturedColorSRV);
    g_pd3dDeviceContext->PSSetShader(g_psEstimate, nullptr, 0);
    g_pd3dDeviceContext->Draw(3, 0);

    // Detach the color SRV before rebinding it at t1 for pass 2.
    g_pd3dDeviceContext->PSSetShaderResources(0, 1, nullsrv2);

    // ---- Pass 2: despeckle raw depth -> backbuffer color + SV_Depth -------
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, g_localDSV);
    g_pd3dDeviceContext->OMSetDepthStencilState(g_dss, 0);
    g_pd3dDeviceContext->OMSetBlendState(g_blendState, nullptr, 0xFFFFFFFF);
    ID3D11ShaderResourceView* srvs[2] = { g_rawDepthSRV, g_capturedColorSRV };
    g_pd3dDeviceContext->PSSetShaderResources(0, 2, srvs);
    g_pd3dDeviceContext->PSSetShader(g_psDespeckle, nullptr, 0);
    g_pd3dDeviceContext->Draw(3, 0);

    // Unbind so the raw-depth target is free to be an RTV again next frame.
    g_pd3dDeviceContext->PSSetShaderResources(0, 2, nullsrv2);
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
}

void Cleanup() {
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    SafeRelease(g_sampler);
    SafeRelease(g_blendState);
    SafeRelease(g_cbPreview);
    SafeRelease(g_vs);
    SafeRelease(g_psEstimate);
    SafeRelease(g_psDespeckle);
    SafeRelease(g_capturedColorSRV);
    SafeRelease(g_capturedColorTex);
    SafeRelease(g_dss);
    SafeRelease(g_localDSV);
    SafeRelease(g_localDepthTex);
    SafeRelease(g_rawDepthTex);
    SafeRelease(g_rawDepthRTV);
    SafeRelease(g_rawDepthSRV);
    SafeRelease(g_mainRenderTargetView);
    SafeRelease(g_pSwapChain);
    SafeRelease(g_pd3dDeviceContext);
    SafeRelease(g_pd3dDevice);

    if (g_deskDup) { g_deskDup->Release(); g_deskDup = nullptr; }
    if (g_hwnd) { DestroyWindow(g_hwnd); g_hwnd = nullptr; }
}

void WriteReshadeIni() {
    FILE* f = fopen("ReShade.ini", "w");
    if (f) {
        fprintf(f,
            "[GENERAL]\n"
            "EffectFilePaths=.\\Shaders\n"
            "TextureSearchPaths=.\\Textures\n"
            "PreprocessorDefinitions=RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN=0,RESHADE_DEPTH_INPUT_IS_REVERSED=0,RESHADE_DEPTH_INPUT_IS_LOGARITHMIC=0\n"
            "AddonPath=.\\addons\n"
            "\n"
            "[GENERIC_DEPTH]\n"
            "DepthCopyBeforeClears=0\n"
            "UseAspectRatioHeuristics=0\n"
            "FilterFormat=0\n"
            "DrawStatsHeuristic=0\n"
        );
        fclose(f);
        std::cout << "Generated ReShade.ini" << std::endl;
    }
}

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR, int) {
    SetProcessDPIAware();

    AllocConsole();
    FILE* f;
    freopen_s(&f, "CONOUT$", "w", stdout);
    freopen_s(&f, "CONOUT$", "w", stderr);

    std::cout << "=== United Overlay Starting ===" << std::endl;

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"UnitedOverlay";
    RegisterClassExW(&wc);

    g_hwnd = CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
        wc.lpszClassName, L"United Overlay",
        WS_POPUP,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, wc.hInstance, nullptr
    );

    if (!g_hwnd) {
        std::cerr << "Failed to create window!" << std::endl;
        return 1;
    }

    SetLayeredWindowAttributes(g_hwnd, 0, 255, LWA_ALPHA);

    if (!SetOverlayCaptureExclusion(g_hwnd)) {
        std::cerr << "Warning: SetWindowDisplayAffinity failed." << std::endl;
    }

    MARGINS m = { -1, -1, -1, -1 };
    DwmExtendFrameIntoClientArea(g_hwnd, &m);

    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0, D3D11_SDK_VERSION, &sd, &g_pSwapChain,
        &g_pd3dDevice, nullptr, &g_pd3dDeviceContext);

    if (FAILED(hr)) {
        std::cerr << "D3D11CreateDeviceAndSwapChain failed! HR: 0x" << std::hex << hr << std::endl;
        Cleanup();
        return 1;
    }

    std::cout << "D3D11 Device created successfully." << std::endl;
    WriteReshadeIni();

    RECT cr;
    GetClientRect(g_hwnd, &cr);
    g_swapWidth = cr.right;
    g_swapHeight = cr.bottom;

    ID3D11Texture2D* bb = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&bb));
    if (bb) {
        g_pd3dDevice->CreateRenderTargetView(bb, nullptr, &g_mainRenderTargetView);
        D3D11_TEXTURE2D_DESC bbDesc;
        bb->GetDesc(&bbDesc);
        g_swapWidth = bbDesc.Width;
        g_swapHeight = bbDesc.Height;
        bb->Release();

        if (!CreateDepthBuffer(g_swapWidth, g_swapHeight)) {
            std::cerr << "Failed to create initial depth buffer!" << std::endl;
            Cleanup();
            return 1;
        }
        std::cout << "Depth buffer created: " << g_swapWidth << "x" << g_swapHeight << std::endl;
    }

    D3D11_DEPTH_STENCIL_DESC dssd = {};
    dssd.DepthEnable = TRUE;
    dssd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dssd.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
    dssd.StencilEnable = FALSE;
    g_pd3dDevice->CreateDepthStencilState(&dssd, &g_dss);

    D3D11_BLEND_DESC bd = {};
    bd.RenderTarget[0].BlendEnable = TRUE;
    bd.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    bd.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    bd.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    g_pd3dDevice->CreateBlendState(&bd, &g_blendState);

    ID3DBlob* vb = nullptr;
    ID3DBlob* pb = nullptr;
    ID3DBlob* errorBlob = nullptr;
    
    hr = D3DCompile(VS_SRC, strlen(VS_SRC), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vb, &errorBlob);
    if (FAILED(hr)) {
        std::cerr << "Vertex shader compilation failed! HR: 0x" << std::hex << hr << std::endl;
        if (errorBlob) {
            std::cerr << "Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            SafeRelease(errorBlob);
        }
        Cleanup();
        return 1;
    }
    
    g_pd3dDevice->CreateVertexShader(vb->GetBufferPointer(), vb->GetBufferSize(), nullptr, &g_vs);
    SafeRelease(vb);

    // Both pixel shaders share HLSL_COMMON (cbuffer + helpers); concatenate the
    // prelude with each body since D3DCompile takes a single source string.
    std::string estimateSrc  = std::string(HLSL_COMMON) + PS_ESTIMATE_SRC;
    std::string despeckleSrc = std::string(HLSL_COMMON) + PS_DESPECKLE_SRC;

    hr = D3DCompile(estimateSrc.c_str(), estimateSrc.size(), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &pb, &errorBlob);
    if (FAILED(hr)) {
        std::cerr << "Estimate PS compilation failed! HR: 0x" << std::hex << hr << std::endl;
        if (errorBlob) {
            std::cerr << "Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            SafeRelease(errorBlob);
        }
        Cleanup();
        return 1;
    }
    g_pd3dDevice->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &g_psEstimate);
    SafeRelease(pb);
    SafeRelease(errorBlob);

    hr = D3DCompile(despeckleSrc.c_str(), despeckleSrc.size(), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &pb, &errorBlob);
    if (FAILED(hr)) {
        std::cerr << "Despeckle PS compilation failed! HR: 0x" << std::hex << hr << std::endl;
        if (errorBlob) {
            std::cerr << "Error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            SafeRelease(errorBlob);
        }
        Cleanup();
        return 1;
    }
    g_pd3dDevice->CreatePixelShader(pb->GetBufferPointer(), pb->GetBufferSize(), nullptr, &g_psDespeckle);
    SafeRelease(pb);
    SafeRelease(errorBlob);

    std::cout << "Shaders compiled successfully." << std::endl;

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(float) * 12; // 2 ints + 6 floats + 4 pad, 16-byte aligned
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    g_pd3dDevice->CreateBuffer(&cbd, nullptr, &g_cbPreview);

    D3D11_SAMPLER_DESC ssd = {};
    ssd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    ssd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    ssd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    ssd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    g_pd3dDevice->CreateSamplerState(&ssd, &g_sampler);

    if (!InitDXGIDuplication()) {
        std::cerr << "Warning: Desktop Duplication failed. Will retry..." << std::endl;
    } else {
        std::cout << "Desktop Duplication initialized." << std::endl;
    }

    ImGui::CreateContext();
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ShowWindow(g_hwnd, SW_SHOWDEFAULT);

    std::cout << "=== Overlay Running ===" << std::endl;
    std::cout << "INSERT : toggle click-through" << std::endl;
    std::cout << "F1     : pause/minimize overlay" << std::endl;
    std::cout << "F7     : attach to focused window" << std::endl;
    std::cout << "HOME   : send ReShade menu key" << std::endl;

    bool done = false;
    uint64_t frameCount = 0;
    uint64_t lastLogFrame = 0;

    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }

        if (done) break;

        if (GetAsyncKeyState(VK_INSERT) & 1) {
            isClickThrough = !isClickThrough;
            LONG_PTR ex = GetWindowLongPtr(g_hwnd, GWL_EXSTYLE);
            if (isClickThrough) {
                ex |= (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
                SetWindowLongPtr(g_hwnd, GWL_EXSTYLE, ex);
                SetWindowPos(g_hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE);
                if (g_targetHwnd) SetForegroundWindow(g_targetHwnd);
            } else {
                ex &= ~(WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
                SetWindowLongPtr(g_hwnd, GWL_EXSTYLE, ex);
                SetWindowPos(g_hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
                SetForegroundWindow(g_hwnd);
            }
        }

        if (GetAsyncKeyState(VK_F1) & 1) {
            g_pausedMinimized = !g_pausedMinimized;
            ShowWindow(g_hwnd, g_pausedMinimized ? SW_MINIMIZE : SW_RESTORE);
        }

        if (GetAsyncKeyState(VK_F7) & 1) {
            HWND focused = GetForegroundWindow();
            if (focused && focused != g_hwnd) {
                wchar_t title[512] = {0};
                if (IsWindowVisible(focused) && !IsIconic(focused) && GetWindowTextW(focused, title, 512) > 0) {
                    g_targetHwnd = focused;
                    std::wcout << L"Attached to: " << title << std::endl;
                }
            }
        }

        if (GetAsyncKeyState(VK_HOME) & 1) {
            OpenReshadeMenu();
        }

        if (g_pausedMinimized) { Sleep(25); continue; }

        UpdateOverlayPosition();
        CaptureFrame();
        RenderOverlayEngine();

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (!isClickThrough) {
            ImGui::Begin("Overlay Control");
            ImGui::Checkbox("Show Depth Preview", &g_showDepthPreview);
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Depth Method: Two-Pass Estimate + Coverage Despeckle");

            ImGui::Separator();
            ImGui::Text("Depth Tuning:");
            ImGui::SliderFloat("Bilateral Radius", &g_varianceRadius, 1.0f, 15.0f);
            ImGui::SliderFloat("Edge Snap (range sigma)", &g_edgeTolerance, 0.01f, 0.5f, "%.3f");
            ImGui::SliderFloat("Haze / Aerial Weight", &g_hazeWeight, 0.0f, 2.0f);
            ImGui::SliderFloat("Ground-Plane Weight", &g_perspectiveWeight, 0.0f, 2.0f);
            ImGui::SliderFloat("Detail Flatten (coverage cutoff)", &g_structureWeight, 0.0f, 1.0f);
            ImGui::SliderFloat("Normal Smoothing (SSR stability)", &g_normalSmooth, 0.0f, 1.0f);
            ImGui::Checkbox("Invert Depth", &g_invertDepth);
            ImGui::SliderFloat("Depth Scale", &g_depthScale, 0.0f, 2.0f);
            ImGui::SliderFloat("Depth Offset", &g_depthOffset, -1.0f, 1.0f);

            ImGui::Separator();
            if (g_targetHwnd) {
                ImGui::Text("Attached to: 0x%p", g_targetHwnd);
                if (ImGui::Button("Detach")) g_targetHwnd = nullptr;
            } else {
                ImGui::TextDisabled("No window attached. Press F7 to attach.");
            }
            
            ImGui::Separator();
            ImGui::Text("INSERT : toggle click-through");
            ImGui::Text("Status: %s", isClickThrough ? "CLICK-THROUGH" : "MENU MODE");
            ImGui::Text("F1     : pause / minimize overlay");
            ImGui::Text("F7     : attach to focused window");
            ImGui::Text("HOME   : send ReShade menu key to target");
            ImGui::TextWrapped("NOTE   : Install ReShade to THIS overlay exe.");
            ImGui::TextWrapped("       : Depth buffer is D32_FLOAT (R32_TYPELESS).");
            if (ImGui::Button("Exit Overlay")) done = true;
            ImGui::End();
        }

        ImGui::Render();
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(0, 0);
        
        frameCount++;
        if (frameCount - lastLogFrame >= 60) {
            lastLogFrame = frameCount;
            HMODULE reshade = GetModuleHandleA("ReShade64.dll");
            std::cout << "[Frame " << frameCount << "] ReShade: " 
                      << (reshade ? "LOADED" : "NOT LOADED") << std::endl;
            if (g_localDepthTex) {
                D3D11_TEXTURE2D_DESC dd;
                g_localDepthTex->GetDesc(&dd);
                std::cout << "  Depth: " << dd.Width << "x" << dd.Height 
                          << " Format=" << dd.Format 
                          << " BindFlags=0x" << std::hex << dd.BindFlags << std::dec << std::endl;
            }
        }

        Sleep(1);
    }

    Cleanup();
    return 0;
}