#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 17:26:00 2025

@author: astronomy_zrf
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.optimize import lsq_linear
from scipy.interpolate import interp1d

def wavelet_denoise_1d(y, wavelet="db4", level=None, mode="symmetric"):
    """
    1D wavelet soft-threshold denoising (Donoho & Johnstone universal threshold).
    """
    y = np.asarray(y, dtype=float)
    w = pywt.Wavelet(wavelet)
    if level is None:
        level = pywt.dwt_max_level(len(y), w.dec_len)

    coeffs = pywt.wavedec(y, w, mode=mode, level=level)
    # 估计噪声
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size > 0 else 0.0
    if sigma <= 0 or not np.isfinite(sigma):
        return y  # 如果失败，返回原始信号

    # 通用阈值
    uthresh = sigma * np.sqrt(2 * np.log(y.size))

    # 阈值化
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(np.sign(c) * np.maximum(np.abs(c) - uthresh, 0.0))

    y_denoised = pywt.waverec(new_coeffs, w, mode=mode)
    # 对齐长度
    if y_denoised.shape[0] != y.shape[0]:
        y_denoised = y_denoised[: y.shape[0]] if y_denoised.shape[0] > y.shape[0] else np.pad(
            y_denoised, (0, y.shape[0] - y_denoised.shape[0]), mode="edge"
        )
    return y_denoised


def fit_spectrum(data, template_path, denoise=True, visualize=True, wavelet="db4"):
    """
    使用PCA模板拟合新光谱数据，并扣除宿主星系成分。
    
    Parameters
    ----------
    data : ndarray, shape (N,2)
        输入的新光谱 [wavelength, flux]。
    template_path : str
        PCA 模板文件路径。
    denoise : bool
        是否对输入 flux 进行小波降噪。
    visualize : bool
        是否绘制拟合与残差结果。
    wavelet : str
        小波基 (用于降噪)。
    
    Returns
    -------
    coeffs : ndarray
        拟合得到的 PCA 系数。
    host_fit : ndarray
        拟合得到的宿主星系成分。
    sn_residual : ndarray
        扣除宿主后残差 (SN 成分)。
    """
    # 读入模板
    template = np.loadtxt(template_path)
    wave_temp = template[:, 0]
    pcs = template[:, 1:]

    # 提取数据
    wave_data, flux_data = data[0], data[1]

    # 可选降噪
    if denoise:
        flux_data = wavelet_denoise_1d(flux_data, wavelet=wavelet, level=4, mode="symmetric")

    # 插值模板到观测波长
    pcs_interp = []
    for i in range(pcs.shape[1]):
        f = interp1d(wave_temp, pcs[:, i], kind="linear", bounds_error=False, fill_value=0)
        pcs_interp.append(f(wave_data))
    pcs_interp = np.array(pcs_interp).T  # shape: (N_wave, N_pc)

    # 拟合
    A = pcs_interp
    b = flux_data
    res = lsq_linear(A, b, bounds=(-np.inf, np.inf))
    coeffs = res.x

    host_fit = A @ coeffs
    sn_residual = flux_data - host_fit

    # 可选可视化
    if visualize:
        mask = (wave_data >= 4000) & (wave_data <= 7000)
        plt.figure(figsize=(10, 6))
        plt.plot(wave_data[mask], flux_data[mask], label="Observed (data)", alpha=0.8)
        plt.plot(wave_data[mask], host_fit[mask], label="Host galaxy (fit)", lw=2)
        plt.plot(wave_data[mask], sn_residual[mask], label="Residual (SN)", lw=2)
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.legend()
        plt.show()

    return coeffs, host_fit, sn_residual


def fit_spectrum_with_sn_template(
    data,
    host_template_path,
    sn_template_path=None,
    sn_n_components=None,
    host_n_components=None,
    denoise=True,
    visualize=True,
    plot_sn_component=True,
    wavelet="db4",
    transient_only=False   # 新增逻辑：只拟合 transient PCA
):
    """
    如果 transient_only=True：
        - 只对 sn_template 进行 PCA 分解拟合
        - 输出系数长度与常规相同：86(host) + 20(SN)
        - host 部分全部补 0
    """

    # -------------------------
    # 读取光谱
    # -------------------------
    wave_data, flux_data = data[0], data[1]

    if denoise:
        flux_data = wavelet_denoise_1d(flux_data, wavelet=wavelet, level=3, mode="symmetric")

    # ===============================
    # Case 1：transient_only = True
    # ===============================
    if transient_only:

        # ---- 加载 SN PCA ----
        assert sn_template_path is not None, "transient_only=True 需要 sn_template_path"

        sn_template = np.loadtxt(sn_template_path)
        wave_sn = sn_template[:, 0]
        pcs_sn = sn_template[:, 1:]

        if sn_n_components is not None:
            pcs_sn = pcs_sn[:, :sn_n_components]  # ← 这里通常是 20 个成分

        # --- 插值 SN PCA ---
        pcs_sn_interp = []
        for i in range(pcs_sn.shape[1]):
            f = interp1d(wave_sn, pcs_sn[:, i], bounds_error=False, fill_value=0)
            pcs_sn_interp.append(f(wave_data))
        pcs_sn_interp = np.array(pcs_sn_interp).T   # (N_wave, N_pc_sn)

        # A 矩阵只包含 SN PCA
        A = pcs_sn_interp

        # --- 线性拟合 ---
        res = lsq_linear(A, flux_data)
        sn_coeffs = res.x

        # -----------------------------
        # 关键点：补齐 host=0，总长度维持 86+20
        # -----------------------------
        HOST_DIM = 86
        SN_DIM = sn_n_components if sn_n_components else pcs_sn_interp.shape[1]

        coeffs = np.concatenate([
            np.zeros(HOST_DIM),  # host 全部填 0
            sn_coeffs            # SN 拟合结果
        ])

        # --- SN 模型 ---
        sn_model = pcs_sn_interp @ sn_coeffs

        # --- 总模型（无 host） ---
        fit_model = sn_model.copy()

        # --- 残差 ---
        residual = flux_data - fit_model
        if denoise:
            residual = wavelet_denoise_1d(residual, wavelet=wavelet, level=4)

        residual_rms = np.sqrt(np.mean(residual**2)) / np.mean(flux_data)

        # --- 可视化 ---
        if visualize:
            mask = (wave_data >= 4000) & (wave_data <= 7000)
            plt.figure(figsize=(10, 6))
            plt.plot(wave_data[mask], flux_data[mask], label="Observed")
            plt.plot(wave_data[mask], fit_model[mask], label="SN Fit", lw=2)
            plt.plot(wave_data[mask], residual[mask], label="Residual")
            if plot_sn_component:
                plt.plot(wave_data[mask], sn_model[mask], "--", label="SN Component")
            plt.legend()
            plt.xlabel("Wavelength")
            plt.ylabel("Flux")
            plt.show()

        return coeffs, fit_model, residual, residual_rms, sn_model

    # ==================================
    # Case 2：transient_only = False
    # 正常模式（host + SN PCA）
    # ==================================

    # --- 读取宿主 PCA ---
    host_template = np.loadtxt(host_template_path)
    wave_host = host_template[:, 0]
    pcs_host = host_template[:, 1:]
    if host_n_components is not None:
        pcs_host = pcs_host[:, :host_n_components]

    # 插值 host PCA
    pcs_host_interp = []
    for i in range(pcs_host.shape[1]):
        f = interp1d(wave_host, pcs_host[:, i], bounds_error=False, fill_value=0)
        pcs_host_interp.append(f(wave_data))
    pcs_host_interp = np.array(pcs_host_interp).T

    # SN PCA
    if sn_template_path is not None:
        sn_template = np.loadtxt(sn_template_path)
        wave_sn = sn_template[:, 0]
        pcs_sn = sn_template[:, 1:]
        if sn_n_components is not None:
            pcs_sn = pcs_sn[:, :sn_n_components]

        pcs_sn_interp = []
        for i in range(pcs_sn.shape[1]):
            f = interp1d(wave_sn, pcs_sn[:, i], bounds_error=False, fill_value=0)
            pcs_sn_interp.append(f(wave_data))
        pcs_sn_interp = np.array(pcs_sn_interp).T

        A = np.hstack([pcs_host_interp, pcs_sn_interp])
    else:
        pcs_sn_interp = None
        A = pcs_host_interp

    res = lsq_linear(A, flux_data)
    coeffs = res.x

    # SN 补齐
    if pcs_sn_interp is None:
        coeffs = np.concatenate([coeffs, np.zeros(sn_n_components)])

    fit_model = A @ res.x
    residual = flux_data - fit_model
    if denoise:
        residual = wavelet_denoise_1d(residual, wavelet=wavelet, level=4)

    host_model = pcs_host_interp @ coeffs[:pcs_host_interp.shape[1]]
    sn_model = None
    if pcs_sn_interp is not None:
        sn_model = pcs_sn_interp @ coeffs[pcs_host_interp.shape[1]:]

    residual_rms = np.sqrt(np.mean(residual**2)) / np.mean(flux_data)

    if visualize:
        mask = (wave_data >= 4000) & (wave_data <= 7000)
        plt.figure(figsize=(10, 6))
        plt.plot(wave_data[mask], flux_data[mask], label="Observed")
        plt.plot(wave_data[mask], fit_model[mask], label="Fit", lw=2)
        plt.plot(wave_data[mask], residual[mask], label="Residual", lw=2)
        if plot_sn_component and (sn_model is not None):
            plt.plot(wave_data[mask], sn_model[mask], "--", label="SN component")
        plt.legend()
        plt.show()

    return coeffs, fit_model, residual, residual_rms, sn_model



if __name__ == "__main__":
    # 示例：从文件读入新数据
    # data = np.loadtxt("your_spectrum.txt")
    
    # 或者外部传入 data (N,2) 数组
    data = np.loadtxt('/Users/astronomy_zrf/Desktop/工作文献2/TDE文献/2025/未命名.txt')
    
    coeffs, host_fit, sn_residual = fit_spectrum(
        data, 
        template_path="/Users/astronomy_zrf/Desktop/工作相关/meansk86.dat", 
        denoise=True, 
        visualize=True
    )