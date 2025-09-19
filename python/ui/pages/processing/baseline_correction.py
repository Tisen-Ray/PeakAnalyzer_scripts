"""
基线校正处理模块
"""
import streamlit as st
import numpy as np
from typing import Dict, Any, Tuple
from scipy import signal
from core.curve import Curve
from core.state_manager import state_manager


class BaselineCorrectionProcessor:
    """基线校正处理器"""
    
    def __init__(self):
        self.methods = {
            '线性': self._linear_baseline,
            '多项式': self._polynomial_baseline,
            '不对称最小二乘': self._asymmetric_least_squares,
            '自适应': self._adaptive_baseline
        }
    
    def render_baseline_correction(self, curve: Curve) -> bool:
        """渲染基线校正界面并执行处理"""
        st.markdown("### 🔧 基线校正")
        
        if not curve or not curve.y_values.size:
            st.warning("请先加载曲线数据")
            return False
        
        # 方法选择
        method = st.selectbox(
            "校正方法",
            options=list(self.methods.keys()),
            key="baseline_method"
        )
        
        # 参数配置
        params = self._render_method_params(method)
        
        # 操作按钮 - 垂直布局
        if st.button("🔧 执行基线校正", key="apply_baseline", width='stretch'):
            return self._execute_baseline_correction_with_confirm(curve, method, params)
        
        if st.button("⏭️ 跳过", key="skip_baseline", width='stretch'):
            st.info("已跳过基线校正")
            return False
        
        return False
    
    def _render_method_params(self, method: str) -> Dict[str, Any]:
        """渲染方法特定参数"""
        params = {}
        
        if method == '线性':
            params['degree'] = st.slider("多项式次数", 1, 10, 1)
            
        elif method == '多项式':
            params['degree'] = st.slider("多项式次数", 2, 15, 3)
            params['robust'] = st.checkbox("使用鲁棒拟合", value=True)
            
        elif method == '不对称最小二乘':
            params['p'] = st.slider("不对称参数", 0.001, 0.5, 0.01, 0.001)
            params['lambda'] = st.slider("平滑参数", 10, 100000, 1000, 10)
            params['max_iter'] = st.slider("最大迭代次数", 10, 1000, 50)
            
        elif method == '自适应':
            params['window_size'] = st.slider("窗口大小", 5, 500, 50)
            params['threshold'] = st.slider("阈值", 0.001, 0.5, 0.05, 0.001)
        
        return params
    
    def _preview_baseline_correction(self, curve: Curve, method: str, params: Dict[str, Any]):
        """预览基线校正效果"""
        try:
            corrected_y = self.methods[method](curve.y_values, params)
            st.success(f"✅ 预览完成 - 方法: {method}")
            st.info(f"原始数据范围: {curve.y_values.min():.0f} - {curve.y_values.max():.0f}")
            st.info(f"校正后范围: {corrected_y.min():.0f} - {corrected_y.max():.0f}")
        except Exception as e:
            st.error(f"❌ 预览失败: {str(e)}")
    
    def _execute_baseline_correction_with_confirm(self, curve: Curve, method: str, params: Dict[str, Any]) -> bool:
        """执行基线校正并允许确认应用"""
        try:
            # 使用session_state管理工作副本
            working_key = f"working_curve_{curve.curve_id}"
            if working_key not in st.session_state:
                st.error("❌ 工作副本未找到，请重新选择曲线")
                return False
            
            # 获取当前工作副本的原始数据
            working_data = st.session_state[working_key]
            original_y = working_data["original_y"].copy()
            
            # 保存原始数据到曲线对象中用于对比显示
            curve._original_y_values = original_y.copy()
            
            # 基于原始数据执行基线校正
            corrected_y = self.methods[method](original_y, params)
            
            # 更新工作副本（临时显示）
            curve.y_values = corrected_y.copy()
            curve.is_baseline_corrected = True
            working_data["is_modified"] = True
            
            # 显示处理结果统计
            st.success(f"✅ 基线校正执行完成 - 方法: {method}")
            st.info(f"原始数据范围: {original_y.min():.0f} - {original_y.max():.0f}")
            st.info(f"校正后范围: {corrected_y.min():.0f} - {corrected_y.max():.0f}")
            
            # 显示当前状态信息（调试用）
            with st.expander("🔍 调试信息", expanded=False):
                stored_curve = state_manager.get_curve(curve.curve_id)
                st.write(f"**当前曲线状态:**")
                st.write(f"- 曲线ID: {curve.curve_id}")
                st.write(f"- 基线校正状态: {curve.is_baseline_corrected}")
                st.write(f"- 存储曲线状态: {stored_curve.is_baseline_corrected}")
                st.write(f"- 当前y值范围: {curve.y_values.min():.0f} - {curve.y_values.max():.0f}")
                st.write(f"- 存储y值范围: {stored_curve.y_values.min():.0f} - {stored_curve.y_values.max():.0f}")
            
            # 确认应用选项
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ 确认应用", key="confirm_baseline"):
                    # 确认应用 - 将工作副本写入存储数据
                    try:
                        # 获取存储的曲线
                        stored_curve = state_manager.get_curve(curve.curve_id)
                        if stored_curve is None:
                            st.error("❌ 无法获取存储的曲线数据")
                            return False
                        
                        # 将工作副本的数据写入存储数据
                        stored_curve.y_values = curve.y_values.copy()
                        stored_curve.is_baseline_corrected = curve.is_baseline_corrected
                        state_manager.update_curve(stored_curve)
                        
                        # 更新工作副本的原始数据为新的存储数据
                        working_data["original_y"] = stored_curve.y_values.copy()
                        working_data["is_modified"] = False
                        working_data["last_applied"] = True
                        
                        # 清除对比数据，不再显示原始曲线虚线
                        curve._original_y_values = None
                        if hasattr(curve, '_original_peaks'):
                            curve._original_peaks = None
                        
                        # 确保工作副本也清除对比数据
                        working_data_curve = working_data["curve"]
                        working_data_curve._original_y_values = None
                        if hasattr(working_data_curve, '_original_peaks'):
                            working_data_curve._original_peaks = None
                        
                        st.success(f"✅ 基线校正已确认应用 (方法: {method})")
                        st.rerun()
                        return True
                        
                    except Exception as e:
                        st.error(f"❌ 确认应用失败: {str(e)}")
                        return False
            
            with col2:
                if st.button("❌ 撤销", key="cancel_baseline"):
                    # 撤销操作 - 恢复到工作副本的原始数据
                    curve.y_values = original_y.copy()
                    curve.is_baseline_corrected = False
                    curve._original_y_values = None  # 清除对比数据
                    working_data["is_modified"] = False
                    st.info("已撤销基线校正，恢复到原始数据")
                    st.rerun()
                    return False
            
            return False
            
        except Exception as e:
            st.error(f"❌ 基线校正失败: {str(e)}")
            return False
    
    def _linear_baseline(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """线性基线校正"""
        degree = params.get('degree', 1)
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        return y - baseline
    
    def _polynomial_baseline(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """多项式基线校正"""
        degree = params.get('degree', 3)
        robust = params.get('robust', True)
        x = np.arange(len(y))
        
        if robust:
            # 使用鲁棒拟合
            from scipy.optimize import curve_fit
            
            def poly_func(x, *coeffs):
                return sum(c * x**i for i, c in enumerate(coeffs))
            
            # 初始猜测
            p0 = [0] * (degree + 1)
            p0[0] = np.mean(y)
            
            try:
                coeffs, _ = curve_fit(poly_func, x, y, p0=p0)
                baseline = poly_func(x, *coeffs)
            except:
                # 如果鲁棒拟合失败，使用普通拟合
                coeffs = np.polyfit(x, y, degree)
                baseline = np.polyval(coeffs, x)
        else:
            coeffs = np.polyfit(x, y, degree)
            baseline = np.polyval(coeffs, x)
        
        return y - baseline
    
    def _asymmetric_least_squares(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """不对称最小二乘基线校正"""
        p = params.get('p', 0.01)
        lam = params.get('lambda', 1000)
        max_iter = params.get('max_iter', 50)
        
        # 简化的ALS算法
        baseline = np.zeros_like(y)
        weights = np.ones_like(y)
        
        for _ in range(max_iter):
            # 计算加权最小二乘
            x = np.arange(len(y))
            A = np.vstack([x**i for i in range(4)]).T  # 三次多项式
            W = np.diag(weights)
            
            try:
                coeffs = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
                baseline = A @ coeffs
            except:
                break
            
            # 更新权重
            diff = y - baseline
            weights = np.where(diff > 0, p, 1 - p)
        
        return y - baseline
    
    def _adaptive_baseline(self, y: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """自适应基线校正"""
        window_size = params.get('window_size', 50)
        threshold = params.get('threshold', 0.05)
        
        # 使用移动最小值作为基线
        baseline = np.zeros_like(y)
        
        for i in range(len(y)):
            start = max(0, i - window_size // 2)
            end = min(len(y), i + window_size // 2)
            window_data = y[start:end]
            
            # 使用分位数作为基线估计
            baseline[i] = np.percentile(window_data, threshold * 100)
        
        return y - baseline
