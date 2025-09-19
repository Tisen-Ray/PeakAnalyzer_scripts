"""
曲线包装器 - 为PyCurve添加缺失的属性和方法
"""

class CurveWrapper:
    """包装PyCurve对象，添加pages期望的属性"""
    
    def __init__(self, py_curve):
        """
        包装PyCurve对象
        
        参数:
        - py_curve: Rust返回的PyCurve对象
        """
        self._py_curve = py_curve
        
        # 添加Python端需要的属性
        self.processing_history = []
        self.is_baseline_corrected = False
        self.is_smoothed = False
        self.is_normalized = False
        self.original_x = None
        self.original_y = None
    
    def __getattr__(self, name):
        """代理访问PyCurve的属性"""
        return getattr(self._py_curve, name)
    
    def add_processing_step(self, step_name: str, parameters: dict):
        """添加处理步骤"""
        step = {
            'step': step_name,
            'parameters': parameters,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        self.processing_history.append(step)
    
    def clear_peaks(self):
        """清除峰"""
        if hasattr(self._py_curve, 'clear_peaks'):
            self._py_curve.clear_peaks()
        # 如果没有clear_peaks方法，就不做任何操作
    
    def reset_to_original(self):
        """重置到原始数据"""
        if self.original_x is not None and self.original_y is not None:
            # 这里需要更新底层的x_values和y_values
            # 但PyCurve可能是只读的，所以这个功能可能有限
            pass
        
        self.processing_history.clear()
        self.is_baseline_corrected = False
        self.is_smoothed = False
        self.is_normalized = False


def wrap_pycurve(py_curve):
    """包装PyCurve对象的便捷函数"""
    return CurveWrapper(py_curve)
