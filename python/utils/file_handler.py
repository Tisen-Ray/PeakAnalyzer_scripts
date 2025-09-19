"""
文件处理工具 - 处理文件上传、验证等
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib


class FileHandler:
    """文件处理类"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "peakanalyzer_temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.supported_formats = {
            '.mzml': 'mzML',
            '.mzxml': 'mzXML', 
            '.mgf': 'MGF',
            '.raw': 'Thermo RAW',
            '.d': 'Agilent .d',
            '.wiff': 'SCIEX WIFF'
        }
    
    def validate_file_format(self, filename: str) -> bool:
        """验证文件格式是否支持"""
        ext = Path(filename).suffix.lower()
        return ext in self.supported_formats
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """获取文件信息"""
        path = Path(filename)
        ext = path.suffix.lower()
        
        return {
            'name': path.name,
            'stem': path.stem,
            'extension': ext,
            'format': self.supported_formats.get(ext, 'Unknown'),
            'is_supported': ext in self.supported_formats
        }
    
    def save_uploaded_file(self, uploaded_file, prefix: str = "upload") -> str:
        """保存上传的文件到临时目录"""
        # 生成唯一文件名
        file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
        temp_filename = f"{prefix}_{file_hash}_{uploaded_file.name}"
        temp_path = self.temp_dir / temp_filename
        
        # 保存文件
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        return str(temp_path)
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理临时文件"""
        import time
        current_time = time.time()
        
        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_hours * 3600:  # 转换为秒
                    try:
                        file_path.unlink()
                    except:
                        pass  # 忽略删除失败的文件
    
    def get_supported_formats(self) -> Dict[str, str]:
        """获取支持的文件格式"""
        return self.supported_formats.copy()
    
    def batch_validate_files(self, filenames: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量验证文件"""
        results = {}
        for filename in filenames:
            results[filename] = self.get_file_info(filename)
        return results
