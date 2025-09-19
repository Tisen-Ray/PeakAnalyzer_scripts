"""
报告生成器 - 生成各种格式的分析报告
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from core.curve import Curve, Peak


class ReportGenerator:
    """报告生成器 - 支持多种导出格式"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        初始化报告生成器
        
        参数:
        - output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.csv_dir = self.output_dir / "csv"
        self.excel_dir = self.output_dir / "excel"
        self.json_dir = self.output_dir / "json"
        self.pdf_dir = self.output_dir / "pdf"
        
        for dir_path in [self.csv_dir, self.excel_dir, self.json_dir, self.pdf_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def export_to_csv(self, curves: List[Curve], options: Dict[str, Any]) -> str:
        """导出为CSV格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 曲线数据CSV
        curves_file = self.csv_dir / f"curves_{timestamp}.csv"
        self._export_curves_to_csv(curves, curves_file, options)
        
        # 峰数据CSV
        if options.get('include_peaks', True):
            peaks_file = self.csv_dir / f"peaks_{timestamp}.csv"
            self._export_peaks_to_csv(curves, peaks_file, options)
        
        return str(curves_file)
    
    def export_to_excel(self, curves: List[Curve], options: Dict[str, Any]) -> str:
        """导出为Excel格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_file = self.excel_dir / f"peak_analysis_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 曲线摘要
                summary_df = self._create_curves_summary_df(curves)
                summary_df.to_excel(writer, sheet_name='曲线摘要', index=False)
                
                # 峰数据
                if options.get('include_peaks', True):
                    peaks_df = self._create_peaks_df(curves)
                    if not peaks_df.empty:
                        peaks_df.to_excel(writer, sheet_name='峰数据', index=False)
                
                # 详细曲线数据 (采样后的数据以减小文件大小)
                for i, curve in enumerate(curves[:5]):  # 最多导出5条曲线的详细数据
                    curve_df = self._create_curve_data_df(curve, sample_points=1000)
                    sheet_name = f"曲线{i+1}_{curve.curve_type}"[:31]  # Excel工作表名长度限制
                    curve_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 元数据
                if options.get('include_metadata', True):
                    metadata_df = self._create_metadata_df(curves)
                    metadata_df.to_excel(writer, sheet_name='元数据', index=False)
        
        except Exception as e:
            # 如果openpyxl不可用，使用xlsxwriter
            try:
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    summary_df = self._create_curves_summary_df(curves)
                    summary_df.to_excel(writer, sheet_name='曲线摘要', index=False)
            except:
                # 如果都不可用，返回CSV文件
                return self.export_to_csv(curves, options)
        
        return str(excel_file)
    
    def export_to_json(self, curves: List[Curve], options: Dict[str, Any]) -> str:
        """导出为JSON格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.json_dir / f"peak_analysis_{timestamp}.json"
        
        export_data = {
            'export_info': {
                'timestamp': timestamp,
                'version': '1.0',
                'curve_count': len(curves),
                'options': options
            },
            'curves': []
        }
        
        for curve in curves:
            curve_data = {
                'curve_id': curve.curve_id,
                'curve_type': curve.curve_type,
                'metadata': curve.metadata,
                'statistics': {
                    'data_points': len(curve.x_values),
                    'rt_range': curve.x_range,
                    'intensity_range': curve.y_range,
                    'max_intensity': curve.max_intensity,
                    'total_area': curve.total_area
                }
            }
            
            # 采样数据点以减小文件大小
            if len(curve.x_values) > 2000:
                indices = np.linspace(0, len(curve.x_values)-1, 2000, dtype=int)
                curve_data['x_values'] = curve.x_values[indices].tolist()
                curve_data['y_values'] = curve.y_values[indices].tolist()
            else:
                curve_data['x_values'] = curve.x_values.tolist()
                curve_data['y_values'] = curve.y_values.tolist()
            
            # 峰数据
            if options.get('include_peaks', True) and curve.peaks:
                curve_data['peaks'] = [peak.to_dict() for peak in curve.peaks]
            
            # 处理历史
            if options.get('include_processing_history', False):
                curve_data['processing_history'] = curve.processing_history
            
            export_data['curves'].append(curve_data)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_file)
    
    def export_to_pdf(self, curves: List[Curve], options: Dict[str, Any]) -> str:
        """导出为PDF报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file = self.pdf_dir / f"peak_analysis_report_{timestamp}.pdf"
        
        try:
            # 尝试使用matplotlib生成简单的PDF报告
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            
            with PdfPages(pdf_file) as pdf:
                # 封面页
                fig, ax = plt.subplots(figsize=(8, 11))
                ax.text(0.5, 0.7, 'Peak Analysis Report', 
                       ha='center', va='center', fontsize=24, fontweight='bold')
                ax.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                       ha='center', va='center', fontsize=12)
                ax.text(0.5, 0.5, f'Curves: {len(curves)}', 
                       ha='center', va='center', fontsize=12)
                
                total_peaks = sum(len(curve.peaks) for curve in curves)
                ax.text(0.5, 0.4, f'Total Peaks: {total_peaks}', 
                       ha='center', va='center', fontsize=12)
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # 曲线图表
                if options.get('include_plots', True):
                    for i, curve in enumerate(curves[:10]):  # 最多10条曲线
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 绘制曲线
                        ax.plot(curve.x_values, curve.y_values, 'b-', linewidth=1)
                        
                        # 标注峰
                        if curve.peaks:
                            peak_x = [peak.rt for peak in curve.peaks]
                            peak_y = [peak.intensity for peak in curve.peaks]
                            ax.plot(peak_x, peak_y, 'ro', markersize=6)
                            
                            # 标注峰号
                            for j, peak in enumerate(curve.peaks):
                                ax.annotate(f'P{j+1}', (peak.rt, peak.intensity), 
                                          xytext=(5, 5), textcoords='offset points',
                                          fontsize=8)
                        
                        ax.set_xlabel('Retention Time (min)')
                        ax.set_ylabel('Intensity')
                        ax.set_title(f'{curve.curve_type} - {curve.metadata.get("original_filename", curve.curve_id)}')
                        ax.grid(True, alpha=0.3)
                        
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                
                # 统计表格
                if options.get('include_statistics', True):
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 创建统计表格
                    table_data = []
                    headers = ['Curve', 'Type', 'Data Points', 'RT Range', 'Max Intensity', 'Peaks']
                    
                    for curve in curves:
                        row = [
                            curve.metadata.get('original_filename', curve.curve_id)[:20],
                            curve.curve_type,
                            len(curve.x_values),
                            f"{curve.x_range[0]:.1f}-{curve.x_range[1]:.1f}",
                            f"{curve.max_intensity:.0f}",
                            len(curve.peaks)
                        ]
                        table_data.append(row)
                    
                    table = ax.table(cellText=table_data, colLabels=headers,
                                   cellLoc='center', loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1.2, 1.5)
                    
                    ax.set_title('Curves Statistics', fontsize=16, fontweight='bold', pad=20)
                    ax.axis('off')
                    
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            
            return str(pdf_file)
            
        except ImportError:
            # 如果matplotlib不可用，创建一个简单的文本报告
            txt_file = pdf_file.with_suffix('.txt')
            self._create_text_report(curves, options, txt_file)
            return str(txt_file)
        except Exception as e:
            # 发生错误时，创建文本报告
            txt_file = pdf_file.with_suffix('.txt')
            self._create_text_report(curves, options, txt_file)
            return str(txt_file)
    
    def _export_curves_to_csv(self, curves: List[Curve], file_path: Path, options: Dict[str, Any]):
        """导出曲线数据到CSV"""
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入标题行
            headers = ['Curve_ID', 'Curve_Type', 'RT_min', 'Intensity', 'File_Name']
            if options.get('include_metadata', True):
                headers.extend(['Metadata_Keys', 'Processing_Steps'])
            writer.writerow(headers)
            
            # 写入数据
            for curve in curves:
                # 采样数据以减小文件大小
                if len(curve.x_values) > 5000:
                    indices = np.linspace(0, len(curve.x_values)-1, 5000, dtype=int)
                    x_data = curve.x_values[indices]
                    y_data = curve.y_values[indices]
                else:
                    x_data = curve.x_values
                    y_data = curve.y_values
                
                for x, y in zip(x_data, y_data):
                    row = [
                        curve.curve_id,
                        curve.curve_type,
                        f"{x:.4f}",
                        f"{y:.2f}",
                        curve.metadata.get('original_filename', '')
                    ]
                    
                    if options.get('include_metadata', True):
                        metadata_keys = ';'.join(curve.metadata.keys())
                        processing_steps = ';'.join([step['step'] for step in curve.processing_history])
                        row.extend([metadata_keys, processing_steps])
                    
                    writer.writerow(row)
    
    def _export_peaks_to_csv(self, curves: List[Curve], file_path: Path, options: Dict[str, Any]):
        """导出峰数据到CSV"""
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入标题行
            headers = [
                'Curve_ID', 'Peak_ID', 'Peak_Number', 'RT_min', 'RT_Start', 'RT_End',
                'Intensity', 'Area', 'Height', 'FWHM', 'Signal_to_Noise', 'Confidence',
                'File_Name'
            ]
            writer.writerow(headers)
            
            # 写入峰数据
            for curve in curves:
                for i, peak in enumerate(curve.peaks, 1):
                    row = [
                        curve.curve_id,
                        peak.peak_id,
                        i,
                        f"{peak.rt:.4f}",
                        f"{peak.rt_start:.4f}",
                        f"{peak.rt_end:.4f}",
                        f"{peak.intensity:.2f}",
                        f"{peak.area:.2e}",
                        f"{peak.height:.2f}",
                        f"{peak.fwhm:.4f}",
                        f"{peak.signal_to_noise:.2f}",
                        f"{peak.confidence:.3f}",
                        curve.metadata.get('original_filename', '')
                    ]
                    writer.writerow(row)
    
    def _create_curves_summary_df(self, curves: List[Curve]) -> pd.DataFrame:
        """创建曲线摘要DataFrame"""
        data = []
        for curve in curves:
            data.append({
                '曲线ID': curve.curve_id,
                '曲线类型': curve.curve_type,
                '文件名': curve.metadata.get('original_filename', ''),
                '数据点数': len(curve.x_values),
                'RT范围_分钟': f"{curve.x_range[0]:.2f} - {curve.x_range[1]:.2f}",
                '最大强度': f"{curve.max_intensity:.0f}",
                '最小强度': f"{curve.min_intensity:.0f}",
                '总面积': f"{curve.total_area:.2e}",
                '峰数量': len(curve.peaks),
                '是否已处理': '是' if curve.processing_history else '否',
                '创建时间': curve.created_at.strftime('%Y-%m-%d %H:%M:%S') if curve.created_at else ''
            })
        
        return pd.DataFrame(data)
    
    def _create_peaks_df(self, curves: List[Curve]) -> pd.DataFrame:
        """创建峰数据DataFrame"""
        data = []
        for curve in curves:
            for i, peak in enumerate(curve.peaks, 1):
                data.append({
                    '曲线ID': curve.curve_id,
                    '峰号': i,
                    '峰ID': peak.peak_id,
                    'RT_分钟': f"{peak.rt:.4f}",
                    'RT开始': f"{peak.rt_start:.4f}",
                    'RT结束': f"{peak.rt_end:.4f}",
                    '强度': f"{peak.intensity:.2f}",
                    '面积': f"{peak.area:.2e}",
                    '峰高': f"{peak.height:.2f}",
                    'FWHM': f"{peak.fwhm:.4f}",
                    '信噪比': f"{peak.signal_to_noise:.2f}",
                    '置信度': f"{peak.confidence:.3f}",
                    '文件名': curve.metadata.get('original_filename', '')
                })
        
        return pd.DataFrame(data)
    
    def _create_curve_data_df(self, curve: Curve, sample_points: int = 1000) -> pd.DataFrame:
        """创建曲线数据DataFrame"""
        # 采样数据
        if len(curve.x_values) > sample_points:
            indices = np.linspace(0, len(curve.x_values)-1, sample_points, dtype=int)
            x_data = curve.x_values[indices]
            y_data = curve.y_values[indices]
        else:
            x_data = curve.x_values
            y_data = curve.y_values
        
        return pd.DataFrame({
            'RT_分钟': x_data,
            '强度': y_data
        })
    
    def _create_metadata_df(self, curves: List[Curve]) -> pd.DataFrame:
        """创建元数据DataFrame"""
        data = []
        for curve in curves:
            for key, value in curve.metadata.items():
                data.append({
                    '曲线ID': curve.curve_id,
                    '元数据键': key,
                    '元数据值': str(value)
                })
        
        return pd.DataFrame(data)
    
    def _create_text_report(self, curves: List[Curve], options: Dict[str, Any], file_path: Path):
        """创建文本格式报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Peak Analysis Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Curves: {len(curves)}\n")
            f.write(f"Total Peaks: {sum(len(curve.peaks) for curve in curves)}\n\n")
            
            for i, curve in enumerate(curves, 1):
                f.write(f"Curve {i}: {curve.curve_type}\n")
                f.write("-" * 30 + "\n")
                f.write(f"File: {curve.metadata.get('original_filename', 'Unknown')}\n")
                f.write(f"Data Points: {len(curve.x_values)}\n")
                f.write(f"RT Range: {curve.x_range[0]:.2f} - {curve.x_range[1]:.2f} min\n")
                f.write(f"Intensity Range: {curve.y_range[0]:.0f} - {curve.y_range[1]:.0f}\n")
                f.write(f"Total Area: {curve.total_area:.2e}\n")
                f.write(f"Peaks Detected: {len(curve.peaks)}\n")
                
                if curve.peaks:
                    f.write("\nPeaks:\n")
                    for j, peak in enumerate(curve.peaks, 1):
                        f.write(f"  Peak {j}: RT={peak.rt:.2f} min, "
                               f"Intensity={peak.intensity:.0f}, "
                               f"Area={peak.area:.2e}, "
                               f"FWHM={peak.fwhm:.2f}\n")
                
                f.write("\n")
    
    def get_export_stats(self) -> Dict[str, Any]:
        """获取导出统计信息"""
        stats = {
            'output_dir': str(self.output_dir),
            'csv_files': len(list(self.csv_dir.glob("*.csv"))),
            'excel_files': len(list(self.excel_dir.glob("*.xlsx"))),
            'json_files': len(list(self.json_dir.glob("*.json"))),
            'pdf_files': len(list(self.pdf_dir.glob("*.pdf"))) + len(list(self.pdf_dir.glob("*.txt")))
        }
        
        # 计算总大小
        total_size = 0
        for dir_path in [self.csv_dir, self.excel_dir, self.json_dir, self.pdf_dir]:
            for file_path in dir_path.glob("*"):
                try:
                    total_size += file_path.stat().st_size
                except:
                    pass
        
        stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return stats
