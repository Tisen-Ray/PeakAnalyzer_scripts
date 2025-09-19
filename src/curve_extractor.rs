use mzdata::prelude::*;
use mzdata::spectrum::Spectrum;
use std::collections::HashMap;
use uuid::Uuid;
use crate::curve::Curve;
use crate::errors::{PeakAnalyzerError, PeakAnalyzerResult};

/// 曲线类型枚举
#[derive(Debug, Clone)]
pub enum CurveType {
    TIC, // Total Ion Current
    EIC, // Extracted Ion Current
    BPC, // Base Peak Chromatogram
}

impl CurveType {
    pub fn as_str(&self) -> &str {
        match self {
            CurveType::TIC => "TIC",
            CurveType::EIC => "EIC", 
            CurveType::BPC => "BPC",
        }
    }
}

/// 提取器配置
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    pub mz_range: (f64, f64),
    pub rt_range: Option<(f64, f64)>,
    pub ms_level: Option<u8>,
}

impl ExtractorConfig {
    /// 创建新的配置
    pub fn new(mz_min: f64, mz_max: f64) -> Self {
        Self {
            mz_range: (mz_min, mz_max),
            rt_range: None,
            ms_level: None,
        }
    }
    
    /// 设置RT范围
    pub fn with_rt_range(mut self, rt_min: f64, rt_max: f64) -> Self {
        self.rt_range = Some((rt_min, rt_max));
        self
    }
    
    /// 设置MS级别
    pub fn with_ms_level(mut self, ms_level: u8) -> Self {
        self.ms_level = Some(ms_level);
        self
    }
}

/// 曲线提取器
pub struct CurveExtractor;

impl CurveExtractor {
    pub fn new() -> Self {
        Self
    }
    
    /// 提取曲线数据
    pub fn extract_curve(
        &self,
        spectra: &[Spectrum],
        curve_type: CurveType,
        config: &ExtractorConfig,
    ) -> PeakAnalyzerResult<Curve> {
        match curve_type {
            CurveType::TIC => self.extract_tic_curve(spectra, config),
            CurveType::EIC => self.extract_eic_curve(spectra, config),
            CurveType::BPC => self.extract_bpc_curve(spectra, config),
        }
    }
    
    /// 提取TIC曲线
    fn extract_tic_curve(
        &self,
        spectra: &[Spectrum],
        config: &ExtractorConfig,
    ) -> PeakAnalyzerResult<Curve> {
        let mut rt_data: HashMap<u64, f64> = HashMap::new();
        
        for spectrum in spectra {
            // 检查MS级别
            if let Some(level) = config.ms_level {
                if spectrum.ms_level() != level {
                    continue;
                }
            }
            
            let rt = spectrum.start_time();
            
            // 检查RT范围
            if let Some((rt_min, rt_max)) = config.rt_range {
                if rt < rt_min || rt > rt_max {
                    continue;
                }
            }
            
            let rt_key = (rt * 1000.0) as u64; // 精确到毫秒
            let peaks = spectrum.peaks();
            
            // 累加指定m/z范围内的强度
            let mut total_intensity = 0.0;
            for peak in peaks.iter() {
                let mz = peak.mz();
                if mz >= config.mz_range.0 && mz <= config.mz_range.1 {
                    total_intensity += peak.intensity() as f64;
                }
            }
            
            *rt_data.entry(rt_key).or_insert(0.0) += total_intensity;
        }
        
        if rt_data.is_empty() {
            return Err(PeakAnalyzerError::extraction_error("在指定范围内没有找到数据点"));
        }
        
        // 排序并生成曲线数据
        let mut sorted_data: Vec<(u64, f64)> = rt_data.into_iter().collect();
        sorted_data.sort_by(|a, b| a.0.cmp(&b.0));
        
        let x_values: Vec<f64> = sorted_data.iter().map(|(k, _)| *k as f64 / 1000.0).collect();
        let y_values: Vec<f64> = sorted_data.iter().map(|(_, v)| *v).collect();
        
        // 创建Curve对象
        let mut curve = Curve::new(
            format!("tic_{}", Uuid::new_v4()),
            "TIC".to_string(),
            x_values,
            y_values,
            "Retention Time".to_string(),
            "Intensity".to_string(),
            "min".to_string(),
            "counts".to_string(),
        );
        
        // 设置MS相关参数
        curve.set_mz_range(config.mz_range.0, config.mz_range.1);
        if let Some((rt_min, rt_max)) = config.rt_range {
            curve.set_rt_range(rt_min, rt_max);
        }
        curve.ms_level = config.ms_level;
        
        // 添加元数据
        curve.add_metadata("extraction_type".to_string(), serde_json::json!("TIC"));
        curve.add_metadata("mz_range".to_string(), serde_json::json!([config.mz_range.0, config.mz_range.1]));
        if let Some(level) = config.ms_level {
            curve.add_metadata("ms_level".to_string(), serde_json::json!(level));
        }
        
        Ok(curve)
    }
    
    /// 提取EIC曲线（与TIC相同的实现，但标记为EIC类型）
    fn extract_eic_curve(
        &self,
        spectra: &[Spectrum],
        config: &ExtractorConfig,
    ) -> PeakAnalyzerResult<Curve> {
        let mut curve = self.extract_tic_curve(spectra, config)?;
        
        // 更新曲线类型和ID
        curve.curve_type = "EIC".to_string();
        curve.id = format!("eic_{}", Uuid::new_v4());
        curve.add_metadata("extraction_type".to_string(), serde_json::json!("EIC"));
        
        Ok(curve)
    }
    
    /// 提取BPC曲线
    fn extract_bpc_curve(
        &self,
        spectra: &[Spectrum],
        config: &ExtractorConfig,
    ) -> PeakAnalyzerResult<Curve> {
        let mut rt_data: HashMap<u64, f64> = HashMap::new();
        
        for spectrum in spectra {
            // 检查MS级别
            if let Some(level) = config.ms_level {
                if spectrum.ms_level() != level {
                    continue;
                }
            }
            
            let rt = spectrum.start_time();
            
            // 检查RT范围
            if let Some((rt_min, rt_max)) = config.rt_range {
                if rt < rt_min || rt > rt_max {
                    continue;
                }
            }
            
            let rt_key = (rt * 1000.0) as u64; // 精确到毫秒
            let peaks = spectrum.peaks();
            
            // 找到指定m/z范围内的最高强度峰
            let mut max_intensity = 0.0;
            for peak in peaks.iter() {
                let mz = peak.mz();
                if mz >= config.mz_range.0 && mz <= config.mz_range.1 {
                    let intensity = peak.intensity() as f64;
                    if intensity > max_intensity {
                        max_intensity = intensity;
                    }
                }
            }
            
            if max_intensity > 0.0 {
                rt_data.insert(rt_key, max_intensity);
            }
        }
        
        if rt_data.is_empty() {
            return Err(PeakAnalyzerError::extraction_error("在指定范围内没有找到数据点"));
        }
        
        // 排序并生成曲线数据
        let mut sorted_data: Vec<(u64, f64)> = rt_data.into_iter().collect();
        sorted_data.sort_by(|a, b| a.0.cmp(&b.0));
        
        let x_values: Vec<f64> = sorted_data.iter().map(|(k, _)| *k as f64 / 1000.0).collect();
        let y_values: Vec<f64> = sorted_data.iter().map(|(_, v)| *v).collect();
        
        // 创建Curve对象
        let mut curve = Curve::new(
            format!("bpc_{}", Uuid::new_v4()),
            "BPC".to_string(),
            x_values,
            y_values,
            "Retention Time".to_string(),
            "Intensity".to_string(),
            "min".to_string(),
            "counts".to_string(),
        );
        
        // 设置MS相关参数
        curve.set_mz_range(config.mz_range.0, config.mz_range.1);
        if let Some((rt_min, rt_max)) = config.rt_range {
            curve.set_rt_range(rt_min, rt_max);
        }
        curve.ms_level = config.ms_level;
        
        // 添加元数据
        curve.add_metadata("extraction_type".to_string(), serde_json::json!("BPC"));
        curve.add_metadata("mz_range".to_string(), serde_json::json!([config.mz_range.0, config.mz_range.1]));
        if let Some(level) = config.ms_level {
            curve.add_metadata("ms_level".to_string(), serde_json::json!(level));
        }
        
        Ok(curve)
    }
}