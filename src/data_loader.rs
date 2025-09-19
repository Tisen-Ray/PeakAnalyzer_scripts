use mzdata::prelude::*;
use mzdata::MZReader;
use mzdata::spectrum::Spectrum;
use crate::errors::{PeakAnalyzerError, PeakAnalyzerResult};

/// 简化的mzdata数据加载器
pub struct MzDataLoader;

impl MzDataLoader {
    pub fn new() -> Self {
        Self
    }
    
    /// 从文件加载光谱数据
    pub fn load_from_file(&mut self, path: &str) -> PeakAnalyzerResult<Vec<Spectrum>> {
        let reader = MZReader::open_path(path)
            .map_err(|e| PeakAnalyzerError::mzdata_error(format!("无法打开文件 {}: {}", path, e)))?;
        
        let mut spectra = Vec::new();
        
        for spectrum in reader {
            spectra.push(spectrum);
        }
        
        if spectra.is_empty() {
            return Err(PeakAnalyzerError::data_error("文件中没有找到光谱数据"));
        }
        
        Ok(spectra)
    }
    
    /// 过滤光谱数据
    pub fn filter_spectra<'a>(
        spectra: &'a [Spectrum],
        ms_level: Option<u8>,
        rt_min: Option<f64>,
        rt_max: Option<f64>,
        mz_min: Option<f64>,
        mz_max: Option<f64>,
    ) -> Vec<&'a Spectrum> {
        spectra.iter()
            .filter(|s| {
                // MS级别过滤
                if let Some(level) = ms_level {
                    if s.ms_level() != level {
                        return false;
                    }
                }
                
                // 保留时间过滤
                if let Some(min) = rt_min {
                    if s.start_time() < min {
                        return false;
                    }
                }
                if let Some(max) = rt_max {
                    if s.start_time() > max {
                        return false;
                    }
                }
                
                // m/z范围过滤（检查是否有数据在指定范围内）
                if let (Some(min), Some(max)) = (mz_min, mz_max) {
                    let peaks = s.peaks();
                    let has_data_in_range = peaks.iter().any(|peak| {
                        let mz = peak.mz();
                        mz >= min && mz <= max
                    });
                    if !has_data_in_range {
                        return false;
                    }
                }
                
                true
            })
            .collect()
    }
}
