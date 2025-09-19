use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 简化的曲线数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Curve {
    /// 曲线唯一标识符
    pub id: String,
    /// 曲线类型 ("TIC", "EIC", "BPC")
    pub curve_type: String,
    
    /// X轴数据点 (通常是时间)
    pub x_values: Vec<f64>,
    /// Y轴数据点 (通常是强度)
    pub y_values: Vec<f64>,
    
    /// 轴标签
    pub x_label: String,
    pub y_label: String,
    pub x_unit: String,
    pub y_unit: String,
    
    /// m/z范围 (用于EIC/BPC曲线)
    pub mz_range: Option<(f64, f64)>,
    /// MS级别
    pub ms_level: Option<u8>,
    
    /// 元数据
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Curve {
    /// 创建新曲线
    pub fn new(
        id: String,
        curve_type: String,
        x_values: Vec<f64>,
        y_values: Vec<f64>,
        x_label: String,
        y_label: String,
        x_unit: String,
        y_unit: String,
    ) -> Self {
        Self {
            id,
            curve_type,
            x_values,
            y_values,
            x_label,
            y_label,
            x_unit,
            y_unit,
            mz_range: None,
            ms_level: None,
            metadata: HashMap::new(),
        }
    }
    
    /// 设置m/z范围
    pub fn set_mz_range(&mut self, mz_min: f64, mz_max: f64) {
        self.mz_range = Some((mz_min, mz_max));
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }
    
    /// 计算曲线下面积
    pub fn calculate_area(&self) -> f64 {
        if self.x_values.len() < 2 {
            return 0.0;
        }
        
        let mut area = 0.0;
        for i in 1..self.x_values.len() {
            let dx = self.x_values[i] - self.x_values[i - 1];
            let avg_y = (self.y_values[i] + self.y_values[i - 1]) / 2.0;
            area += dx * avg_y;
        }
        area
    }
    
    /// 获取最大强度
    pub fn max_intensity(&self) -> f64 {
        self.y_values.iter().fold(0.0_f64, |a, &b| a.max(b))
    }
    
    /// 获取最小强度
    pub fn min_intensity(&self) -> f64 {
        self.y_values.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }
}