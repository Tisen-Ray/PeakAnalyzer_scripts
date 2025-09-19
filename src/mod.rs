/// 数据结构模块
pub mod curve;

/// 数据加载模块 - 使用mzdata读取MS数据
pub mod data_loader;

/// 曲线提取模块 - 从光谱数据提取曲线
pub mod curve_extractor;

/// 错误处理模块
pub mod errors;

// 重新导出主要类型
pub use curve::Curve;
pub use data_loader::MzDataLoader;
pub use curve_extractor::{CurveExtractor, CurveType, ExtractorConfig};
pub use errors::{PeakAnalyzerError, PeakAnalyzerResult};
