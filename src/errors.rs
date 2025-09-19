use std::error::Error;
use std::fmt;

/// 统一的错误类型
#[derive(Debug)]
pub enum PeakAnalyzerError {
    /// 文件IO错误
    IoError(std::io::Error),
    /// mzdata相关错误
    MzDataError(String),
    /// 数据处理错误
    DataError(String),
    /// 配置错误
    ConfigError(String),
    /// 提取错误
    ExtractionError(String),
    /// Python绑定错误
    PythonError(String),
}

impl fmt::Display for PeakAnalyzerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PeakAnalyzerError::IoError(e) => write!(f, "IO错误: {}", e),
            PeakAnalyzerError::MzDataError(e) => write!(f, "mzdata错误: {}", e),
            PeakAnalyzerError::DataError(e) => write!(f, "数据错误: {}", e),
            PeakAnalyzerError::ConfigError(e) => write!(f, "配置错误: {}", e),
            PeakAnalyzerError::ExtractionError(e) => write!(f, "提取错误: {}", e),
            PeakAnalyzerError::PythonError(e) => write!(f, "Python绑定错误: {}", e),
        }
    }
}

impl Error for PeakAnalyzerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PeakAnalyzerError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PeakAnalyzerError {
    fn from(err: std::io::Error) -> Self {
        PeakAnalyzerError::IoError(err)
    }
}

/// 简化的Result类型
pub type PeakAnalyzerResult<T> = Result<T, PeakAnalyzerError>;

impl PeakAnalyzerError {
    /// 创建mzdata错误
    pub fn mzdata_error<S: Into<String>>(message: S) -> Self {
        PeakAnalyzerError::MzDataError(message.into())
    }
    
    /// 创建数据错误
    pub fn data_error<S: Into<String>>(message: S) -> Self {
        PeakAnalyzerError::DataError(message.into())
    }
    
    /// 创建配置错误
    pub fn config_error<S: Into<String>>(message: S) -> Self {
        PeakAnalyzerError::ConfigError(message.into())
    }
    
    /// 创建提取错误
    pub fn extraction_error<S: Into<String>>(message: S) -> Self {
        PeakAnalyzerError::ExtractionError(message.into())
    }
    
    /// 创建Python错误
    pub fn python_error<S: Into<String>>(message: S) -> Self {
        PeakAnalyzerError::PythonError(message.into())
    }
}
