use derive_more::Display;
use stark_engine_macros::stark_config;

use crate::data_type::DataType;

#[repr(C)]
#[derive(Display, Copy, Eq, Hash)]
#[stark_config]
pub enum QuantizationMode {
    #[serde(rename = "uint4")]
    U4,
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "uint8")]
    U8,
    #[serde(rename = "uint5")]
    U5,
    #[serde(rename = "uint6")]
    U6,
    // Internal coarse proposal storage; model specs never admit this width.
    #[serde(rename = "uint3")]
    U3,
}

impl QuantizationMode {
    pub const fn bits(&self) -> u32 {
        match self {
            Self::U3 => 3,
            Self::U4 => 4,
            Self::U5 => 5,
            Self::U6 => 6,
            Self::I8 | Self::U8 => 8,
        }
    }

    /// Physical byte width of a tightly packed row. New affine widths require
    /// complete groups at admission, so each row starts on a byte boundary.
    pub fn packed_row_bytes(
        &self,
        columns: u32,
    ) -> Option<u32> {
        columns.checked_mul(self.bits()).map(|bits| bits.div_ceil(8))
    }

    pub fn packing_divisor(&self) -> u32 {
        let bits = self.bits();
        assert_eq!(8 % bits, 0, "QuantizationMode bit width ({bits}) must divide 8 evenly");
        8 / bits
    }

    pub fn storage_type(&self) -> DataType {
        match self {
            QuantizationMode::U3 | QuantizationMode::U4 | QuantizationMode::U5 | QuantizationMode::U6 => DataType::U8,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
    }

    pub fn weight_codes_sign_flip_mask(&self) -> Option<u8> {
        match self {
            QuantizationMode::U4 => Some(0x88),
            QuantizationMode::I8 => None,
            QuantizationMode::U8 => Some(0x80),
            QuantizationMode::U3 | QuantizationMode::U5 | QuantizationMode::U6 => None,
        }
    }
}
