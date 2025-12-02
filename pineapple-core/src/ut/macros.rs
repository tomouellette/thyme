// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

#[macro_export]
macro_rules! impl_enum_dispatch {
    // Case when the method takes &self with a lifetime and has NO arguments
    ($enum_name:ident<$lifetime:lifetime>, $($variant:ident),*; $fn_name:ident(&$self_lifetime:lifetime self) -> $ret:ty) => {
        impl<$lifetime> $enum_name<$lifetime> {
            pub fn $fn_name(&$self_lifetime self) -> $ret {
                match self {
                    $(Self::$variant(v) => v.$fn_name(),)*
                }
            }
        }
    };

    // Case when the method takes &self and has arguments
    ($enum_name:ident, $($variant:ident),*; $fn_name:ident(&self, $($arg:ident : $arg_ty:ty),+) -> $ret:ty) => {
        impl $enum_name {
            pub fn $fn_name(&self, $($arg: $arg_ty),+) -> $ret {
                match self {
                    $(Self::$variant(v) => v.$fn_name($($arg),+),)*
                }
            }
        }
    };

    // Case when the method takes &self and has NO arguments
    ($enum_name:ident, $($variant:ident),*; $fn_name:ident(&self) -> $ret:ty) => {
        impl $enum_name {
            pub fn $fn_name(&self) -> $ret {
                match self {
                    $(Self::$variant(v) => v.$fn_name(),)*
                }
            }
        }
    };

    // Case when the method takes &mut self and has arguments
    ($enum_name:ident, $($variant:ident),*; $fn_name:ident(&mut self, $($arg:ident : $arg_ty:ty),+) -> $ret:ty) => {
        impl $enum_name {
            pub fn $fn_name(&mut self, $($arg: $arg_ty),+) -> $ret {
                match self {
                    $(Self::$variant(v) => v.$fn_name($($arg),+),)*
                }
            }
        }
    };

    // Case when the method takes &mut self and has NO arguments
    ($enum_name:ident, $($variant:ident),*; $fn_name:ident(&mut self) -> $ret:ty) => {
        impl $enum_name {
            pub fn $fn_name(&mut self) -> $ret {
                match self {
                    $(Self::$variant(v) => v.$fn_name(),)*
                }
            }
        }
    };
}
