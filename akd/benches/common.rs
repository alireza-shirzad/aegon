// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#[macro_export]
macro_rules! bench_config {
    ( $x:ident ) => {
        paste::paste! {
            // NOTE(new_config): Add a new configuration here

            #[cfg(feature = "whatsapp_v1")]
            fn [<$x _ whatsapp_v1_config>](c: &mut Criterion) {
                $x::<akd_core::WhatsAppV1Configuration>(c)
            }

            #[cfg(feature = "experimental")]
            fn [<$x _ experimental_config>](c: &mut Criterion) {
                $x::<akd_core::ExperimentalConfiguration<akd_core::ExampleLabel>>(c)
            }
        }
    };
}

#[macro_export]
macro_rules! group_config {
    ( $( $group:path ),+ $(,)* ) => {
        paste::paste! {
            // NOTE(new_config): Add a new configuration here

            #[cfg(feature = "whatsapp_v1")]
            criterion_group!(
                $(
                    [<$group _ whatsapp_v1_config>],
                )+
            );

            #[cfg(feature = "experimental")]
            criterion_group!(
                $(
                    [<$group _ experimental_config>],
                )+
            );
        }
    };
}
