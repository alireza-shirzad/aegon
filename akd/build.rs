// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/shard.proto",
                "proto/coordinator.proto",
                "proto/masking.proto",
                "proto/srs.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
