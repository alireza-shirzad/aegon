// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! How a deployment's shards are partitioned into independent
//! Fiat-Shamir chains.
//!
//! Aegon's audit derives two chain scalars per epoch, `r_index` and
//! `r_value`, by absorbing every shard's new data commitment into a
//! rolling transcript. Each scalar weights that epoch's delta in a
//! random linear combination, and its only job is to be
//! unpredictable to the server *before the server fixes the
//! commitments it weights*.
//!
//! A transcript over one group's commitments delivers that for that
//! group's shards exactly as well as a transcript over all `η` does.
//! The directory-wide transcript is a batching convenience, not a
//! soundness requirement — so a deployment may run `G` independent
//! accumulators instead of one, at a cost of `log2 G` bits against a
//! 128-bit challenge.
//!
//! Why bother: the groups never interact, so every audit-side cost
//! divides by `G` and the pieces run concurrently. That matters most
//! for the recursive (IVC) auditor, whose compression step is linear
//! in circuit size — see
//! [`ivc::grouped`](crate::aegon::ivc::grouped). The classic
//! per-epoch auditor is unaffected in aggregate cost; it simply
//! tracks `G` scalar pairs instead of one.
//!
//! Grouping does **not** touch the lookup or history paths. Those
//! verify `rand_index` openings by comparing a placement opening
//! against a live one; neither side re-derives `r_index`, so how the
//! scalar was produced is invisible to them.

use std::ops::Range;

use crate::aegon::error::AegonError;

/// How a deployment's shards are partitioned into folding chains.
///
/// Groups are **contiguous** shard ranges, which keeps a group's
/// shards adjacent in the Merkle tree over `per_shard` — so a group's
/// leaves form a subtree and a future optimisation could hand an
/// auditor a single subtree path per group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupPlan {
    n_shards: usize,
    groups: usize,
}

impl GroupPlan {
    /// Partition `n_shards` into `groups` equal contiguous ranges.
    ///
    /// Requires `groups` to divide `n_shards` exactly. Unequal groups
    /// would need per-group [`IvcAuditParams`] — different shapes
    /// mean different R1CS — which would give back the setup saving
    /// and most of the simplicity, so it is rejected rather than
    /// silently supported.
    pub fn new(n_shards: usize, groups: usize) -> Result<Self, AegonError> {
        if groups == 0 || n_shards == 0 {
            return Err(AegonError::Config(
                "group plan needs at least one shard and one group".into(),
            ));
        }
        if !n_shards.is_multiple_of(groups) {
            return Err(AegonError::Config(format!(
                "group count {groups} must divide the shard count {n_shards} exactly"
            )));
        }
        Ok(Self { n_shards, groups })
    }

    /// A single chain over every shard — today's behaviour.
    pub fn single(n_shards: usize) -> Result<Self, AegonError> {
        Self::new(n_shards, 1)
    }

    /// Total shards across all groups.
    pub fn n_shards(&self) -> usize {
        self.n_shards
    }

    /// Number of independent folding chains.
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Shards in each group.
    pub fn shards_per_group(&self) -> usize {
        self.n_shards / self.groups
    }

    /// The shard index range group `g` covers.
    pub fn range(&self, g: usize) -> Range<usize> {
        let k = self.shards_per_group();
        (g * k)..((g + 1) * k)
    }

    /// The group a shard belongs to.
    pub fn group_of(&self, shard: usize) -> usize {
        shard / self.shards_per_group()
    }

    /// Slice `items` into one contiguous chunk per group.
    ///
    /// Errors if `items` does not cover exactly `n_shards` entries —
    /// the check that stops a prover from folding a partition the
    /// verifier did not ask for.
    pub fn split<'a, T>(&self, items: &'a [T]) -> Result<Vec<&'a [T]>, AegonError> {
        if items.len() != self.n_shards {
            return Err(AegonError::Config(format!(
                "chain-group plan covers {} shards, got {}",
                self.n_shards,
                items.len()
            )));
        }
        Ok((0..self.groups).map(|g| &items[self.range(g)]).collect())
    }
}
