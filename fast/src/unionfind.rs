//! Disjoint-set union (union-find) for resolving linked user IDs into people.
//!
//! `rules.md`: `links/` holds ID pairs, joining is transitive, and a person's
//! name is the byte-wise smallest of their IDs.
//!
//! Why this structure and not a graph traversal: the generator produces chains
//! **hundreds of links long** (3 people, plus 1 in 20,000 of the rest, have
//! 50-400 IDs "linked one after the next in a random order"), and 0.4% of people
//! share a device whose ID "links them all into one person" — thousands of people
//! merged into a single set at full scale. So this must be iterative (no
//! recursion, which would risk a stack overflow on a long chain) and near-flat
//! under repeated lookups.
//!
//! Union by size plus path halving gives effectively constant-time operations.

pub struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }

    /// Grow to cover `n` elements, each initially its own set. IDs are interned
    /// as they are discovered, so the set count grows during parsing.
    pub fn grow_to(&mut self, n: usize) {
        while self.parent.len() < n {
            self.parent.push(self.parent.len() as u32);
            self.size.push(1);
        }
    }

    /// Representative of `x`'s set. Iterative, with path halving: every other
    /// node on the path is repointed at its grandparent, which flattens the tree
    /// without needing a second pass or any recursion.
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    /// Join the sets containing `a` and `b`. A self-link, or a repeat of an
    /// existing link either way round, is a no-op — which is exactly what
    /// `rules.md` requires.
    pub fn union(&mut self, a: u32, b: u32) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        // Attach the smaller tree under the larger one to keep depth down.
        if self.size[ra as usize] < self.size[rb as usize] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] += self.size[rb as usize];
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlinked_ids_are_their_own_people() {
        let mut uf = UnionFind::new(3);
        assert_ne!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(1), uf.find(2));
    }

    #[test]
    fn joining_is_transitive() {
        // rules.md: if `a,b` and `b,c` are lines, then a, b and c are one person.
        let mut uf = UnionFind::new(3);
        uf.union(0, 1);
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(2));
    }

    #[test]
    fn self_links_and_repeats_either_way_round_change_nothing() {
        let mut uf = UnionFind::new(3);
        uf.union(0, 0); // self-link
        uf.union(0, 1);
        uf.union(1, 0); // repeat, reversed
        uf.union(0, 1); // repeat
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
    }

    #[test]
    fn long_chain_links_into_one_person_without_recursing() {
        // The generator makes chains of 50-400 IDs "linked one after the next".
        // This is 100k, far beyond that, to prove there is no recursion depth limit.
        let n = 100_000;
        let mut uf = UnionFind::new(n);
        for i in 0..n as u32 - 1 {
            uf.union(i, i + 1);
        }
        let root = uf.find(0);
        for i in 0..n as u32 {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn shared_device_merges_many_people_into_one() {
        // 0.4% of people share a device ID that links them all into one person.
        let mut uf = UnionFind::new(10_001);
        let device = 10_000;
        for person in 0..10_000u32 {
            uf.union(person, device);
        }
        let root = uf.find(0);
        for person in 0..10_000u32 {
            assert_eq!(uf.find(person), root);
        }
    }

    #[test]
    fn grow_to_adds_singletons() {
        let mut uf = UnionFind::new(1);
        uf.union(0, 0);
        uf.grow_to(5);
        assert_eq!(uf.len(), 5);
        assert_eq!(uf.find(4), 4);
        uf.union(0, 4);
        assert_eq!(uf.find(0), uf.find(4));
        assert_ne!(uf.find(0), uf.find(3));
    }
}
