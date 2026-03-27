//! Vibegraph - Fast context memory graph database for LLMs
//!
//! Core structures: Node, Edge, Graph using index-based relationships
//! (no smart pointers for relationships).

/// Type-safe index for nodes. (index, generation) — generation prevents stale IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32, pub u32);

/// Type-safe index for edges. (index, generation) — generation prevents stale IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32, pub u32);

/// A graph node storing context data and edge relationships via indices.
/// Uses tombstone flag for deletion (no array shifting, stable indices).
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique node identifier (index into Graph::nodes).
    pub id: NodeId,
    /// Tombstone flag: true means slot is free (deleted).
    pub deleted: bool,
    /// Payload data (e.g., LLM context text).
    pub data: String,
    /// Outgoing edge indices (no smart pointers - raw index refs).
    pub edges_out: Vec<EdgeId>,
    /// Incoming edge indices (no smart pointers - raw index refs).
    pub edges_in: Vec<EdgeId>,
}

impl Node {
    /// Create a new node with given id (includes generation) and data.
    #[inline]
    pub fn new(id: NodeId, data: impl Into<String>) -> Self {
        Node {
            id,
            deleted: false,
            data: data.into(),
            edges_out: Vec::new(),
            edges_in: Vec::new(),
        }
    }
}

/// A directed edge connecting two nodes via index references.
/// Uses tombstone flag for deletion (no array shifting, stable indices).
#[derive(Debug, Clone)]
pub struct Edge {
    /// Unique edge identifier (index into Graph::edges).
    pub id: EdgeId,
    /// Tombstone flag: true means slot is free (deleted).
    pub deleted: bool,
    /// Source node index (no smart pointer).
    pub source: NodeId,
    /// Target node index (no smart pointer).
    pub target: NodeId,
    /// Payload data (e.g., relationship label or metadata).
    pub data: String,
}

impl Edge {
    /// Create a new edge with given id (includes generation), source, target, data.
    #[inline]
    pub fn new(id: EdgeId, source: NodeId, target: NodeId, data: impl Into<String>) -> Self {
        Edge {
            id,
            deleted: false,
            source,
            target,
            data: data.into(),
        }
    }
}

/// The graph container holding nodes and edges in flat vectors.
/// Relationships are stored as indices (u32) for fast, cache-friendly access.
///
/// Deletion uses tombstones + freelist: no array shifting, stable IDs,
/// freed slots are reused by add_node / add_edge.
#[derive(Debug, Default)]
pub struct Graph {
    /// All nodes stored contiguously (some slots may be deleted/tombstoned).
    pub nodes: Vec<Node>,
    /// All edges stored contiguously (some slots may be deleted/tombstoned).
    pub edges: Vec<Edge>,
    /// Freelist of node indices available for reuse (no shifting).
    free_nodes: Vec<NodeId>,
    /// Freelist of edge indices available for reuse (no shifting).
    free_edges: Vec<EdgeId>,
}

impl Graph {
    /// Create an empty graph.
    #[inline]
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            free_nodes: Vec::new(),
            free_edges: Vec::new(),
        }
    }

    /// Add a node with the given data. Returns the new NodeId.
    /// Reuses a freed slot from freelist if available (no array growth).
    /// Increments generation on reuse so old NodeIds become invalid.
    pub fn add_node(&mut self, data: impl Into<String>) -> NodeId {
        if let Some(old_id) = self.free_nodes.pop() {
            // Reuse existing tombstoned slot: bump generation
            let idx = old_id.0 as usize;
            let new_gen = old_id.1 + 1;
            let new_id = NodeId(idx as u32, new_gen);
            let node = Node::new(new_id, data);
            self.nodes[idx] = node;
            new_id
        } else {
            // No free slots: append new with gen 0
            let idx = self.nodes.len() as u32;
            let id = NodeId(idx, 0);
            let node = Node::new(id, data);
            self.nodes.push(node);
            id
        }
    }

    /// Add a directed edge from source to target with given data.
    /// Returns the new EdgeId.
    /// Panics if source or target NodeId is out of bounds or deleted.
    /// Increments generation on reuse so old EdgeIds become invalid.
    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        data: impl Into<String>,
    ) -> EdgeId {
        // Validate node indices exist and not deleted
        assert!(
            !self.nodes[source.0 as usize].deleted,
            "add_edge: source node is deleted"
        );
        assert!(
            !self.nodes[target.0 as usize].deleted,
            "add_edge: target node is deleted"
        );

        if let Some(old_id) = self.free_edges.pop() {
            // Reuse tombstoned edge slot: bump generation
            let idx = old_id.0 as usize;
            let new_gen = old_id.1 + 1;
            let new_id = EdgeId(idx as u32, new_gen);
            let edge = Edge::new(new_id, source, target, data);
            self.edges[idx] = edge;
            self.nodes[source.0 as usize].edges_out.push(new_id);
            self.nodes[target.0 as usize].edges_in.push(new_id);
            new_id
        } else {
            // No free slots: append new with gen 0
            let idx = self.edges.len() as u32;
            let id = EdgeId(idx, 0);
            let edge = Edge::new(id, source, target, data);
            self.nodes[source.0 as usize].edges_out.push(id);
            self.nodes[target.0 as usize].edges_in.push(id);
            self.edges.push(edge);
            id
        }
    }

    /// Get a node by id (returns None if out of bounds, deleted, or stale generation).
    #[inline]
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes
            .get(id.0 as usize)
            .filter(|n| !n.deleted && n.id == id)
    }

    /// Get a mutable node by id (returns None if out of bounds, deleted, or stale generation).
    #[inline]
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes
            .get_mut(id.0 as usize)
            .filter(|n| !n.deleted && n.id == id)
    }

    /// Get an edge by id (returns None if out of bounds, deleted, or stale generation).
    #[inline]
    pub fn get_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges
            .get(id.0 as usize)
            .filter(|e| !e.deleted && e.id == id)
    }

    /// Get a mutable edge by id (returns None if out of bounds, deleted, or stale generation).
    #[inline]
    pub fn get_edge_mut(&mut self, id: EdgeId) -> Option<&mut Edge> {
        self.edges
            .get_mut(id.0 as usize)
            .filter(|e| !e.deleted && e.id == id)
    }

    /// Remove a node and all its incident edges (tombstone, no shifting).
    /// Returns true if the node existed and was removed; false if already deleted or OOB.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        let idx = id.0 as usize;
        if idx >= self.nodes.len() {
            return false;
        }
        if self.nodes[idx].deleted {
            return false; // already removed
        }

        // Mark node as deleted
        self.nodes[idx].deleted = true;

        // Collect edges to remove (copy IDs since we'll mutate edge lists)
        let out_edges: Vec<EdgeId> = self.nodes[idx].edges_out.clone();
        let in_edges: Vec<EdgeId> = self.nodes[idx].edges_in.clone();

        // Remove outgoing edges: delete edge, remove from target's edges_in
        for e_id in out_edges {
            self.remove_edge_internal(e_id, /*remove_from_source=*/ false);
        }
        // Remove incoming edges: delete edge, remove from source's edges_out
        for e_id in in_edges {
            self.remove_edge_internal(e_id, /*remove_from_source=*/ true);
        }

        // Clear node's adjacency lists (they reference deleted edges now)
        self.nodes[idx].edges_out.clear();
        self.nodes[idx].edges_in.clear();

        // Add node slot to freelist for reuse
        self.free_nodes.push(id);
        true
    }

    /// Internal: remove an edge by id.
    /// If remove_from_source is true, also remove from source's edges_out.
    /// Returns true if removed; false if OOB or already deleted.
    fn remove_edge_internal(&mut self, id: EdgeId, remove_from_source: bool) -> bool {
        let idx = id.0 as usize;
        if idx >= self.edges.len() {
            return false;
        }
        if self.edges[idx].deleted {
            return false;
        }

        self.edges[idx].deleted = true;

        let src = self.edges[idx].source;
        let tgt = self.edges[idx].target;

        // Remove edge id from target's edges_in (always; edge points target)
        if let Some(t) = self.nodes.get_mut(tgt.0 as usize) {
            t.edges_in.retain(|&eid| eid != id);
        }

        // Optionally remove from source's edges_out
        if remove_from_source {
            if let Some(s) = self.nodes.get_mut(src.0 as usize) {
                s.edges_out.retain(|&eid| eid != id);
            }
        }

        // Add edge slot to freelist
        self.free_edges.push(id);
        true
    }

    /// Remove a single edge by id (tombstone, no shifting).
    /// Cleans up from both endpoints' adjacency lists.
    /// Returns true if removed; false if OOB or already deleted.
    pub fn remove_edge(&mut self, id: EdgeId) -> bool {
        self.remove_edge_internal(id, /*remove_from_source=*/ true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_and_edge() {
        let mut g = Graph::new();
        let a = g.add_node("node A");
        let b = g.add_node("node B");
        let e = g.add_edge(a, b, "connects");

        assert_eq!(a.0, 0);
        assert_eq!(b.0, 1);
        assert_eq!(e.0, 0);

        assert_eq!(g.get_node(a).unwrap().data, "node A");
        assert_eq!(g.get_node(b).unwrap().data, "node B");
        assert_eq!(g.get_edge(e).unwrap().source, a);
        assert_eq!(g.get_edge(e).unwrap().target, b);

        // Check relationship lists
        assert!(g.get_node(a).unwrap().edges_out.contains(&e));
        assert!(g.get_node(b).unwrap().edges_in.contains(&e));
    }

    #[test]
    fn test_remove_node_no_shift_and_freelist_reuse() {
        let mut g = Graph::new();

        let a = g.add_node("A"); // 0
        let b = g.add_node("B"); // 1
        let c = g.add_node("C"); // 2
        assert_eq!(g.nodes.len(), 3);

        // Add edges
        let e1 = g.add_edge(a, b, "a->b");
        let e2 = g.add_edge(b, c, "b->c");
        assert_eq!(g.edges.len(), 2);

        // Remove node B (index 1)
        assert!(g.remove_node(b));
        assert!(g.get_node(b).is_none()); // tombstoned -> None
        assert!(g.get_edge(e1).is_none()); // incident edge tombstoned
        assert!(g.get_edge(e2).is_none()); // incident edge tombstoned

        // Arrays NOT shifted: length stays same, slots 0 and 2 still valid
        assert_eq!(g.nodes.len(), 3);
        assert!(g.get_node(a).is_some());
        assert!(g.get_node(c).is_some());

        // Freelist has the freed node id
        assert!(g.free_nodes.contains(&b));

        // Add new node: should reuse slot 1 (b's old index)
        let d = g.add_node("D");
        assert_eq!(d.0, 1); // reused B's slot!
        assert!(g.free_nodes.is_empty()); // freelist consumed

        // d is valid, a and c still valid
        assert_eq!(g.get_node(d).unwrap().data, "D");
        assert_eq!(g.get_node(a).unwrap().data, "A");
        assert_eq!(g.get_node(c).unwrap().data, "C");

        // Arrays still no shift
        assert_eq!(g.nodes.len(), 3);
    }

    #[test]
    fn test_remove_edge_freelist_reuse() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");

        let e1 = g.add_edge(a, b, "e1");
        let e2 = g.add_edge(a, b, "e2");
        assert_eq!(g.edges.len(), 2);

        assert!(g.remove_edge(e1));
        assert!(g.get_edge(e1).is_none());
        assert!(g.get_edge(e2).is_some()); // e2 still works

        assert_eq!(g.edges.len(), 2); // no shift
        assert!(g.free_edges.contains(&e1));

        // New edge reuses slot
        let e3 = g.add_edge(b, a, "e3");
        assert_eq!(e3.0, 0); // reused e1's slot
        assert!(g.free_edges.is_empty());
    }

    // ========== ZOMBIE TESTS ==========

    #[test]
    fn test_zombie_node_not_accessible() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");

        assert!(g.remove_node(a));

        // get_node must return None for deleted
        assert!(g.get_node(a).is_none(), "deleted node should be None");
        assert!(g.get_node(b).is_some(), "non-deleted node should exist");

        // Raw slot still exists (by design) but is tombstoned
        assert!(g.nodes[a.0 as usize].deleted, "raw slot should be tombstoned");
    }

    #[test]
    fn test_zombie_edge_not_accessible() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let e = g.add_edge(a, b, "link");

        assert!(g.remove_edge(e));

        assert!(g.get_edge(e).is_none(), "deleted edge should be None");
        assert!(g.edges[e.0 as usize].deleted, "raw edge slot should be tombstoned");
    }

    #[test]
    fn test_zombie_incident_edges_removed_with_node() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");

        let e_ab = g.add_edge(a, b, "a->b");
        let e_bc = g.add_edge(b, c, "b->c");
        let e_ca = g.add_edge(c, a, "c->a"); // to a

        // Remove node A
        assert!(g.remove_node(a));

        // All edges touching A must be gone
        assert!(g.get_edge(e_ab).is_none(), "edge a->b should be deleted");
        assert!(g.get_edge(e_ca).is_none(), "edge c->a should be deleted");
        assert!(g.get_edge(e_bc).is_some(), "edge b->c should still exist");

        // b's adjacency lists should not reference deleted edges
        let b_node = g.get_node(b).unwrap();
        assert!(
            !b_node.edges_in.iter().any(|eid| *eid == e_ab),
            "b.edges_in should not contain deleted e_ab"
        );
        assert!(
            !b_node.edges_out.iter().any(|eid| *eid == e_ab),
            "b.edges_out should not contain deleted e_ab"
        );
    }

    #[test]
    fn test_zombie_readded_node_clean() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let e = g.add_edge(a, b, "link");

        assert!(g.remove_node(a));

        // Re-add: slot 0 reused
        let a2 = g.add_node("A2");
        assert_eq!(a2.0, 0);

        // a2 should have fresh empty edge lists
        let a2_node = g.get_node(a2).unwrap();
        assert!(a2_node.edges_out.is_empty(), "re-added node should have no old edges_out");
        assert!(a2_node.edges_in.is_empty(), "re-added node should have no old edges_in");
        assert_eq!(a2_node.data, "A2");

        // Old edge slot is still tombstoned (not reachable)
        assert!(g.get_edge(e).is_none());

        // Add new edge from a2
        let e2 = g.add_edge(a2, b, "a2->b");
        assert!(g.get_edge(e2).is_some());
    }

    #[test]
    fn test_zombie_readded_edge_clean() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");

        let e1 = g.add_edge(a, b, "e1");
        assert!(g.remove_edge(e1));

        // Re-add edge: slot 0 reused
        let e2 = g.add_edge(b, a, "e2");
        assert_eq!(e2.0, 0);

        // e2 should be a fresh edge (slot overwritten)
        let edge = g.get_edge(e2).unwrap();
        assert_eq!(edge.source, b);
        assert_eq!(edge.target, a);
        assert_eq!(edge.data, "e2");
        assert!(!edge.deleted);

        // Old e1 EdgeId is not accessible anymore
        assert!(g.get_edge(e1).is_none(), "old e1 EdgeId should return None");

        // Slot 0 now holds the new e2 (not tombstoned)
        assert!(!g.edges[0].deleted, "reused slot is now a valid edge");
    }

    #[test]
    fn test_double_remove_is_idempotent() {
        let mut g = Graph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let e = g.add_edge(a, b, "e");

        // First remove
        assert!(g.remove_node(a));
        assert!(!g.remove_node(a), "second remove_node should return false");

        // Edge already removed too
        assert!(!g.remove_edge(e), "remove_edge on deleted should return false");

        // Graph still healthy
        assert!(g.get_node(b).is_some());
    }

    // ========== STRESS TESTS ==========

    #[test]
    fn test_stress_add_remove_cycles() {
        // Many cycles of add/remove to stress freelist and detect zombies
        let mut g = Graph::new();

        for cycle in 0..1000 {
            // Add a batch of nodes
            let mut ids = Vec::new();
            for i in 0..10 {
                ids.push(g.add_node(format!("n{}_{}", cycle, i)));
            }
            // Add edges among them (simple chain)
            for i in 0..9 {
                g.add_edge(ids[i], ids[i + 1], "chain");
            }

            // Remove half of them
            for i in (0..10).step_by(2) {
                assert!(g.remove_node(ids[i]));
            }

            // Remaining half should be accessible
            for i in (1..10).step_by(2) {
                assert!(g.get_node(ids[i]).is_some(), "cycle {} node {} should exist", cycle, i);
            }
        }

        // Final sanity: graph still works
        let x = g.add_node("final");
        let y = g.add_node("final2");
        let e = g.add_edge(x, y, "final_edge");
        assert!(g.get_node(x).is_some());
        assert!(g.get_edge(e).is_some());
    }

    #[test]
    fn test_stress_freelist_reuse_no_leak() {
        // Verify freelist actually reuses and doesn't keep growing unbounded
        let mut g = Graph::new();

        for _ in 0..5000 {
            let a = g.add_node("x");
            let b = g.add_node("y");
            let _e = g.add_edge(a, b, "e");
            assert!(g.remove_node(a));
            assert!(g.remove_node(b));
        }

        // After all cycles, we expect nodes/edges arrays to be bounded
        // (only 2 active nodes + some tombstoned slots from final partial state)
        // The freelist should have been heavily reused.
        // We don't hardcode exact numbers but verify no explosion.
        assert!(g.nodes.len() <= 100, "nodes vec should not explode: {}", g.nodes.len());
        assert!(g.edges.len() <= 100, "edges vec should not explode: {}", g.edges.len());

        // Graph still functional
        let a = g.add_node("post-stress-a");
        let b = g.add_node("post-stress-b");
        let e = g.add_edge(a, b, "post-stress-e");
        assert!(g.get_node(a).is_some());
        assert!(g.get_edge(e).is_some());
    }

    #[test]
    fn test_stress_mass_deletion_then_readd() {
        let mut g = Graph::new();

        // Create a larger graph
        let n = 200usize;
        let mut node_ids = Vec::with_capacity(n);
        for i in 0..n {
            node_ids.push(g.add_node(format!("node{}", i)));
        }
        // Add edges: each node connects to next (chain)
        for i in 0..n - 1 {
            g.add_edge(node_ids[i], node_ids[i + 1], "link");
        }
        assert_eq!(g.nodes.len(), n);
        assert_eq!(g.edges.len(), n - 1);

        // Delete all even-indexed nodes
        for i in (0..n).step_by(2) {
            g.remove_node(node_ids[i]);
        }

        // Verify odds still exist
        for i in (1..n).step_by(2) {
            assert!(g.get_node(node_ids[i]).is_some());
        }

        // Re-add new nodes (should reuse slots)
        let mut new_ids = Vec::new();
        for i in 0..n / 2 {
            new_ids.push(g.add_node(format!("new{}", i)));
        }

        // New nodes should have reused the deleted slots
        // (freelist should have been exhausted and more added)
        for id in &new_ids {
            assert!(g.get_node(*id).is_some());
            assert!(g.get_node(*id).unwrap().edges_out.is_empty());
            assert!(g.get_node(*id).unwrap().edges_in.is_empty());
        }

        // Old odd nodes still intact
        assert!(g.get_node(node_ids[1]).is_some());
        assert!(g.get_node(node_ids[n - 1]).is_some());
    }

    #[test]
    fn test_stress_bidirectional_cleanup() {
        // Verify edges_in/edges_out are cleaned on both sides
        let mut g = Graph::new();

        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");

        let e1 = g.add_edge(a, b, "a->b");
        let e2 = g.add_edge(b, c, "b->c");
        let e3 = g.add_edge(c, a, "c->a");

        // Remove b
        g.remove_node(b);

        // a should not have e1 in edges_out anymore
        let a_node = g.get_node(a).unwrap();
        assert!(
            !a_node.edges_out.contains(&e1),
            "a.edges_out should not contain deleted e1"
        );
        // a still has e3 in edges_in (from c)
        assert!(
            a_node.edges_in.contains(&e3),
            "a.edges_in should still contain e3"
        );

        // c should not have e2 in edges_in anymore
        let c_node = g.get_node(c).unwrap();
        assert!(
            !c_node.edges_in.contains(&e2),
            "c.edges_in should not contain deleted e2"
        );
        assert!(
            c_node.edges_out.contains(&e3),
            "c.edges_out should still contain e3"
        );

        // e1, e2 deleted; e3 alive
        assert!(g.get_edge(e1).is_none());
        assert!(g.get_edge(e2).is_none());
        assert!(g.get_edge(e3).is_some());
    }
}
