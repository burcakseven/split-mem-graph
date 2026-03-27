//! Vibegraph - Fast context memory graph database for LLMs
//!
//! Core structures: Node, Edge, Graph using index-based relationships
//! (no smart pointers for relationships).

/// Type-safe index for nodes. Wraps a u32 index into the nodes vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// Type-safe index for edges. Wraps a u32 index into the edges vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

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
    /// Create a new node with given id and data. Edge lists start empty.
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
    /// Create a new edge with given id, source, target, and data.
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
    pub fn add_node(&mut self, data: impl Into<String>) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            // Reuse existing tombstoned slot
            let node = Node::new(id, data);
            self.nodes[id.0 as usize] = node;
            id
        } else {
            // No free slots: append new
            let id = NodeId(self.nodes.len() as u32);
            let node = Node::new(id, data);
            self.nodes.push(node);
            id
        }
    }

    /// Add a directed edge from source to target with given data.
    /// Returns the new EdgeId.
    /// Panics if source or target NodeId is out of bounds or deleted.
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

        if let Some(id) = self.free_edges.pop() {
            // Reuse existing tombstoned edge slot
            let edge = Edge::new(id, source, target, data);
            self.edges[id.0 as usize] = edge;
            // Record edge in node's outgoing/incoming lists
            self.nodes[source.0 as usize].edges_out.push(id);
            self.nodes[target.0 as usize].edges_in.push(id);
            id
        } else {
            // No free slots: append new
            let id = EdgeId(self.edges.len() as u32);
            let edge = Edge::new(id, source, target, data);
            self.nodes[source.0 as usize].edges_out.push(id);
            self.nodes[target.0 as usize].edges_in.push(id);
            self.edges.push(edge);
            id
        }
    }

    /// Get a node by id (returns None if out of bounds or deleted).
    #[inline]
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes
            .get(id.0 as usize)
            .filter(|n| !n.deleted)
    }

    /// Get a mutable node by id (returns None if out of bounds or deleted).
    #[inline]
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes
            .get_mut(id.0 as usize)
            .filter(|n| !n.deleted)
    }

    /// Get an edge by id (returns None if out of bounds or deleted).
    #[inline]
    pub fn get_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges
            .get(id.0 as usize)
            .filter(|e| !e.deleted)
    }

    /// Get a mutable edge by id (returns None if out of bounds or deleted).
    #[inline]
    pub fn get_edge_mut(&mut self, id: EdgeId) -> Option<&mut Edge> {
        self.edges
            .get_mut(id.0 as usize)
            .filter(|e| !e.deleted)
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
}
