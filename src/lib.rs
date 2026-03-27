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
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique node identifier (index into Graph::nodes).
    pub id: NodeId,
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
            data: data.into(),
            edges_out: Vec::new(),
            edges_in: Vec::new(),
        }
    }
}

/// A directed edge connecting two nodes via index references.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Unique edge identifier (index into Graph::edges).
    pub id: EdgeId,
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
            source,
            target,
            data: data.into(),
        }
    }
}

/// The graph container holding nodes and edges in flat vectors.
/// Relationships are stored as indices (u32) for fast, cache-friendly access.
#[derive(Debug, Default)]
pub struct Graph {
    /// All nodes stored contiguously.
    pub nodes: Vec<Node>,
    /// All edges stored contiguously.
    pub edges: Vec<Edge>,
}

impl Graph {
    /// Create an empty graph.
    #[inline]
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node with the given data. Returns the new NodeId.
    pub fn add_node(&mut self, data: impl Into<String>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        let node = Node::new(id, data);
        self.nodes.push(node);
        id
    }

    /// Add a directed edge from source to target with given data.
    /// Returns the new EdgeId.
    /// Panics if source or target NodeId is out of bounds.
    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        data: impl Into<String>,
    ) -> EdgeId {
        // Validate node indices exist
        let _ = &self.nodes[source.0 as usize];
        let _ = &self.nodes[target.0 as usize];

        let id = EdgeId(self.edges.len() as u32);
        let edge = Edge::new(id, source, target, data);

        // Record edge in node's outgoing/incoming lists
        self.nodes[source.0 as usize].edges_out.push(id);
        self.nodes[target.0 as usize].edges_in.push(id);

        self.edges.push(edge);
        id
    }

    /// Get a node by id (returns None if out of bounds).
    #[inline]
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)
    }

    /// Get a mutable node by id.
    #[inline]
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id.0 as usize)
    }

    /// Get an edge by id (returns None if out of bounds).
    #[inline]
    pub fn get_edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.0 as usize)
    }

    /// Get a mutable edge by id.
    #[inline]
    pub fn get_edge_mut(&mut self, id: EdgeId) -> Option<&mut Edge> {
        self.edges.get_mut(id.0 as usize)
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
}
