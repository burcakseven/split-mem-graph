//! Integration tests for Vibegraph
//!
//! These tests combine multiple features (add, remove, traverse, freelist, etc.)
//! to verify the entire system works correctly together.

use vibegraph::{Graph, NodeId};

/// Helper: collect all accessible node data (skips deleted).
fn accessible_node_data(g: &Graph) -> Vec<String> {
    g.nodes
        .iter()
        .filter(|n| !n.deleted)
        .map(|n| n.data.clone())
        .collect()
}

#[test]
fn it_builds_graph_adds_nodes_edges_and_removes() {
    let mut g = Graph::new();

    // Add 5 nodes
    let a = g.add_node("A");
    let b = g.add_node("B");
    let c = g.add_node("C");
    let d = g.add_node("D");
    let e = g.add_node("E");

    // Add edges forming a simple chain: A -> B -> C -> D -> E
    let e1 = g.add_edge(a, b, "a-b");
    let e2 = g.add_edge(b, c, "b-c");
    let _e3 = g.add_edge(c, d, "c-d");
    let _e4 = g.add_edge(d, e, "d-e");

    // Verify counts
    assert_eq!(g.nodes.len(), 5);
    assert_eq!(g.edges.len(), 4);

    // Verify all accessible
    let data = accessible_node_data(&g);
    assert_eq!(data, vec!["A", "B", "C", "D", "E"]);

    // Remove middle node B
    assert!(g.remove_node(b));
    assert!(g.get_node(b).is_none());
    assert!(g.get_edge(e1).is_none()); // incident edge deleted
    assert!(g.get_edge(e2).is_none()); // incident edge deleted

    // Remaining nodes: A, C, D, E (B deleted, count stays 5)
    assert_eq!(g.nodes.len(), 5);
    let data2 = accessible_node_data(&g);
    assert_eq!(data2, vec!["A", "C", "D", "E"]);

    // Capacity unchanged (no shrink)
    assert!(g.nodes.capacity() >= 5);
}

#[test]
fn it_reuses_freed_slots_and_bumps_generation() {
    let mut g = Graph::new();

    let a = g.add_node("first A");
    assert_eq!(a, NodeId(0, 0));

    g.remove_node(a);
    assert!(g.get_node(a).is_none());

    // Re-add should reuse slot 0 and bump gen
    let a2 = g.add_node("second A");
    assert_eq!(a2, NodeId(0, 1));

    // Old ID rejected
    assert!(g.get_node(a).is_none());
    // New ID works
    assert_eq!(g.get_node(a2).unwrap().data, "second A");
}

#[test]
fn it_traverses_bfs_from_multiple_starts() {
    let mut g = Graph::new();

    let root1 = g.add_node("root1");
    let root2 = g.add_node("root2");
    let child1 = g.add_node("child1");
    let child2 = g.add_node("child2");
    let grandchild = g.add_node("grandchild");

    g.add_edge(root1, child1, "r1->c1");
    g.add_edge(root2, child2, "r2->c2");
    g.add_edge(child1, grandchild, "c1->gc");

    // From both roots, depth 2
    let visited = g.traverse(&[root1, root2], 2);

    assert!(visited.contains(&root1));
    assert!(visited.contains(&root2));
    assert!(visited.contains(&child1));
    assert!(visited.contains(&child2));
    assert!(visited.contains(&grandchild));
    // No duplicates
    assert_eq!(visited.iter().filter(|&&x| x == grandchild).count(), 1);
}

#[test]
fn it_traverses_with_edges_returns_subgraph() {
    let mut g = Graph::new();

    let a = g.add_node("A");
    let b = g.add_node("B");
    let c = g.add_node("C");

    let e1 = g.add_edge(a, b, "a->b");
    let e2 = g.add_edge(b, c, "b->c");

    let (nodes, edges) = g.traverse_with_edges(&[a], 2);

    assert_eq!(nodes.len(), 3);
    assert_eq!(edges.len(), 2);
    assert!(edges.contains(&e1));
    assert!(edges.contains(&e2));
}

#[test]
fn it_handles_removal_of_all_nodes_then_readd() {
    let mut g = Graph::new();

    let n1 = g.add_node("n1");
    let n2 = g.add_node("n2");
    g.add_edge(n1, n2, "e");

    // Remove all
    g.remove_node(n1);
    g.remove_node(n2);

    assert!(g.get_node(n1).is_none());
    assert!(g.get_node(n2).is_none());
    // Arrays still exist, capacity stable
    assert!(g.nodes.capacity() >= 2);

    // Re-add fresh nodes (reuse slots)
    let m1 = g.add_node("m1");
    let m2 = g.add_node("m2");
    g.add_edge(m1, m2, "new-e");

    assert!(g.get_node(m1).is_some());
    assert!(g.get_node(m2).is_some());
    assert!(g.get_edge(g.edges.iter().find(|e| !e.deleted).unwrap().id).is_some());
}

#[test]
fn it_preserves_stable_ids_across_mixed_operations() {
    let mut g = Graph::new();

    let a = g.add_node("A");
    let b = g.add_node("B");
    let c = g.add_node("C");

    let e_ab = g.add_edge(a, b, "a->b");
    let e_bc = g.add_edge(b, c, "b->c");

    // Snapshot IDs
    let a_snap = a;
    let b_snap = b;
    let c_snap = c;
    let e_ab_snap = e_ab;
    let e_bc_snap = e_bc;

    // Remove B
    g.remove_node(b);

    // A and C still have same IDs
    assert_eq!(g.get_node(a_snap).unwrap().data, "A");
    assert_eq!(g.get_node(c_snap).unwrap().data, "C");

    // Old edge IDs invalid
    assert!(g.get_edge(e_ab_snap).is_none());
    assert!(g.get_edge(e_bc_snap).is_none());

    // Re-add B (new generation)
    let b2 = g.add_node("B2");
    assert_ne!(b2.1, b_snap.1); // generation changed

    // A can still be found with original ID
    assert_eq!(g.get_node(a_snap).unwrap().data, "A");
}

#[test]
fn it_traverse_depth_zero_returns_only_starts() {
    let mut g = Graph::new();
    let a = g.add_node("A");
    let b = g.add_node("B");
    g.add_edge(a, b, "a->b");

    let v = g.traverse(&[a], 0);
    assert_eq!(v, vec![a]);
}

#[test]
fn it_traverse_skips_deleted_nodes_in_path() {
    let mut g = Graph::new();

    let a = g.add_node("A");
    let b = g.add_node("B");
    let c = g.add_node("C");

    g.add_edge(a, b, "a->b");
    g.add_edge(b, c, "b->c");

    // Remove B -> A can no longer reach C
    g.remove_node(b);

    let v = g.traverse(&[a], 5);
    assert_eq!(v, vec![a]); // only A reachable
}

#[test]
fn it_double_remove_is_safe_and_idempotent() {
    let mut g = Graph::new();
    let a = g.add_node("A");
    let e = g.add_edge(a, a, "self"); // self-loop

    assert!(g.remove_node(a));
    assert!(!g.remove_node(a)); // second remove false
    assert!(!g.remove_edge(e)); // edge already gone

    // Graph still usable
    let b = g.add_node("B");
    assert!(g.get_node(b).is_some());
}

#[test]
fn it_nodes_and_edges_capacity_never_shrinks() {
    let mut g = Graph::new();

    for _ in 0..100 {
        let n = g.add_node("x");
        g.remove_node(n);
    }

    // Capacity stays at least as large as peak
    assert!(g.nodes.capacity() >= 1);
    // Edges capacity is usize (>=0 always), just verify it's non-panicking
    let _ = g.edges.capacity();
}

#[test]
fn it_full_workflow_llm_context_simulation() {
    // Simulate: build a knowledge graph, remove outdated, traverse for context
    let mut g = Graph::new();

    let query = g.add_node("query: rust ownership");
    let fact1 = g.add_node("fact: ownership prevents bugs");
    let fact2 = g.add_node("fact: borrow checker enforces ownership");
    let fact3 = g.add_node("fact: RAII pattern in C++");
    let tag_mem = g.add_node("tag: memory safety");

    g.add_edge(query, fact1, "mentions");
    g.add_edge(fact1, fact2, "related");
    g.add_edge(fact2, tag_mem, "has_tag");
    g.add_edge(fact3, tag_mem, "has_tag"); // C++ also memory safety

    // Outdate fact3
    g.remove_node(fact3);

    // Get context from query within 3 hops
    let (nodes, edges) = g.traverse_with_edges(&[query], 3);

    // Should include query, fact1, fact2, tag_mem
    assert!(nodes.iter().any(|n| g.get_node(*n).map(|x| x.data.contains("rust")).unwrap_or(false)));
    assert!(nodes.iter().any(|n| g.get_node(*n).map(|x| x.data.contains("ownership")).unwrap_or(false)));
    assert!(nodes.iter().any(|n| g.get_node(*n).map(|x| x.data.contains("borrow")).unwrap_or(false)));
    assert!(nodes.iter().any(|n| g.get_node(*n).map(|x| x.data.contains("memory safety")).unwrap_or(false)));

    // fact3 (C++) removed, should not appear
    assert!(!nodes.iter().any(|n| g.get_node(*n).map(|x| x.data.contains("C++")).unwrap_or(false)));

    // Edges returned
    assert!(!edges.is_empty());
}
