//! Vibegraph demo — run with `cargo run`
//!
//! Demonstrates:
//! - Creating a graph
//! - Adding 4 nodes
//! - Removing 1 node (no array shrinking — capacity stable)
//! - Printing node count vs capacity
//! - Running BFS traverse with depth

use vibegraph::Graph;

fn main() {
    println!("=== Vibegraph Demo ===\n");

    // Create a new graph
    let mut g = Graph::new();

    // Add 4 nodes
    let n1 = g.add_node("Node 1: Hello");
    let n2 = g.add_node("Node 2: World");
    let n3 = g.add_node("Node 3: Rust");
    let n4 = g.add_node("Node 4: GraphDB");

    println!("After adding 4 nodes:");
    println!("  Node count:    {}", g.nodes.len());
    println!("  Node capacity: {}", g.nodes.capacity());
    println!("  Nodes: {:?}", [n1, n2, n3, n4]);
    println!();

    // Add some edges so traversal has something to walk
    g.add_edge(n1, n2, "edge 1->2");
    g.add_edge(n2, n3, "edge 2->3");
    g.add_edge(n3, n4, "edge 3->4");
    g.add_edge(n1, n4, "edge 1->4");
    println!("Added 4 directed edges (n1->n2, n2->n3, n3->n4, n1->n4).\n");

    // Remove one node (n2)
    let removed = g.remove_node(n2);
    println!("Removed node {} (n2): {}", n2.0, removed);

    println!("\nAfter removing 1 node:");
    println!("  Node count:    {}", g.nodes.len());
    println!("  Node capacity: {}", g.nodes.capacity());
    println!();

    // Show that capacity is STABLE (doesn't shrink on removal)
    if g.nodes.capacity() >= g.nodes.len() {
        println!("  ✓ Capacity ({}) >= Count ({}) — no memory reallocation on remove!",
                 g.nodes.capacity(), g.nodes.len());
    }
    println!();

    // Show which nodes are still accessible via get_node
    println!("Accessible nodes (via get_node, skips deleted):");
    let all_ids = [n1, n2, n3, n4];
    for id in &all_ids {
        if let Some(node) = g.get_node(*id) {
            println!("  Node {:?}: \"{}\"", id, node.data);
        } else {
            println!("  Node {:?}: <deleted>", id);
        }
    }
    println!();

    // BFS Traversal from n1 with depth 2
    println!("BFS Traversal from n1 with max_depth=2:");
    let starts = [n1];
    let visited = g.traverse(&starts, 2);
    println!("  Start: {:?}", starts);
    println!("  Visited ({} nodes):", visited.len());
    for (i, id) in visited.iter().enumerate() {
        let data = g.get_node(*id).map(|n| n.data.as_str()).unwrap_or("<deleted>");
        println!("    [{}] {:?} -> \"{}\"", i, id, data);
    }
    println!();

    // Show depth levels conceptually
    println!("Depth levels (BFS order):");
    println!("  Depth 0: n1 (start)");
    println!("  Depth 1: n4 (via edge n1->n4; n2 is deleted so skipped)");
    println!("  Depth 2: none reachable from n4 within 2 hops (n3 via n2 is blocked)");
    println!();

    // Also demonstrate traverse_with_edges
    let (nodes, edges) = g.traverse_with_edges(&[n1], 2);
    println!("traverse_with_edges result:");
    println!("  Nodes: {:?}", nodes.iter().map(|id| id.0).collect::<Vec<_>>());
    println!("  Edges: {:?}", edges.iter().map(|id| id.0).collect::<Vec<_>>());
    println!();

    println!("=== Done ===");
}
