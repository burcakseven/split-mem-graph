# Vibegraph — Fast Context Memory Graph Database for LLMs

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Vibegraph** is a lightweight, in-memory graph database skeleton written in Rust, designed as a fast **context memory** layer for LLM applications. It provides stable IDs, efficient BFS traversal, and zero-allocation removal via a freelist + generational-index design.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)
- [LLM Context Memory Use Case](#llm-context-memory-use-case)
- [Examples](#examples)
- [Testing](#testing)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Stable IDs** | `NodeId(index, generation)` — IDs never change; stale IDs return `None` |
| **No array shifting** | Remove = tombstone + freelist; capacity stays stable |
| **Generational reuse** | Freed slots get bumped generation → old IDs are zombie-proof |
| **BFS Traversal** | `traverse(starts, depth)` returns nodes in BFS order |
| **Subgraph extraction** | `traverse_with_edges` returns `(nodes, edges)` for full context |
| **Directed graph** | `edges_out` / `edges_in` per node; traverse follows outgoing |
| **No smart pointers** | Relationships stored as raw indices (`u32`) — cache-friendly |
| **Pure Rust** | Zero external dependencies |

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
vibegraph = { path = "path/to/vibegraph" }
# or once published:
# vibegraph = "0.1"
```

Or clone and run:

```bash
git clone https://github.com/burcakseven/split-mem-graph/tree/main
cd vibegraph
cargo run          # demo
cargo test         # all 34 tests pass
```

---

## Quick Start

```rust
use vibegraph::Graph;

fn main() {
    let mut g = Graph::new();

    // Add nodes
    let query = g.add_node("query: what is Rust?");
    let fact1 = g.add_node("fact: Rust has ownership");
    let fact2 = g.add_node("fact: Ownership prevents bugs");
    let tag   = g.add_node("tag: memory safety");

    // Add edges
    g.add_edge(query, fact1, "mentions");
    g.add_edge(fact1, fact2, "related");
    g.add_edge(fact1, tag, "has_tag");

    // BFS context gathering (depth 2)
    let ctx = g.traverse(&[query], 2);
    println!("Context nodes: {:?}", ctx);

    // Full subgraph (nodes + edges)
    let (nodes, edges) = g.traverse_with_edges(&[query], 2);
    println!("Nodes: {}, Edges: {}", nodes.len(), edges.len());
}
```

Run `cargo run` for an interactive demo that shows freelist stability.

---

## Core Concepts

### `NodeId` / `EdgeId`

```rust
pub struct NodeId(pub u32, pub u32); // (index, generation)
pub struct EdgeId(pub u32, pub u32);
```

- **Index**: position in `Vec<Node>` / `Vec<Edge>`
- **Generation**: increments on slot reuse — prevents zombie access

### `Node` / `Edge`

```rust
pub struct Node {
    pub id: NodeId,
    pub deleted: bool,        // tombstone flag
    pub data: String,
    pub edges_out: Vec<EdgeId>,
    pub edges_in: Vec<EdgeId>,
}

pub struct Edge {
    pub id: EdgeId,
    pub deleted: bool,
    pub source: NodeId,
    pub target: NodeId,
    pub data: String,
}
```

### `Graph`

```rust
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    // internal: free_nodes, free_edges (freelists)
}
```

---

## API Reference

### Construction

| Method | Returns | Notes |
|--------|---------|-------|
| `Graph::new()` | `Graph` | Empty graph |
| `g.add_node(data)` | `NodeId` | Reuses freelist slot if available |
| `g.add_edge(src, tgt, data)` | `EdgeId` | Validates nodes exist & not deleted |

### Access

| Method | Returns | Notes |
|--------|---------|-------|
| `g.get_node(id)` | `Option<&Node>` | `None` if OOB, deleted, or stale gen |
| `g.get_node_mut(id)` | `Option<&mut Node>` | Same |
| `g.get_edge(id)` | `Option<&Edge>` | Same |
| `g.get_edge_mut(id)` | `Option<&mut Edge>` | Same |

### Removal

| Method | Returns | Notes |
|--------|---------|-------|
| `g.remove_node(id)` | `bool` | Tombstone + freelist; cascades to incident edges |
| `g.remove_edge(id)` | `bool` | Tombstone + freelist; cleans adjacency lists |

### Traversal

| Method | Returns | Notes |
|--------|---------|-------|
| `g.traverse(starts, depth)` | `Vec<NodeId>` | BFS from starts, depth-limited, no duplicates |
| `g.traverse_with_edges(starts, depth)` | `(Vec<NodeId>, Vec<EdgeId>)` | Same + edges in subgraph |

---

## Design Decisions

### 1. Generational Indices (Zombie-Proof IDs)

```
First alloc:  NodeId(0, 0)
Remove node:  slot tombstoned, freelist.push(NodeId(0, 0))
Re-add:       NodeId(0, 1)  ← gen bumps
Old ID(0,0):  get_node() → None  ← stale!
```

This prevents the classic "ABA" bug where a stale ID silently accesses new data.

### 2. Tombstone + Freelist (No Array Shifting)

- Remove = set `deleted=true`, push ID to freelist
- `Vec` length/capacity **never shrinks**
- New `add_node` pops from freelist first

Result: **O(1)** removal, **O(1)** re-add, stable memory footprint.

### 3. No Smart Pointers for Relationships

Edges store `NodeId` (just two `u32`s), not `Rc`, `Arc`, or `Box<Node>`.  
Node adjacency lists are `Vec<EdgeId>` (indices).

**Benefits:**
- Cache-friendly (flat memory)
- No reference counting overhead
- `Copy` IDs are trivial to store/pass

### 4. Directed Graph, Outgoing Traversal

`traverse` follows `edges_out` only. For bidirectional context, call twice or use a custom traversal.

### 5. LLM-First Data Model

- `data: String` on nodes/edges holds arbitrary text
- Traversal depth = "context radius" for prompt assembly
- Deleted nodes auto-skipped → no accidental stale context

---

## LLM Context Memory Use Case

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Prompt Assembly Pipeline                               │
├─────────────────────────────────────────────────────────────┤
│  1. User query → create query node (or find existing)       │
│  2. Traverse from query with depth N                        │
│  3. Collect node.data strings                               │
│  4. Assemble prompt: [system] + context + [user]            │
│  5. (Optional) Remove old/outdated nodes                    │
└─────────────────────────────────────────────────────────────┘
```

**Example:**

```rust
let (nodes, _edges) = g.traverse_with_edges(&[query_id], 2);
let context: String = nodes
    .into_iter()
    .filter_map(|id| g.get_node(id).map(|n| n.data.clone()))
    .collect::<Vec<_>>()
    .join("\n---\n");

let prompt = format!("System: You are a helpful assistant.\n\nContext:\n{}\n\nUser: {}", context, user_query);
```

---

## Examples

### Remove and Reuse (Freelist Demo)

```rust
let mut g = Graph::new();
let a = g.add_node("A");            // NodeId(0, 0)
g.remove_node(a);                    // freelist gets NodeId(0, 0)
let a2 = g.add_node("A2");           // NodeId(0, 1) — reused, gen bumped
assert!(g.get_node(a).is_none());    // old ID invalid
assert_eq!(g.get_node(a2).unwrap().data, "A2");
```

### BFS Context Gathering

```rust
let ctx = g.traverse(&[seed], 2);
// ctx[0] = seed (depth 0)
// ctx[1..] = neighbors at depth 1, then 2
```

### Capacity Stability

```rust
for _ in 0..1000 {
    let n = g.add_node("x");
    g.remove_node(n);
}
assert!(g.nodes.len() <= 1000);      // bounded
assert!(g.nodes.capacity() >= 1);    // never shrunk
```

---

## Testing

```bash
cargo test         # 22 unit + 11 integration + 1 doc = 34 tests
cargo run          # interactive demo (main.rs)
```

Tests cover:
- Add/remove nodes & edges
- Freelist reuse & generation bumping
- Zombie prevention (stale IDs → None)
- BFS traversal (single/multi-start, depth, skip deleted)
- Capacity stability
- Full LLM context workflow simulation

---

## License

MIT — see [LICENSE](LICENSE) if present, or assume MIT for this skeleton.

---

## Contributing

This is a skeleton project. Ideas for extension:

- [ ] Bidirectional traversal (`edges_in` + `edges_out`)
- [ ] Weighted edges / shortest-path
- [ ] Persistence (serialize nodes + edges)
- [ ] Thread-safe variant (`RwLock<Graph>`)
- [ ] Attribute maps on nodes/edges

PRs welcome!
