use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
};

pub trait NodeConstraints: Debug + Clone + PartialEq + Eq + Hash {}
impl<T: Debug + Clone + PartialEq + Eq + Hash> NodeConstraints for T {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node<T: NodeConstraints>(pub T);

#[allow(dead_code)]
pub trait Graph<T: NodeConstraints> {
    fn neighbors(&self, node: &Node<T>) -> Option<&[Node<T>]>;
    fn vertices(&self) -> Vec<&Node<T>>;
    fn edges(&self) -> Vec<(&Node<T>, &Node<T>)>;
}

pub struct SparseGraph<T: NodeConstraints> {
    connections: HashMap<Node<T>, Vec<Node<T>>>,
}

#[allow(dead_code)]
impl<T: NodeConstraints> SparseGraph<T> {
    pub fn new() -> SparseGraph<T> {
        SparseGraph {
            connections: HashMap::new(),
        }
    }
    pub fn with_capacity(capacity: usize) -> SparseGraph<T> {
        SparseGraph {
            connections: HashMap::with_capacity(capacity),
        }
    }
}

impl<T: NodeConstraints> Graph<T> for SparseGraph<T> {
    fn neighbors(&self, node: &Node<T>) -> Option<&[Node<T>]> {
        match self.connections.get(node) {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn vertices(&self) -> Vec<&Node<T>> {
        self.connections.keys().collect()
    }

    fn edges(&self) -> Vec<(&Node<T>, &Node<T>)> {
        self.connections
            .iter()
            .flat_map(|(vertex, neighbors)| neighbors.iter().map(move |n| (vertex, n)))
            .collect()
    }
}

#[allow(dead_code)]
pub trait GraphBuilder<T: NodeConstraints, G: Graph<T>> {
    fn new() -> Self;
    fn with_capacity(capacity: usize) -> Self;
    fn insert(&mut self, source: Node<T>, destination: Node<T>);
    fn to_graph(self) -> G;
}

pub struct SparseBuilder<T: NodeConstraints> {
    connections: HashMap<Node<T>, HashSet<Node<T>>>,
}

impl<T: NodeConstraints> GraphBuilder<T, SparseGraph<T>> for SparseBuilder<T> {
    fn new() -> Self {
        SparseBuilder {
            connections: HashMap::new(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        SparseBuilder {
            connections: HashMap::with_capacity(capacity),
        }
    }

    fn insert(&mut self, source: Node<T>, destination: Node<T>) {
        match self.connections.get_mut(&source) {
            Some(existing) => {
                existing.insert(destination);
            }
            None => {
                let mut n = HashSet::new();
                n.insert(destination);
                self.connections.insert(source, n);
            }
        }
    }

    fn to_graph(self) -> SparseGraph<T> {
        SparseGraph {
            connections: self
                .connections
                .into_iter()
                .map(|(vertex, neighborhood)| (vertex, neighborhood.into_iter().collect()))
                .collect(),
        }
    }
}
