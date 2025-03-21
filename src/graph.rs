use std::{collections::HashMap, fmt::Debug, hash::Hash};

trait NodeConstraints: Debug + Clone + PartialEq + Eq + Hash {}
impl<T: Debug + Clone + PartialEq + Eq + Hash> NodeConstraints for T {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node<T: NodeConstraints> {
    data: T,
}

pub trait Graph<T: NodeConstraints> {
    fn neighbors(&self, node: &Node<T>) -> Option<&[Node<T>]>;
    fn vertices(&self) -> Vec<&Node<T>>;
}

pub struct SparseGraph<T: NodeConstraints> {
    connections: HashMap<Node<T>, Vec<Node<T>>>,
}

impl<T: NodeConstraints> Graph<T> for SparseGraph<T> {
    fn neighbors(&self, node: &Node<T>) -> Option<&[Node<T>]> {
        match self.connections.get(node) {
            Some(values) => Some(values),
            None => None,
        }
    }

    fn vertices(&self) -> Vec<&Node<T>> {
        self.connections.keys().collect::<Vec<_>>()
    }
}
