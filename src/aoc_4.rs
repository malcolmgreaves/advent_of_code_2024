use core::iter::Iterator;
use std::cmp::Ordering;

use regex::Regex;

use crate::io_help::{self, to_int};

// https://adventofcode.com/2024/day/4

pub fn solution_pt1() -> i32 {
    &io_help::read_lines("./inputs/4").collect::<Vec<String>>()
}

pub fn solution_pt2() -> i32 {
    io_help::read_lines("./inputs/4").collect::<String>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution_pt1() {
        let result = solve_inputs_mul(&[example_input.to_string()]);
        assert!(result == 161);
    }

    #[test]
    fn example_solution_pt2() {
        let result = solve_inputs_conditionals(example_input.to_string());
        assert!(result == 48);
    }
}
