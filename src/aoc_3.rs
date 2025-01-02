use core::iter::Iterator;
use std::fmt::Display;

use regex::Regex;

use crate::io_help::{self, to_int};

pub fn solution() -> i32 {
    // https://adventofcode.com/2024/day/3
    solve_inputs(&io_help::read_lines("./inputs/3").collect::<Vec<String>>())
}

pub struct MulGroup {
    left: i32,
    right: i32,
}

impl MulGroup {
    pub fn multiply(&self) -> i32 {
        self.left * self.right
    }
}

fn solve_inputs(input_lines: &[String]) -> i32 {
    input_lines
        .iter()
        .map(|l| sum_mul_groups(&regex_mul_groups(l.to_string())))
        .sum()
}

fn sum_mul_groups(mul_groups: &[MulGroup]) -> i32 {
    mul_groups.iter().map(|x| x.multiply()).sum()
}

fn regex_mul_groups(input_line: String) -> Vec<MulGroup> {
    let mul_regex = Regex::new(r"mul\([0-9]*,[0-9]*\)").unwrap();
    mul_regex
        .captures_iter(&input_line)
        .flat_map(|m| {
            m.iter()
                .flatten()
                .map(|x| {
                    let y = x.as_str();
                    // println!("{}", format!("match: {y}"));
                    let trimmed = y
                        .trim()
                        .replace("mul", "")
                        .replace('(', "")
                        .replace(')', "");
                    // println!("{}", format!("trimmed: {trimmed}"));
                    let bits = trimmed.split(",").map(|x| x.trim()).collect::<Vec<&str>>();
                    let (left, right) = (to_int(bits[0]), to_int(bits[1]));
                    // let (_, [left, right]) = Regex::new(r"\d+").unwrap().captures(x.as_str()).unwrap().extract();
                    // println!("{}", format!("left: {left} | right: {right}"));
                    MulGroup { left, right }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution() {
        let example_input =
            "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))";
        let result = solve_inputs(&[example_input.to_string()]);
        assert!(result == 161);
    }
}
