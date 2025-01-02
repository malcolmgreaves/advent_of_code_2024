use core::iter::Iterator;
use std::cmp::Ordering;

use regex::{Match, Regex};

use crate::io_help::{self, to_int};

pub fn solution_pt1() -> i32 {
    // https://adventofcode.com/2024/day/3
    solve_inputs_mul(&io_help::read_lines("./inputs/3").collect::<Vec<String>>())
}

pub fn solution_pt2() -> i32 {
    // turn a string into a sequence of instructions
    // instructions are: mul(L,R), do, don't
    // then process as state machine
    solve_inputs_conditionals(&io_help::read_lines("./inputs/3").collect::<Vec<String>>())
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

fn solve_inputs_conditionals(input_lines: &[String]) -> i32 {
    input_lines
        .iter()
        .map(|l| sum_states(&process_into_states(l)))
        .sum()
}

pub enum State {
    MulGroup { left: i32, right: i32 },
    Do,
    Dont,
}

fn process_into_states(input_line: &str) -> Vec<State> {
    panic!("unimplemented");
    let dont_regex = Regex::new(r"don't\(\)").unwrap();
    let do_regex = Regex::new(r"do\(\)").unwrap();
    let mul_regex = Regex::new(r"mul\([0-9]*,[0-9]*\)").unwrap();

    let captures = |regex: Regex| -> Vec<_> {
        regex
            .captures(input_line)
            .unwrap()
            .iter()
            .flatten()
            .collect::<Vec<_>>()
    };

    let mut dont_caps = captures(dont_regex);
    let mut do_caps = captures(do_regex);
    let mut mul_caps = captures(mul_regex);

    dont_caps.sort_by(|x, y| compare_matches_start(*x, *y));
    do_caps.sort_by(|x, y| compare_matches_start(*x, *y));
    mul_caps.sort_by(|x, y| compare_matches_start(*x, *y));

    // go through, 3 counters
    // take each one and then convert into State

    panic!("unimplemented")
}

fn compare_matches_start(x: Match, y: Match) -> Ordering {
    if x.start() < y.start() {
        Ordering::Less
    } else if x.start() > y.start() {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn sum_states(states: &[State]) -> i32 {
    let mut sum = 0;
    let mut active = true;
    for s in states {
        match s {
            State::MulGroup { left, right } => {
                if active {
                    let x = left * right;
                    sum += x;
                }
            }
            State::Do => active = true,
            State::Dont => active = false,
        }
    }
    sum
}

fn solve_inputs_mul(input_lines: &[String]) -> i32 {
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
        let result = solve_inputs_mul(&[example_input.to_string()]);
        assert!(result == 161);
    }
}
