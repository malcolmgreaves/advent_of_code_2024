use core::iter::Iterator;
use std::cmp::Ordering;

use regex::Regex;

use crate::io_help::{self, to_int32};

// https://adventofcode.com/2024/day/3

pub fn solution_pt1() -> u64 {
    solve_inputs_mul(&io_help::read_lines("./inputs/3").collect::<Vec<String>>())
}

pub fn solution_pt2() -> u64 {
    // turn a string into a sequence of instructions
    // instructions are: mul(L,R), do, don't
    // then process as state machine
    solve_inputs_conditionals(io_help::read_lines("./inputs/3").collect::<String>())
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

#[derive(Debug)]
struct Indexed<T> {
    index: usize,
    item: T,
}

fn solve_inputs_conditionals(input: String) -> u64 {
    sum_states(&process_into_states(&input))
}

#[derive(Clone, Copy, Debug)]
pub enum State {
    MulGroup { left: i32, right: i32 },
    Do,
    Dont,
}

fn process_into_states(input_line: &str) -> Vec<State> {
    let captures = |regex: Regex, do_or_dont_state: State| -> Vec<_> {
        match do_or_dont_state {
            State::MulGroup { .. } => panic!("Should only supply State::Do or State::Dont !!"),
            State::Do => (),
            State::Dont => (),
        }
        regex
            .find_iter(input_line)
            .map(|x| Indexed {
                index: x.start(),
                item: do_or_dont_state.clone(),
            })
            .collect::<Vec<_>>()
    };

    let dont_caps = captures(Regex::new(r"don't\(\)").unwrap(), State::Dont);
    let do_caps = captures(Regex::new(r"do\(\)").unwrap(), State::Do);
    let mul_caps = regex_mul_groups(input_line.to_string())
        .iter()
        .map(|x| Indexed {
            index: x.index,
            item: State::MulGroup {
                left: x.item.left,
                right: x.item.right,
            },
        })
        .collect::<Vec<_>>();

    let mut sorting_proto_states: Vec<Indexed<State>> = Vec::new();
    sorting_proto_states.extend(dont_caps);
    sorting_proto_states.extend(do_caps);
    sorting_proto_states.extend(mul_caps);

    sorting_proto_states.sort_by(compare_indexed);

    // had to put 'Copy' derriviation to make work
    // **IS** there a way to move it?
    // it will **ONLY** be used in the return value...seems silly that this cannot be done?
    sorting_proto_states
        .iter()
        .map(|x| x.item)
        .collect::<Vec<_>>()
}

fn compare_indexed<'a, 'b, A, B>(x: &'a Indexed<A>, y: &'b Indexed<B>) -> Ordering {
    // fn compare_indexed<A, B>(x: Indexed<A>, y: Indexed<B>) -> Ordering {
    if x.index < y.index {
        Ordering::Less
    } else if x.index > y.index {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn sum_states(states: &[State]) -> u64 {
    let mut sum: u64 = 0;
    let mut active = true;
    for s in states {
        match s {
            State::MulGroup { left, right } => {
                if active {
                    let x = left * right;
                    assert!(x > 0);
                    sum += (x as u64);
                }
            }
            State::Do => {
                active = true;
            }
            State::Dont => {
                active = false;
            }
        }
    }
    sum
}

fn solve_inputs_mul(input_lines: &[String]) -> u64 {
    input_lines
        .iter()
        .map(|l| sum_mul_groups(regex_mul_groups(l.to_string()).iter().map(|x| &x.item)))
        .sum()
}

fn sum_mul_groups<'a>(mul_groups: impl Iterator<Item = &'a MulGroup>) -> u64 {
    mul_groups
        .map(|x| {
            let r = x.multiply();
            assert!(r > 0);
            r as u64
        })
        .sum()
}

fn regex_mul_groups(input_line: String) -> Vec<Indexed<MulGroup>> {
    let mul_regex = Regex::new(r"mul\([0-9]*,[0-9]*\)").unwrap();
    mul_regex
        .captures_iter(&input_line)
        .flat_map(|m| {
            m.iter()
                .flatten()
                .map(|x| {
                    let y = x.as_str();
                    let trimmed = y
                        .trim()
                        .replace("mul", "")
                        .replace('(', "")
                        .replace(')', "");
                    let bits = trimmed.split(",").map(|x| x.trim()).collect::<Vec<&str>>();
                    let (left, right) = (to_int32(bits[0]), to_int32(bits[1]));
                    Indexed {
                        index: x.start(),
                        item: MulGroup { left, right },
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_solution_pt1() {
        let example_input =
            "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))";
        let result = solve_inputs_mul(&[example_input.to_string()]);
        assert!(result == 161);
    }

    #[test]
    fn example_solution_pt2() {
        let example_input =
            "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))";
        let result = solve_inputs_conditionals(example_input.to_string());
        assert!(result == 48);
    }

    #[test]
    fn example_first_line_pt2() {
        let example_from_input_file = "^+'*>,,why()mul(229,919)&$-#^~mul(187,600)@<select()mul(430,339)<)*/-when()%mul(248,922)~+when()<do()^}%where()@select() what()why()who(809,724)mul(617,192)$*from()what(168,899)mul(333,411)()$select(){+how()%mul(284,904)when();who()mul(218,212)>[#' *+&mul(388,743):~^&;do()when()&^&^mul(415,678)>what(180,378)when()/)!#how()~&do()(((]how()[~{;what()mul(792,328)[;(,why()#mul(767,729)(what()@-why()}who()how()where(373,159),mul(91,503)select()~;where()@+;;++don't()mul(766,411)~'&%what(217,603)>why()mul(528,603);how() &who()mul(418,950)-select()mul(440,425)mul(42,798):what()[^%mul(28,566)from()<%>]//(<mul(167,358)'%](#mul(77,714)mul(748,367)]*mul(124,693);where(156,464)^(^[what()why();do()<>*mul(121,164)";
        let result = solve_inputs_conditionals(example_from_input_file.to_string());
        assert!(result == 2709546);
    }
}
