use std::fmt::Display;

use num_format::{Locale, ToFormattedString};

use crate::{
    io_help,
    nums::{binary_enumeration, ternary_enumeration, Binary, Ternary},
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct OperativeEquation {
    result: u64,
    numbers: Vec<usize>,
}

impl Display for OperativeEquation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result_str = self.result.to_formatted_string(&Locale::en);
        let numbers_str = self
            .numbers
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        write!(f, "{}", format!("{result_str}: {numbers_str}"))
    }
}

type Equation = Vec<Element>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Element {
    Num(usize),
    Op(Operator),
}

impl Element {
    fn mul() -> Element {
        Element::Op(Operator::Mul)
    }
    fn add() -> Element {
        Element::Op(Operator::Add)
    }
    fn cat() -> Element {
        Element::Op(Operator::Cat)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operator {
    Mul, // 0
    Add, // 1
    Cat, // 2
}

fn display_equation(equation: &Equation) -> String {
    let str_elements: Vec<String> = equation
        .iter()
        .map(|x| match x {
            Element::Num(n) => n.to_string(),
            Element::Op(operator) => match operator {
                Operator::Mul => "*".to_string(),
                Operator::Add => "+".to_string(),
                Operator::Cat => "||".to_string(),
            },
        })
        .collect();
    str_elements.join(" ")
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    // total calibration result, which is the sum of the test values from just the equations that could possibly be true
    let lines = io_help::read_lines("./inputs/7").collect::<Vec<String>>();
    let operative_equations = create_equations(&lines);
    total_binary_calibration_result(&operative_equations)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

fn create_equations(lines: &[String]) -> Vec<OperativeEquation> {
    lines
        .iter()
        .map(|line| {
            let bits = line.split(": ").into_iter().collect::<Vec<_>>();
            let result = bits[0].parse::<u64>().unwrap();
            let numbers = bits[1]
                .split(" ")
                .into_iter()
                .map(|x| x.parse::<usize>().unwrap())
                .collect::<Vec<_>>();
            OperativeEquation { result, numbers }
        })
        .collect()
}

fn total_binary_calibration_result(op_eqns: &[OperativeEquation]) -> u64 {
    op_eqns
        .iter()
        .map(|op_eqn| {
            if number_of_valid_binary_equations(op_eqn) > 0 {
                op_eqn.result
            } else {
                0
            }
        })
        .sum()
}

fn number_of_valid_binary_equations(op_eqn: &OperativeEquation) -> usize {
    binary_equations_from(op_eqn)
        .into_iter()
        .filter(|eqn| valid_equation(op_eqn, eqn))
        .count()
}

fn binary_equations_from(eqn: &OperativeEquation) -> Vec<Equation> {
    if eqn.numbers.len() == 0 {
        Vec::new()
    } else {
        let base_seq = create_base_seq(eqn);

        let mut final_equations = Vec::new();
        final_equations.push(base_seq.clone());

        let operator_positions = create_operator_positions(&base_seq);

        binary_enumeration(operator_positions.len())
            .into_iter()
            .map(|binrep| {
                // println!("binrep: {binrep:?} | operator_positions: {operator_positions:?}");

                let mut new_eqn: Equation = operator_positions
                    .iter()
                    .zip(binrep.into_iter())
                    .flat_map(|(idx_op, bin)| {
                        // println!("idx_op: {idx_op} | bin: {bin:?}");
                        let new_operator: Element = match bin {
                            Binary::Zero => Element::mul(),
                            Binary::One => Element::add(),
                        };
                        // Since idx_op is always odd from base_seq's indicies,
                        // we know that -1 from it is always even.
                        // Furthermore, any odd > 0, that odd-1 is >0 too!
                        [base_seq[idx_op - 1].clone(), new_operator]
                    })
                    .collect();
                // don't forget about the last number
                new_eqn.push(base_seq[base_seq.len() - 1].clone());
                new_eqn
            })
            .collect()
    }
}

fn create_base_seq(eqn: &OperativeEquation) -> Equation {
    let mut base_seq = (0..eqn.numbers.len())
        .flat_map(|i| [Element::Num(eqn.numbers[i]), Element::mul()])
        .collect::<Vec<_>>();
    base_seq.remove(base_seq.len() - 1);
    // println!("base_seq: {base_seq:?}");
    base_seq
}

fn create_operator_positions(base_seq: &Equation) -> Vec<usize> {
    let operator_positions = (0..base_seq.len())
        .filter(|i| {
            // println!("i: {i} -> i % 2: {}", i % 2);
            i % 2 == 1
        })
        .collect::<Vec<_>>();
    operator_positions.iter().for_each(|i| match &base_seq[*i] {
        Element::Op(_) => (),
        invalid => {
            panic!("expecting all odd positions to be operators (Mul/Add), got {invalid:?} at {i}")
        }
    });
    operator_positions
}

fn valid_equation(op_eqn: &OperativeEquation, eqn: &Equation) -> bool {
    compute(eqn) == op_eqn.result
}

fn compute(eqn: &Equation) -> u64 {
    assert!(
        eqn.len() > 2,
        "Minimum equation length is 3, but found: {}",
        eqn.len()
    );

    let mut result = match &eqn[0] {
        Element::Num(x) => (*x) as u64,
        unknown => panic!("first element MUST be a number! not: {unknown:?}"),
    };
    let mut last_op: &Operator = match &eqn[1] {
        Element::Op(op) => op,
        unknown => panic!("second element MUST be an operator! not: {unknown:?}"),
    };

    (2..eqn.len()).for_each(|i| match &eqn[i] {
        Element::Num(n) => match last_op {
            Operator::Add => result += (*n) as u64,
            Operator::Mul => result *= (*n) as u64,
            Operator::Cat => {
                let concattoonated_num = format!("{result}{}", *n);
                result = concattoonated_num.parse::<u64>().unwrap();
            }
        },
        Element::Op(op) => last_op = op,
    });

    result
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/7").collect::<Vec<String>>();
    let operative_equations = create_equations(&lines);
    total_ternary_calibration_result(&operative_equations)
}

fn total_ternary_calibration_result(op_eqns: &[OperativeEquation]) -> u64 {
    op_eqns
        .iter()
        .map(|op_eqn| {
            if number_of_valid_ternary_equations(op_eqn) > 0 {
                op_eqn.result
            } else {
                0
            }
        })
        .sum()
}

fn number_of_valid_ternary_equations(op_eqn: &OperativeEquation) -> usize {
    ternary_equations_from(op_eqn)
        .into_iter()
        .filter(|eqn| valid_equation(op_eqn, eqn))
        .count()
}

fn ternary_equations_from(eqn: &OperativeEquation) -> Vec<Equation> {
    if eqn.numbers.len() == 0 {
        Vec::new()
    } else {
        let base_seq = create_base_seq(eqn);

        let mut final_equations = Vec::new();
        final_equations.push(base_seq.clone());

        let operator_positions = create_operator_positions(&base_seq);

        ternary_enumeration(operator_positions.len())
            .into_iter()
            .map(|ternrep| {
                // println!("binrep: {binrep:?} | operator_positions: {operator_positions:?}");

                let mut new_eqn: Equation = operator_positions
                    .iter()
                    .zip(ternrep.into_iter())
                    .flat_map(|(idx_op, tern)| {
                        // println!("idx_op: {idx_op} | tern: {tern:?}");
                        let new_operator: Element = match tern {
                            Ternary::Zero => Element::mul(),
                            Ternary::One => Element::add(),
                            Ternary::Two => Element::cat(),
                        };
                        // Since idx_op is always odd from base_seq's indicies,
                        // we know that -1 from it is always even.
                        // Furthermore, any odd > 0, that odd-1 is >0 too!
                        [base_seq[idx_op - 1].clone(), new_operator]
                    })
                    .collect();
                // don't forget about the last number
                new_eqn.push(base_seq[base_seq.len() - 1].clone());
                new_eqn
            })
            .collect()
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        190: 10 19
        3267: 81 40 27
        83: 17 5
        156: 15 6
        7290: 6 8 6 15
        161011: 16 10 13
        192: 17 8 14
        21037: 9 7 18 13
        292: 11 6 16 20
    "};

    lazy_static! {
        static ref EXAMPLE_INPUT: Vec<OperativeEquation> = vec![
            OperativeEquation {
                result: 190,
                numbers: vec![10, 19]
            },
            OperativeEquation {
                result: 3267,
                numbers: vec![81, 40, 27]
            },
            OperativeEquation {
                result: 83,
                numbers: vec![17, 5]
            },
            OperativeEquation {
                result: 156,
                numbers: vec![15, 6]
            },
            OperativeEquation {
                result: 7290,
                numbers: vec![6, 8, 6, 15]
            },
            OperativeEquation {
                result: 161011,
                numbers: vec![16, 10, 13]
            },
            OperativeEquation {
                result: 192,
                numbers: vec![17, 8, 14]
            },
            OperativeEquation {
                result: 21037,
                numbers: vec![9, 7, 18, 13]
            },
            OperativeEquation {
                result: 292,
                numbers: vec![11, 6, 16, 20]
            },
        ];
    }

    #[test]
    fn construct() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let created = create_equations(&lines);
        let example: &Vec<OperativeEquation> = &EXAMPLE_INPUT;
        assert_eq!(created, *example);
    }

    #[test]
    fn known_equations_from_0() {
        let example: &OperativeEquation = &EXAMPLE_INPUT[0];
        let expected = vec![Element::Num(10), Element::mul(), Element::Num(19)];

        let actual: Vec<Equation> = binary_equations_from(example);
        println!("ACTUAL EQUATIONS: {actual:?}");
        assert_eq!(actual.len(), 2, "actual equations: {:?}", actual);

        println!("compute[0]: {}", compute(&actual[0]));
        println!("compute[1]: {}", compute(&actual[1]));

        let actual_valid: Vec<Equation> = actual
            .into_iter()
            .filter(|x| valid_equation(example, x))
            .collect();
        assert_eq!(actual_valid.len(), 1);
        assert_eq!(actual_valid[0], expected);
    }

    #[test]
    fn known_total_calibration_result_example() {
        let op_eqns: &[OperativeEquation] = &EXAMPLE_INPUT;
        let acutal = total_binary_calibration_result(op_eqns);
        assert_eq!(acutal, 3749);

        let op_eqns = vec![OperativeEquation {
            result: 1069,
            numbers: vec![4, 2, 5, 2, 989],
        }];
        let actual = total_binary_calibration_result(&op_eqns);
        assert_eq!(actual, op_eqns[0].result);
    }

    #[test]
    fn compute_simple_example() {
        // 1 + 2 * 3 * 4 + 5 --> (evaluates as): ((((1+2) * 3) * 4) + 5 = 41
        let eqn: Equation = vec![
            Element::Num(1),
            Element::Op(Operator::Add),
            Element::Num(2),
            Element::Op(Operator::Mul),
            Element::Num(3),
            Element::Op(Operator::Mul),
            Element::Num(4),
            Element::Op(Operator::Add),
            Element::Num(5),
        ];
        let actual = compute(&eqn);
        assert_eq!(actual, 41);
    }

    #[test]
    fn pt1_soln_example() {
        let operative_equation_strs = [
            ("65816524826: 3 29 7 451 5 64 26 1 1 9", false),
            ("35544: 19 78 6 345 9", true),
            ("41: 1 2 3 4 5", true),
            ("30: 1 2 3 4 5", false),
            ("4373193255: 9 49 911 933 3 235", false),
        ];

        operative_equation_strs
            .into_iter()
            .for_each(|(x, has_solution)| {
                // println!("example: {x}");
                let op_eqn = {
                    let op_eqns = create_equations(&[x.to_string()]);
                    assert_eq!(op_eqns.len(), 1);
                    op_eqns
                        .into_iter()
                        .next()
                        .expect("Expecting exactly one operative equation from one line.")
                };
                // println!("{op_eqn:?}");
                println!("{op_eqn}\n--");
                let n_valid = binary_equations_from(&op_eqn)
                    .iter()
                    .map(|x| {
                        let line = format!(
                            "\t{} = {}",
                            display_equation(x),
                            compute(x).to_formatted_string(&Locale::en),
                        );
                        let final_line = if valid_equation(&op_eqn, &x) {
                            format!("{} =================> IS VALID !!!!", line)
                        } else {
                            line
                        };
                        println!("{final_line}");
                        x
                    })
                    .filter(|eqn| valid_equation(&op_eqn, eqn))
                    .count();
                println!(
                    "# valid equations: {}",
                    n_valid.to_formatted_string(&Locale::en)
                );

                let actual = total_binary_calibration_result(&[op_eqn.clone()]);
                println!("TCR: {actual}");
                if has_solution {
                    assert_eq!(
                        actual, op_eqn.result,
                        "has solution but total calibration result was invalid"
                    );
                } else {
                    assert_eq!(
                        actual, 0,
                        "does not have solution thus no total calibration result"
                    );
                }
            })
    }

    #[test]
    fn example_total_ternary_calibration_result() {
        let op_eqns: &[OperativeEquation] = &EXAMPLE_INPUT;
        let acutal = total_ternary_calibration_result(op_eqns);
        assert_eq!(acutal, 11387);
    }
    
}
