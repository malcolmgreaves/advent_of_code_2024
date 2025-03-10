use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::utils::{Res, proc_elements_result};

use crate::io_help;

// pub struct PrintRulesQueue<'a> {
//     rules: &'a [PrintRule],
//     queue: &'a [PrintJob<'a>],
#[derive(Debug, PartialEq)]
pub struct PrintRulesQueue {
    rules: Vec<PrintRule>,
    queue: Vec<PrintJob>,
}

#[derive(Debug, PartialEq)]
pub struct PrintRule {
    before: Page,
    after: Page,
}

// pub type PrintJob<'a> = &'a [Page];
pub type PrintJob = Vec<Page>;

pub type Page = usize;

fn construct_before_afters_map(print_queue_rules: &[PrintRule]) -> WorkingPrintRules {
    let mut rules: HashMap<Page, HashSet<Page>> = HashMap::new();
    print_queue_rules
        .iter()
        .for_each(|PrintRule { before, after }| {
            match rules.get_mut(before) {
                Some(existing_afters) => {
                    existing_afters.insert(*after);
                }
                None => {
                    let mut new_afters: HashSet<Page> = HashSet::new();
                    new_afters.insert(*after);
                    rules.insert(*before, new_afters);
                }
            };
        });
    WorkingPrintRules {
        before_afters: rules,
    }
}

struct WorkingPrintRules {
    before_afters: HashMap<Page, HashSet<Page>>,
}

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/5").collect::<Vec<String>>();
    match create_print_rules_queue_from_input(&lines) {
        Ok(print_queue_rules) => sum_of_middles_of_valid_print_queues(&print_queue_rules),
        Err(error) => panic!("Could not create print queue rules from inputs/5 !! {error}"),
    }
}

fn create_print_rules_queue_from_input(lines: &[String]) -> Res<PrintRulesQueue> {
    let index_separation = {
        let mut idx = None;
        for i in 0..lines.len() {
            if lines[i] == "" {
                idx = Some(i);
                break;
            }
        }
        match idx {
            Some(i) => Ok(i),
            None => Err("Could not find index separating rules from print queue!"),
        }
    };

    let rules_part = &lines[0..index_separation?];
    let queue_part = &lines[(index_separation?) + 1..lines.len()];

    let rules_final = proc_elements_result(
        |x: &String| -> Res<PrintRule> { create_rule(x.to_string()) },
        &rules_part,
    )?;

    let queue_final = proc_elements_result(
        |x: &String| -> Res<Vec<Page>> { create_queue(x.to_string()) },
        &queue_part,
    )?;

    Ok(PrintRulesQueue {
        rules: rules_final,
        queue: queue_final,
    })
}

fn create_rule(rule: String) -> Res<PrintRule> {
    // "a|b" ==> PrintRule{before: a, after: b}
    match rule.find("|") {
        Some(index) => {
            let before = rule[0..index].parse::<usize>()?;
            let after = rule[(index + 1)..rule.len()].parse::<usize>()?;
            Ok(PrintRule { before, after })
        }
        None => Err(format!("Could not find separator (|) for rule: {rule}").into()),
    }
}

fn create_queue(queue: String) -> Res<Vec<Page>> {
    // "a,b,c,d" ==> vec![a,b,c,d]
    // let bits = queue.split(",");
    // let mut pages: Vec<Page> = Vec::new();
    // for x in bits.into_iter() {
    //     match x.to_string().parse::<usize>() {
    //         Ok(page) => {
    //             pages.push(page)
    //         },
    //         Err(error) => {
    //             return Err(error.to_string().into());
    //         },
    //     }
    // }
    // Ok(pages)
    proc_elements_result(
        |x: &&str| -> Res<usize> {
            match x.parse::<usize>() {
                Ok(page) => Ok(page),
                Err(error) => Err(error.to_string().into()),
            }
        },
        &queue.split(",").collect::<Vec<_>>(),
    )
}

fn sum_of_middles_of_valid_print_queues(print_queue_rules: &PrintRulesQueue) -> u64 {
    sum_of_middles_of_valid_print_queues_(
        &construct_before_afters_map(&print_queue_rules.rules),
        &print_queue_rules.queue,
    )
}

fn sum_of_middles_of_valid_print_queues_(prq: &WorkingPrintRules, print_queue: &[PrintJob]) -> u64 {
    let valid_queues: Vec<&PrintJob> = {
        let mut valids: Vec<&PrintJob> = Vec::new();
        print_queue.iter().for_each(|seq: &PrintJob| {
            if check_is_valid_print_sequence(&prq, seq) {
                valids.push(seq);
            }
        });
        valids
    };

    let mids = middles_of(&valid_queues);
    mids.iter().map(|x| *x as u64).sum()
}

fn check_is_valid_print_sequence(prq: &WorkingPrintRules, seq: &[Page]) -> bool {
    let mut visited: HashSet<Page> = HashSet::new();

    for page in seq.iter() {
        let is_valid = match prq.before_afters.get(page) {
            Some(afters) => {
                for v in visited.iter() {
                    if afters.contains(v) {
                        // println!("\tINVALID! rule is: {} is before {}, but found violation in this sequence: {:?}", page, v, seq);
                        return false;
                    }
                }
                true
            }
            None => true,
        };
        if !is_valid {
            return false;
        } else {
            visited.insert(*page);
        }
    }
    true
}

fn middles_of(queue: &[&PrintJob]) -> Vec<Page> {
    queue
        .iter()
        .map(|seq| {
            // rounds down, still usize
            let idx_middle = seq.len() / 2;
            seq[idx_middle]
        })
        .collect::<Vec<_>>()
}

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/5").collect::<Vec<String>>();
    match create_print_rules_queue_from_input(&lines) {
        Ok(print_queue_rules) => sum_of_middles_on_only_invalid_fixed_queues(&print_queue_rules),
        Err(error) => panic!("Could not create print queue rules from inputs/5 !! {error}"),
    }
}

fn sum_of_middles_on_only_invalid_fixed_queues(print_queue_rules: &PrintRulesQueue) -> u64 {
    let prq = construct_before_afters_map(&print_queue_rules.rules);
    let fixed_print_queue = print_queue_rules
        .queue
        .iter()
        .flat_map(|x| {
            // uses stable sort => won't change correct print queues
            if !check_is_valid_print_sequence(&prq, x) {
                Some(fix_queue(&prq, x))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    sum_of_middles_of_valid_print_queues_(&prq, &fixed_print_queue)
}

fn fix_queue(prq: &WorkingPrintRules, seq: &[Page]) -> PrintJob {
    // structure this as a sort
    // the rules are the cmp actions

    let x_before_y = |page_x: &usize, page_y: &usize| -> bool {
        match prq.before_afters.get(page_x) {
            Some(afters_x) => afters_x.contains(page_y),
            None => false,
        }
    };

    // println!("[fixing] BEFORE: {seq:?}");
    let mut fixing = seq.to_vec();
    fixing.sort_by(|page_a, page_b| {

        let a_before_b = x_before_y(&page_a, &page_b);
        let b_before_a = x_before_y(&page_b, &page_a);

        match (a_before_b, b_before_a) {
            (true, true) => panic!(
                "Cannot have rules where two pages both appear before each other! (A:{page_a} and B:{page_b})\n[A] {page_a} -> {:?}\n[B] {page_b} -> {:?}",
                prq.before_afters.get(page_a), prq.before_afters.get(page_b)
            ),
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        }
    });
    // println!("[fixing] AFTER: {fixing:?}");
    fixing
}

#[cfg(test)]
mod test {

    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

    use super::*;

    lazy_static! {
        static ref EXAMPLE_INPUT_AOC: PrintRulesQueue = {
            PrintRulesQueue {
                rules: vec![
                    PrintRule {
                        before: 47,
                        after: 53,
                    },
                    PrintRule {
                        before: 97,
                        after: 13,
                    },
                    PrintRule {
                        before: 97,
                        after: 61,
                    },
                    PrintRule {
                        before: 97,
                        after: 47,
                    },
                    PrintRule {
                        before: 75,
                        after: 29,
                    },
                    PrintRule {
                        before: 61,
                        after: 13,
                    },
                    PrintRule {
                        before: 75,
                        after: 53,
                    },
                    PrintRule {
                        before: 29,
                        after: 13,
                    },
                    PrintRule {
                        before: 97,
                        after: 29,
                    },
                    PrintRule {
                        before: 53,
                        after: 29,
                    },
                    PrintRule {
                        before: 61,
                        after: 53,
                    },
                    PrintRule {
                        before: 97,
                        after: 53,
                    },
                    PrintRule {
                        before: 61,
                        after: 29,
                    },
                    PrintRule {
                        before: 47,
                        after: 13,
                    },
                    PrintRule {
                        before: 75,
                        after: 47,
                    },
                    PrintRule {
                        before: 97,
                        after: 75,
                    },
                    PrintRule {
                        before: 47,
                        after: 61,
                    },
                    PrintRule {
                        before: 75,
                        after: 61,
                    },
                    PrintRule {
                        before: 47,
                        after: 29,
                    },
                    PrintRule {
                        before: 75,
                        after: 13,
                    },
                    PrintRule {
                        before: 53,
                        after: 13,
                    },
                ],
                queue: vec![
                    vec![75, 47, 61, 53, 29],
                    vec![97, 61, 53, 29, 13],
                    vec![75, 29, 13],
                    vec![75, 97, 47, 61, 53],
                    vec![61, 13, 29],
                    vec![97, 13, 75, 29, 47],
                ],
            }
        };
    }

    #[test]
    fn create_rules_queue_from_input() {
        let input_str = indoc! {"
            47|53
            97|13
            97|61
            97|47
            75|29
            61|13
            75|53
            29|13
            97|29
            53|29
            61|53
            97|53
            61|29
            47|13
            75|47
            97|75
            47|61
            75|61
            47|29
            75|13
            53|13

            75,47,61,53,29
            97,61,53,29,13
            75,29,13
            75,97,47,61,53
            61,13,29
            97,13,75,29,47
        "};

        let lines = read_lines_in_memory(input_str).collect::<Vec<_>>();
        // for l in lines.iter() {
        //     println!("[L] '{l}', {}", l == "");
        // }
        match create_print_rules_queue_from_input(&lines) {
            Ok(print_rules_queue) => {
                let expected: &PrintRulesQueue = &EXAMPLE_INPUT_AOC;
                assert_eq!(print_rules_queue, *expected, "actual != expected")
            }
            Err(error) => panic!("{}", error),
        }
    }

    #[test]
    fn pt1_first_valid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[0];
        assert!(
            check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "First should be valid: {:?}",
            seq,
        );
    }
    #[test]
    fn pt1_second_valid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[1];
        assert!(
            check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "Second should be valid: {:?}",
            seq,
        );
    }

    #[test]
    fn pt1_third_valid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[2];
        assert!(
            check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "Third should be valid: {:?}",
            seq,
        );
    }

    #[test]
    fn pt1_fourth_invalid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[3];
        assert!(
            !check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "Fourth should be invalid: {:?}",
            seq,
        );
    }

    #[test]
    fn pt1_fifth_invalid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[4];
        assert!(
            !check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "Fifth should be invalid: {:?}",
            seq,
        );
    }

    #[test]
    fn pt1_sixth_invalid() {
        let seq = &EXAMPLE_INPUT_AOC.queue[5];
        assert!(
            !check_is_valid_print_sequence(
                &construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules),
                &seq,
            ),
            "Sixth should be invalid: {:?}",
            seq,
        );
    }

    #[test]
    fn pt1_sum_of_middles_of_valid_print_queues() {
        let result = sum_of_middles_of_valid_print_queues(&EXAMPLE_INPUT_AOC);
        assert_eq!(143, result as i32, "Actual != Expected");
    }

    #[test]
    fn pt2_no_fix_queue_first() {
        let good_queue = vec![75, 47, 61, 53, 29];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &good_queue);
        assert_eq!(fixed, good_queue, "actual != expected");
    }

    #[test]
    fn pt2_no_fix_queue_second() {
        let good_queue = vec![97, 61, 53, 29, 13];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &good_queue);
        assert_eq!(fixed, good_queue, "actual != expected");
    }

    #[test]
    fn pt2_no_fix_queue_third() {
        let good_queue = vec![75, 29, 13];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &good_queue);
        assert_eq!(fixed, good_queue, "actual != expected");
    }

    #[test]
    fn pt2_fix_queue_fourth() {
        let bad_queue = vec![75, 97, 47, 61, 53];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &bad_queue);
        let expected = vec![97, 75, 47, 61, 53];
        assert_eq!(fixed, expected, "actual != expected");
    }

    #[test]
    fn pt2_fix_queue_fifth() {
        let bad_queue = vec![61, 13, 29];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &bad_queue);
        let expected = vec![61, 29, 13];
        assert_eq!(fixed, expected, "actual != expected");
    }

    #[test]
    fn pt2_fix_queue_sixth() {
        let bad_queue = vec![97, 13, 75, 29, 47];
        let prq = construct_before_afters_map(&EXAMPLE_INPUT_AOC.rules);
        let fixed = fix_queue(&prq, &bad_queue);
        let expected = vec![97, 75, 47, 29, 13];
        assert_eq!(fixed, expected, "actual != expected");
    }

    #[test]
    fn pt2_sum_of_middles_on_only_invalid_fixed_queues() {
        assert_eq!(
            sum_of_middles_on_only_invalid_fixed_queues(&EXAMPLE_INPUT_AOC),
            123,
            "actual != expected"
        );
    }
}
