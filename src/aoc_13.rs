use crate::{
    io_help,
    utils::{argmin, collect_results},
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Button {
    x: u64,
    y: u64,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Location {
    x: u64,
    y: u64,
}

impl Location {
    fn increment(&self, val: u64) -> Self {
        Self {
            x: self.x + val,
            y: self.y + val,
        }
    }
}

// A Claw Machine.
//
// Each press of its A or B buttons will move the claw in the specified X & Y directions.
// The prize is located at a specific X,Y coordinate.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ClawMach {
    a: Button,
    b: Button,
    prize: Location,
}

impl ClawMach {
    fn constraint_x(&self) -> (u64, u64, u64) {
        (self.a.x, self.b.x, self.prize.x)
    }

    fn constraint_y(&self) -> (u64, u64, u64) {
        (self.a.y, self.b.y, self.prize.y)
    }
}

fn construct(lines: &[String]) -> Result<Vec<ClawMach>, String> {
    let (claw_machines, parse_errors) = collect_results(lines.chunks(4).map(parse_claw_machine));
    if parse_errors.len() != 0 {
        let error = format!(
            "Created {} ClawMach structs but found {} parsing errors!:\n\t{}",
            claw_machines.len(),
            parse_errors.len(),
            parse_errors.join("\t\t")
        );
        Result::Err(error)
    } else {
        Result::Ok(claw_machines)
    }
}

fn parse_claw_machine(chunk: &[String]) -> Result<ClawMach, String> {
    if chunk.len() < 3 {
        return Result::Err(format!(
            "Chunk must be at least 3 lines to parse into Claw Machine struct. Only found: {}",
            chunk.len()
        ));
    }
    Result::Ok(ClawMach {
        a: parse_button(&chunk[0])?,
        b: parse_button(&chunk[1])?,
        prize: parse_prize(&chunk[2])?,
    })
}

fn parse_button(s: &str) -> Result<Button, String> {
    //  Button A: X+94, Y+34
    match s.split(": ").last() {
        Some(x) => {
            let parts = x.trim().split(", ").collect::<Vec<_>>();
            if parts.len() != 2 {
                Result::Err(format!("Button: Invalid X and Y specifier: '{}'", s))
            } else {
                let x = parse_num_from("+", parts[0])?;
                let y = parse_num_from("+", parts[1])?;
                Result::Ok(Button { x, y })
            }
        }
        None => Result::Err(format!("Button: Missing colon ':' in format: '{}'", s)),
    }
}

fn parse_prize(s: &str) -> Result<Location, String> {
    // Prize: X=8400, Y=5400
    match s.split(": ").last() {
        Some(x) => {
            let parts = x.trim().split(", ").collect::<Vec<_>>();
            if parts.len() != 2 {
                Result::Err(format!("Prize: Invalid X and Y specifier: '{}'", s))
            } else {
                let x = parse_num_from("=", parts[0])?;
                let y = parse_num_from("=", parts[1])?;
                Result::Ok(Location { x, y })
            }
        }
        None => Result::Err(format!("Prize: Missing colon ':' in format: '{}'", s)),
    }
}

fn parse_num_from(symbol: &str, s: &str) -> Result<u64, String> {
    match s.split(symbol).last() {
        Some(x) => x.parse::<u64>().map_err(|e| e.to_string()),
        None => Result::Err(format!(
            "missing separator symbol '{symbol}' in input string: '{s}'"
        )),
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/13").collect::<Vec<String>>();
    let claw_machines = construct(&lines).unwrap();
    calculate_solution(claw_machines.iter().map(solve_brute_force))
}

fn calculate_solution(solved: impl Iterator<Item = Option<Press>>) -> u64 {
    solved.fold(0, |s, maybe_press| match maybe_press {
        Some(press) => press.cost() + s,
        None => s,
    })
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Press {
    a: u64,
    b: u64,
}

impl Press {
    fn cost(&self) -> u64 {
        cost(&self, 3, 1)
    }
}

fn cost(p: &Press, a_cost: u64, b_cost: u64) -> u64 {
    p.a * a_cost + p.b * b_cost
}

fn locate(claw: &ClawMach, p: &Press) -> Location {
    let x = p.a * claw.a.x + p.b * claw.b.x;
    let y = p.a * claw.a.y + p.b * claw.b.y;
    Location { x, y }
}

fn verify(claw: &ClawMach, p: &Press) -> bool {
    locate(claw, p) == claw.prize
}

fn brute_force(limit: u64, claw: &ClawMach) -> Vec<Press> {
    (0..limit)
        .flat_map(|a| {
            (0..limit).flat_map(move |b| {
                let p = Press { a, b };
                if verify(claw, &p) { Some(p) } else { None }
            })
        })
        .collect()
}

fn solve_brute_force(claw: &ClawMach) -> Option<Press> {
    let mut possibilities = brute_force(100, claw);
    if possibilities.len() == 0 {
        None
    } else {
        let index_of_min = argmin(
            &possibilities
                .iter()
                .map(|p| (p, p.cost()))
                .collect::<Vec<_>>(),
            |(_, c)| *c,
        );
        Some(possibilities.swap_remove(index_of_min))
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

const MODIFIER_PART_2: u64 = 10000000000000;

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/13").collect::<Vec<String>>();
    let claw_machines = construct(&lines)
        .unwrap()
        .into_iter()
        .map(|claw| ClawMach {
            a: claw.a,
            b: claw.b,
            prize: claw.prize.increment(10000000000000),
        })
        .collect::<Vec<_>>();
    calculate_solution(claw_machines.iter().map(solve_dynamic_programming))
}

fn solve_dynamic_programming(claw: &ClawMach) -> Option<Press> {
    // Button A: X+94, Y+34
    // Button B: X+22, Y+67
    // Prize: X=10000000008400, Y=10000000005400
    //  ==>
    //      94*A + 22*B = 10000000008400
    //      34*A + 67*B = 10000000005400
    let (a, b, _) = ilp_solve((3, 1), claw.constraint_x(), claw.constraint_y(), true);
    let p = Press { a, b };
    if verify(claw, &p) { Some(p) } else { None }
}

fn ilp_solve(
    (c_a, c_b): (u64, u64),               /* button token costs */
    (a_x, b_x, x_total): (u64, u64, u64), /* button A */
    (a_y, b_y, y_total): (u64, u64, u64), /* button B */
    is_exact: bool,
) -> (u64, u64, u64) {
    let constraint = if is_exact {
        |a: u64, b: u64| -> bool { a == b } as fn(u64, u64) -> bool
    } else {
        |a: u64, b: u64| -> bool { a <= b } as fn(u64, u64) -> bool
    };

    let mut optimal_a = 0;
    let mut optimal_b = 0;
    let mut optimal = u64::MIN;

    // let mut solutions = Vec::new();

    let lower_bound_x = if a_x > 0 {
        (x_total / a_x).saturating_sub((x_total / a_x))
    } else {
        0
    };
    // iterate through feasiable values for button A (x1)
    for x in lower_bound_x..=(x_total / a_x) {
        if a_y * x > y_total {
            // stop if we've violated constrainty
            // println!("\tstop[1]: a_y * x > y_total: {a_y} * {x} = {} > {y_total}", a_y * x);
            break;
        }

        let lower_bound_y = if b_x > 0 {
            (x_total.saturating_sub(a_x * x)) / b_x
        } else {
            0
        };
        // iterate through feasiable values for button B (x2)
        for y in lower_bound_y..=((x_total - a_x * x) / b_x) {
            if a_y * x + b_y * y > y_total {
                // println!("\tstop[2]: a_y * x + b_y * y > y_total: {a_y} * {x} + {b_y} * {y} = {} > {y_total}", a_y * x + b_y * y);
                // stop if we've violated constraint
                break;
            }

            // check if constraints are satisfied
            if constraint(a_x * x + b_x * y, x_total) && constraint(a_y * x + b_y * y, y_total) {
                let objective = c_a * x + c_b * y;
                // Update optimal solution if it's the smallest found
                if objective > optimal {
                    optimal_a = x;
                    optimal_b = y;
                    optimal = objective;
                    // let zzz = (optimal_a.clone(), optimal_b.clone(), optimal.clone());
                    // println!("\tSOLUTION: {zzz:?}");
                    // solutions.push(zzz);
                }
            }
        }
    }

    (optimal_a, optimal_b, optimal)
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR_PART_1: &str = indoc! {"
        Button A: X+94, Y+34
        Button B: X+22, Y+67
        Prize: X=8400, Y=5400

        Button A: X+26, Y+66
        Button B: X+67, Y+21
        Prize: X=12748, Y=12176

        Button A: X+17, Y+86    
        Button B: X+84, Y+37
        Prize: X=7870, Y=6450

        Button A: X+69, Y+23
        Button B: X+27, Y+71
        Prize: X=18641, Y=10279
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED_PART_1: Vec<ClawMach> = vec![
            ClawMach {
                a: Button { x: 94, y: 34 },
                b: Button { x: 22, y: 67 },
                prize: Location { x: 8400, y: 5400 }
            },
            ClawMach {
                a: Button { x: 26, y: 66 },
                b: Button { x: 67, y: 21 },
                prize: Location { x: 12748, y: 12176 }
            },
            ClawMach {
                a: Button { x: 17, y: 86 },
                b: Button { x: 84, y: 37 },
                prize: Location { x: 7870, y: 6450 }
            },
            ClawMach {
                a: Button { x: 69, y: 23 },
                b: Button { x: 27, y: 71 },
                prize: Location { x: 18641, y: 10279 }
            },
        ];
    }

    const EXAMPLE_INPUT_STR_PART_2: &str = indoc! {"
        Button A: X+94, Y+34
        Button B: X+22, Y+67
        Prize: X=10000000008400, Y=10000000005400

        Button A: X+26, Y+66
        Button B: X+67, Y+21
        Prize: X=10000000012748, Y=10000000012176

        Button A: X+17, Y+86
        Button B: X+84, Y+37
        Prize: X=10000000007870, Y=10000000006450

        Button A: X+69, Y+23
        Button B: X+27, Y+71
        Prize: X=10000000018641, Y=10279
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED_PART_2: Vec<ClawMach> = vec![
            ClawMach {
                a: Button { x: 94, y: 34 },
                b: Button { x: 22, y: 67 },
                prize: Location {
                    x: 10000000008400,
                    y: 10000000005400
                }
            },
            ClawMach {
                a: Button { x: 26, y: 66 },
                b: Button { x: 67, y: 21 },
                prize: Location {
                    x: 10000000012748,
                    y: 10000000012176
                }
            },
            ClawMach {
                a: Button { x: 17, y: 86 },
                b: Button { x: 84, y: 37 },
                prize: Location {
                    x: 10000000007870,
                    y: 10000000006450
                }
            },
            ClawMach {
                a: Button { x: 69, y: 23 },
                b: Button { x: 27, y: 71 },
                prize: Location {
                    x: 10000000018641,
                    y: 10000000010279
                }
            },
        ];
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        let do_test = |example: &str, expected: &[ClawMach]| {
            let a = construct(&read_lines_in_memory(EXAMPLE_INPUT_STR_PART_1).collect::<Vec<_>>());
            match a {
                Result::Ok(actual) => {
                    assert_eq!(actual.len(), 4);
                    let expected: &[ClawMach] = &EXAMPLE_EXPECTED_PART_1;
                    assert_eq!(&actual, expected);
                }
                Result::Err(error) => {
                    assert!(false, "Expecting ok parse but found error: {error:?}")
                }
            }
        };
        do_test(EXAMPLE_INPUT_STR_PART_1, &EXAMPLE_EXPECTED_PART_1);
        do_test(EXAMPLE_INPUT_STR_PART_2, &EXAMPLE_EXPECTED_PART_2);
    }

    #[test]
    fn construction_single() {
        let actual = construct(&[
            "Button A: X+94, Y+34".to_string(),
            "Button B: X+22, Y+67".to_string(),
            "Prize: X=8400, Y=5400".to_string(),
        ]);
        match actual {
            Result::Ok(actuals) => {
                assert_eq!(actuals.len(), 1);
                assert_eq!(
                    actuals[0],
                    ClawMach {
                        a: Button { x: 94, y: 34 },
                        b: Button { x: 22, y: 67 },
                        prize: Location { x: 8400, y: 5400 }
                    }
                );
            }
            Result::Err(error) => assert!(false, "Expecting ok parse, but found error: {error:?}"),
        };
    }

    #[test]
    fn construction_error() {
        let actual = construct(&[
            "Button A: X+94, Y+34".to_string(),
            "Button B: X+22, Y+67".to_string(),
        ]);
        assert!(
            actual.is_err(),
            "Expecting an error, instead it is: {actual:?}"
        );
    }

    ///////////////////////////////////////////////

    fn solve_example_1<F>(f: F)
    where
        F: Fn(&ClawMach) -> Option<Press>,
    {
        let claw = &EXAMPLE_EXPECTED_PART_1[0];
        match f(claw) {
            Some(actual) => {
                assert_eq!(actual.a, 80);
                assert_eq!(actual.b, 40);
                assert_eq!(actual.cost(), 280);
            }
            None => assert!(
                false,
                "{claw:?} should have a solution with a min cost of 280"
            ),
        }
    }

    fn solve_example_3<F>(f: F)
    where
        F: Fn(&ClawMach) -> Option<Press>,
    {
        let claw = &EXAMPLE_EXPECTED_PART_1[2];
        match f(claw) {
            Some(actual) => {
                assert_eq!(actual.a, 38);
                assert_eq!(actual.b, 86);
                assert_eq!(actual.cost(), 200);
            }
            None => assert!(
                false,
                "{claw:?} should have a solution with a min cost of 200"
            ),
        }
    }

    fn solve_example_no_solution<F>(claw: &ClawMach, f: F)
    where
        F: Fn(&ClawMach) -> Option<Press>,
    {
        match f(claw) {
            Some(actual) => assert!(
                false,
                "{claw:?} should not have solution! found: {actual:?}"
            ),
            None => (),
        }
    }

    #[test]
    fn solve_example_1_part1() {
        solve_example_1(solve_brute_force);
    }

    #[test]
    fn solve_example_2_part1() {
        solve_example_no_solution(&EXAMPLE_EXPECTED_PART_1[1], solve_brute_force);
    }

    #[test]
    fn solve_example_3_part1() {
        solve_example_3(solve_brute_force);
    }

    #[test]
    fn solve_example_4_part1() {
        solve_example_no_solution(&EXAMPLE_EXPECTED_PART_1[3], solve_brute_force);
    }

    #[test]
    fn solve_example_part1() {
        let claws: &[ClawMach] = &EXAMPLE_EXPECTED_PART_1;
        let expected = 480;
        let actual = calculate_solution(claws.iter().map(solve_brute_force));
        assert_eq!(actual, expected);
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1(), 30973);
    }

    ///////////////////////////////////////////////

    #[test]
    fn solve_example_1_ilp() {
        solve_example_1(solve_dynamic_programming);
    }

    #[test]
    fn solve_example_2_ilp() {
        solve_example_no_solution(&EXAMPLE_EXPECTED_PART_1[1], solve_dynamic_programming);
    }

    #[test]
    fn solve_example_3_ilp() {
        solve_example_3(solve_dynamic_programming);
    }

    #[test]
    fn solve_example_4_ilp() {
        solve_example_no_solution(&EXAMPLE_EXPECTED_PART_1[3], solve_dynamic_programming);
    }

    #[test]
    fn solve_example_part1_ilp() {
        let claws: &[ClawMach] = &EXAMPLE_EXPECTED_PART_1;
        let expected = 480;
        let actual = calculate_solution(claws.iter().map(solve_dynamic_programming));
        assert_eq!(actual, expected);
    }

    // #[test]
    // fn solve_example_part2_ilp() {
    //     let claws: &[ClawMach] = &EXAMPLE_EXPECTED_PART_2;
    //     println!("solving #1...");
    //     assert!(solve_dynamic_programming(&claws[0]).is_none());
    //     println!("solving #2...");
    //     assert!(solve_dynamic_programming(&claws[1]).is_some());
    //     println!("solving #3...");
    //     assert!(solve_dynamic_programming(&claws[2]).is_none());
    //     println!("solving #4...");
    //     assert!(solve_dynamic_programming(&claws[3]).is_some());
    //     println!("solved!");
    // }

    #[test]
    fn pt2_soln_example() {
        todo!();
    }

    ///////////////////////////////////////////////
}
