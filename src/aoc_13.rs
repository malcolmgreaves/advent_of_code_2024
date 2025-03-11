use crate::{io_help, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    let lines = io_help::read_lines("./inputs/13").collect::<Vec<String>>();
    panic!();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    let lines = io_help::read_lines("./inputs/13").collect::<Vec<String>>();
    panic!();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Button {
    x: usize,
    y: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Location {
    x: usize,
    y: usize,
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

fn parse_num_from(symbol: &str, s: &str) -> Result<usize, String> {
    match s.split(symbol).last() {
        Some(x) => x.parse::<usize>().map_err(|e| e.to_string()),
        None => Result::Err(format!(
            "missing separator symbol '{symbol}' in input string: '{s}'"
        )),
    }
}

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::{io_help::read_lines_in_memory, utils::Res};

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
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
        static ref EXAMPLE_EXPECTED: Vec<ClawMach> = vec![
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

    #[test]
    fn construction() {
        let a = construct(&read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>());
        match a {
            Result::Ok(actual) => {
                assert_eq!(actual.len(), 4);
                let expected: &[ClawMach] = &EXAMPLE_EXPECTED;
                assert_eq!(&actual, expected);
            }
            Result::Err(error) => assert!(false, "Expecting ok parse but found error: {error:?}"),
        }
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

    #[test]
    fn pt1_soln_example() {
        panic!();
    }

    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
