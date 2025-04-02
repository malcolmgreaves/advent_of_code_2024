use crate::io_help;

///////////////////////////////////////////////////////////////////////////////////////////////////

struct TODO {

}

fn construct(lines: impl Iterator<Item=String>) -> Result<TODO, String> {
    panic!()
}


///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/???");
    let _ = lines;
    Err(format!("part 1 is unimplemented!"))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/???");
    let _ = lines;
    Err(format!("part 2 is unimplemented!"))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::io_help::read_lines_in_memory;

    use super::*;

    ///////////////////////////////////////////////

    const EXAMPLE_INPUT_STR: &str = indoc! {"
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED: Option<u8> = None;
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        panic!();
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt1_soln_example() {
        panic!();
    }

    ///////////////////////////////////////////////

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        panic!();
    }
}
