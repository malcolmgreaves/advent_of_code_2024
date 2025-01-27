
///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> u64 {
    todo!()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> u64 {
    todo!()
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use indoc::indoc;
    use io_help::read_lines_in_memory;
    use lazy_static::lazy_static;

    use super::*;

    const EXAMPLE_INPUT_STR: &str = indoc! {"
        
    "};

    lazy_static! {
        static ref EXAMPLE_INPUT: ??? = vec![
            
        ];
    }

    #[test]
    fn construct() {
        let lines = read_lines_in_memory(EXAMPLE_INPUT_STR).collect::<Vec<_>>();
        let created = ???;
        let example: &??? = &EXAMPLE_INPUT;
        assert_eq!(created, *example);
    }

    #[test]
    fn pt1_soln_example() {
        panic!()
    }

    #[test]
    fn pt2_soln_example() {
        panic!()
    }
}
