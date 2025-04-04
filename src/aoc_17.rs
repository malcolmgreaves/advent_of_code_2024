use std::{cmp::Ordering, collections::VecDeque};

use crate::{io_help, search::binary_search_on_answer, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

type Register = u64;

#[allow(non_snake_case)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Computer {
    A: Register,
    B: Register,
    C: Register,
}

type Opcode = u8;
type Operand = u8;

type RawProgram = Vec<(Opcode, Operand)>;
type Program = Vec<Instruction>;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Instruction {
    adv(Operand),
    bxl(Operand),
    bst(Operand),
    jnz(Operand),
    bxc(Operand),
    out(Operand),
    bdv(Operand),
    cdv(Operand),
}

impl Instruction {
    fn new(opcode: Opcode, operand: Operand) -> Result<Instruction, String> {
        match opcode {
            0 => Ok(Self::adv(operand)),
            1 => Ok(Self::bxl(operand)),
            2 => Ok(Self::bst(operand)),
            3 => Ok(Self::jnz(operand)),
            4 => Ok(Self::bxc(operand)),
            5 => Ok(Self::out(operand)),
            6 => Ok(Self::bdv(operand)),
            7 => Ok(Self::cdv(operand)),
            _ => Err(format!("unrecognized opcode: {opcode} -- must be in [0,7]")),
        }
    }

    fn opcode(&self) -> Opcode {
        match self {
            Self::adv(_) => 0,
            Self::bxl(_) => 1,
            Self::bst(_) => 2,
            Self::jnz(_) => 3,
            Self::bxc(_) => 4,
            Self::out(_) => 5,
            Self::bdv(_) => 6,
            Self::cdv(_) => 7,
        }
    }

    fn operand(&self) -> Operand {
        match self {
            Self::adv(operand) => *operand,
            Self::bxl(operand) => *operand,
            Self::bst(operand) => *operand,
            Self::jnz(operand) => *operand,
            Self::bxc(operand) => *operand,
            Self::out(operand) => *operand,
            Self::bdv(operand) => *operand,
            Self::cdv(operand) => *operand,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Executable {
    pc: usize,
    computer: Computer,
    program: Program,
}

impl Executable {
    pub fn new(computer: &Computer, program: &Program) -> Executable {
        Executable {
            pc: 0,
            computer: computer.clone(),
            program: program.clone(),
        }
    }

    pub fn increment(&mut self) {
        // since program is compiled into instructions, incrementing the
        // instruction pointer by 1 means we're jumping over 2 in the raw format
        self.pc += 1;
    }

    pub fn jump(&mut self, instruction_pointer: usize) {
        self.pc = instruction_pointer;
    }

    #[allow(dead_code)]
    pub fn current_instruction(&self) -> Option<&Instruction> {
        if self.is_ready() {
            Some(&self.program[self.pc])
        } else {
            None
        }
    }

    pub fn is_ready(&self) -> bool {
        self.pc < self.program.len()
    }

    fn execute(&mut self) -> Vec<String> {
        let mut output = Vec::new();
        // println!("program: {:?}", self.program);
        while self.is_ready() {
            // println!("\tpc: {} {:?}", self.pc, self.program[self.pc]);
            let (pc, maybe_output) = run_step(&mut self.computer, &self.program[self.pc]);
            match pc {
                Some(instruction_pointer) => {
                    // println!("\t\tjumping to: {instruction_pointer}");
                    self.jump(instruction_pointer)
                }
                None => self.increment(),
            };
            match maybe_output {
                Some(o) => {
                    // println!("\t\toutputting: {o}");
                    output.push(o);
                }
                None => (),
            };
        }
        output
    }

    fn execute_compare_output(&mut self, comparision_output: &[String]) -> bool {
        let mut comp_i = 0;
        while self.is_ready() {
            let (pc, maybe_output) = run_step(&mut self.computer, &self.program[self.pc]);
        }
    }
}

fn compile(program: RawProgram) -> Result<Program, String> {
    let (instructions, errors) = collect_results(
        program
            .into_iter()
            .map(|(opcode, operand)| Instruction::new(opcode, operand)),
    );
    if errors.len() > 0 {
        Err(format!(
            "Failed to parse {} (opcode,operand)s into instructions:\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ))
    } else {
        Ok(instructions)
    }
}

fn construct(mut lines: impl Iterator<Item = String>) -> Result<(Computer, Program), String> {
    let register_lines = [
        match lines.next() {
            Some(l) => l,
            None => {
                return Err(format!(
                    "expecting first line to be Register A but iterator is empty"
                ));
            }
        },
        match lines.next() {
            Some(l) => l,
            None => {
                return Err(format!(
                    "expecting second line to be Register B but iterator is empty"
                ));
            }
        },
        match lines.next() {
            Some(l) => l,
            None => {
                return Err(format!(
                    "expecting third line to be Register C but iterator is empty"
                ));
            }
        },
    ];
    let computer = parse_computer(&register_lines)?;

    // ingore the next line -> is a blank line
    _ = lines.next();

    let raw = match lines.next() {
        Some(program_line) => parse_raw_program(program_line),
        None => Err(format!(
            "expecting last line to be program line, but iterator is empty"
        )),
    }?;
    let program = compile(raw)?;

    Ok((computer, program))
}

fn parse_computer(lines: &[String]) -> Result<Computer, String> {
    if lines.len() != 3 {
        return Err(format!(
            "expecting three (3) register lines but found: {}",
            lines.len()
        ));
    }
    let (mut registers, errors) = collect_results(
        lines
            .iter()
            .map(|line| {
                let mut bits = line.split(": ").collect::<Vec<_>>();
                if bits.len() != 2 {
                    Err(format!(
                        "Expecting each register to list its label and value with ':'. Failed to parse: {line}"
                    ))
                } else {
                    bits.swap_remove(1).parse::<Register>().map_err(|e| format!("{e}"))
                }
            })
    );
    #[allow(non_snake_case)]
    if errors.len() > 0 {
        Err(format!(
            "found {} errors when parsing register lines:\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ))
    } else {
        assert!(registers.len() == 3);
        let C = registers.remove(2);
        let B = registers.remove(1);
        let A = registers.remove(0);
        Ok(Computer { A, B, C })
    }
}

fn parse_raw_program(line: String) -> Result<RawProgram, String> {
    let raw_program = {
        let mut bits = line.split("Program: ").collect::<Vec<_>>();
        match bits.len() {
            1 => bits.swap_remove(0),
            2 => bits.swap_remove(1),
            _ => {
                return Err(format!(
                    "expecting raw program to be 'Program: <raw>' or just '<raw>' -- invalid format: '{line}'"
                ));
            }
        }
    };
    let (ops, errors) = collect_results(
        raw_program
            .split(",")
            .map(|x| x.parse::<u8>().map_err(|e| format!("{e}"))),
    );
    if errors.len() > 0 {
        Err(format!(
            "encountered {} errors when parsing raw program's opcode & operand pairs:\n\t{}",
            errors.len(),
            errors.join("\n\t")
        ))
    } else {
        if ops.len() % 2 != 0 {
            Err(format!(
                "every opcode must have one operand -- found an odd number of ops: {}",
                ops.len()
            ))
        } else {
            let raw = ops.chunks_exact(2);
            assert_eq!(raw.remainder().len(), 0);
            Ok(raw
                .into_iter()
                .map(|chunk_2| {
                    assert_eq!(chunk_2.len(), 2);
                    (chunk_2[0], chunk_2[1])
                })
                .collect())
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Updates computer's registers with the result of the instruction.
// Returns the possibly new program counter and possible output.
//
// If the optional PC is None, then instruction advancement should continue as normal.
fn run_step(computer: &mut Computer, instruction: &Instruction) -> (Option<usize>, Option<String>) {
    match instruction {
        Instruction::adv(combo) => {
            computer.A = div(&computer, *combo);
        }
        Instruction::bxl(literal) => {
            // ^ is bitwise XOR: https://doc.rust-lang.org/std/ops/trait.BitXor.html
            let val = computer.B ^ *literal as Register;
            computer.B = val;
        }
        Instruction::bst(combo) => {
            let val = combo_operand_value(computer, *combo) % 8;
            computer.B = val;
        }
        Instruction::jnz(literal) => {
            // do nothing if register A is zero
            if computer.A != 0 {
                return (Some(*literal as usize), None);
            }
        }
        Instruction::bxc(_) => {
            // ignores operand: only there for "legacy" reasons :)
            let val = computer.B ^ computer.C;
            computer.B = val;
        }
        Instruction::out(combo) => {
            let val = combo_operand_value(computer, *combo) % 8;
            return (None, Some(format!("{val}")));
        }
        Instruction::bdv(combo) => {
            computer.B = div(&computer, *combo);
        }
        Instruction::cdv(combo) => {
            computer.C = div(&computer, *combo);
        }
    }

    (None, None)
}

fn div(computer: &Computer, combo: Operand) -> Register {
    computer.A / (2 as Register).pow(combo_operand_value(computer, combo).try_into().unwrap())
}

fn combo_operand_value(computer: &Computer, combo: Operand) -> Register {
    match combo {
        0..=3 => combo as Register,
        4 => computer.A,
        5 => computer.B,
        6 => computer.C,
        7 => panic!("Combo operand 7 is reserved and will not appear in valid programs."),
        _ => panic!("unrecognized Operand value: {combo}"),
    }
}

pub fn solution_pt1() -> Result<String, String> {
    let lines = io_help::read_lines("./inputs/17");
    let (computer, program) = construct(lines)?;
    let mut exe = Executable::new(&computer, &program);
    let output = exe.execute();
    Ok(output.join(","))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/17");
    let (computer, program) = construct(lines)?;

    println!("program: {:?}", program);

    match minimum_register_a_for_quine(&computer, &program, Algo::Special) {
        Some(register_a) => Ok(register_a as Register),
        None => Err(format!(
            "could not find a value for register A that makes this program a quine: '{}'",
            stringify_program(&program),
        )),
    }
}

#[allow(dead_code)]
fn raw_program(program: &Program) -> RawProgram {
    program
        .iter()
        .map(|instr| (instr.opcode(), instr.operand()))
        .collect::<Vec<_>>()
}

fn stringify_program(program: &Program) -> String {
    program
        .iter()
        .map(|instr| format!("{},{}", instr.opcode(), instr.operand()))
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Algo {
    BruteForce,
    BinaryBrute,
    BinarySearchLen,
    Special,
}

fn minimum_register_a_for_quine(
    computer: &Computer,
    program: &Program,
    choice: Algo,
) -> Option<Register> {
    let program_str = stringify_program(program);

    let run_output_for_a = |register_a: Register| -> String {
        let mut exe = {
            let mut c = computer.clone();
            c.A = register_a;
            Executable::new(&c, program)
        };
        let output = exe.execute().join(",");
        // println!("[{register_a}] {output} =?= {program_str}");
        // println!("[{register_a}] len(output)={} =?= len(program)={}", output.len(), program_str.len());
        // println!("[{register_a}] {output} ({}) =?= ({}) {program_str}", output.len(), program_str.len());
        output
    };

    let is_found = |register_a: Register| -> bool { run_output_for_a(register_a) == program_str };

    match choice {
        Algo::BruteForce => {
            for a in Register::MIN..=Register::MAX {
                if is_found(a) {
                    return Some(a);
                }
            }
        }
        Algo::BinaryBrute => {
            let mut queue = VecDeque::new();
            queue.push_back((Register::MIN, Register::MAX));
            while let Some((low, high)) = queue.pop_front() {
                if high < low + 1 {
                    return None;
                }

                if (high - low) < u16::MAX as Register {
                    // switch to iterative: it is very fast to go through 2^16 values
                    for i in low..=high {
                        if is_found(i) {
                            return Some(i);
                        }
                    }
                    // didn't work -> discard this whole range
                    continue;
                }

                let midpoint = low + ((high - low) / 2);
                if is_found(midpoint) {
                    return Some(midpoint);
                }
                queue.push_back((low, midpoint));
                queue.push_back((midpoint, high));
            }
        }
        Algo::BinarySearchLen => {
            let ans = binary_search_on_answer(
                Register::MIN,
                Register::MAX,
                |register_a: Register| -> bool {
                    run_output_for_a(register_a).len() == program_str.len()
                },
            );
            if is_found(ans) {
                return Some(ans);
            }
        }
        Algo::Special => {
            let (low, high) = binary_search_range_on_answer(|register_a: Register| -> Ordering {
                // run_output_for_a(register_a).len().cmp(&program_str.len())
                program_str.len().cmp(&run_output_for_a(register_a).len())
            });

            for a in low..=high {
                if is_found(a) {
                    return Some(a);
                }
            }
        }
    }
    return None;
}

pub fn binary_search_range_on_answer(
    program_to_output_len: impl Fn(Register) -> Ordering,
) -> (Register, Register) {
    let mut low = Register::MIN;
    let mut high = Register::MAX;

    let mut high_range = Register::MIN;
    while low + 1 < high {
        let midpoint = low + (high.checked_sub(low).unwrap() / 2);

        match program_to_output_len(midpoint) {
            Ordering::Equal => {
                high_range = midpoint;
                // keep going -> what's the TOP of this equal range?
                low = midpoint;
            }
            Ordering::Greater => {
                low = midpoint;
            }
            Ordering::Less => {
                high = midpoint;
            }
        }
    }

    low = Register::MIN;
    high = high_range;
    let mut low_range = Register::MAX;
    while low + 1 < high {
        let midpoint = low + (high.checked_sub(low).unwrap() / 2);
        match program_to_output_len(midpoint) {
            Ordering::Equal => {
                low_range = midpoint;
                // keep going -> what's the BOTTOM of this equal range?
                high = midpoint;
            }
            Ordering::Greater => {
                low = midpoint;
            }
            Ordering::Less => {
                high = midpoint;
            }
        }
    }

    println!("[STOP] range of equal-len programs is: [{low_range}, {high_range}]");
    (low_range, high_range)
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
        Register A: 729
        Register B: 0
        Register C: 0

        Program: 0,1,5,4,3,0
    "};

    lazy_static! {
        static ref EXAMPLE_EXPECTED_COMPUTER: Computer = Computer { A: 729, B: 0, C: 0 };
        static ref EXAMPLE_EXPECTED_PROGRAM: Program = vec![
            Instruction::adv(1),
            Instruction::out(4),
            Instruction::jnz(0),
        ];
    }

    ///////////////////////////////////////////////

    #[test]
    fn construction() {
        let expected_computer: &Computer = &EXAMPLE_EXPECTED_COMPUTER;
        let expected_program: &Program = &EXAMPLE_EXPECTED_PROGRAM;
        let (computer, program) = construct(read_lines_in_memory(EXAMPLE_INPUT_STR)).unwrap();
        assert_eq!(computer, *expected_computer);
        assert_eq!(program, *expected_program);
    }

    ///////////////////////////////////////////////

    /// If register C contains 9, the program 2,6 would set register B to 1.
    #[test]
    fn test_run_step_1() {
        let mut computer = Computer { A: 0, B: 0, C: 9 };
        let program = compile(parse_raw_program("2,6".to_string()).unwrap()).unwrap();
        let _ = run_step(&mut computer, &program[0]);
        assert_eq!(computer.B, 1);
    }

    /// If register A contains 10, the program 5,0,5,1,5,4 would output 0,1,2.
    #[test]
    fn test_run_step_2() {
        let mut exe = Executable {
            pc: 0,
            computer: Computer { A: 10, B: 0, C: 0 },
            program: compile(parse_raw_program("5,0,5,1,5,4".to_string()).unwrap()).unwrap(),
        };
        let output = exe.execute();
        assert_eq!(output.join(","), "0,1,2");
    }

    /// If register A contains 2024, the program 0,1,5,4,3,0 would output 4,2,5,6,7,7,7,7,3,1,0 and leave 0 in register A.
    #[test]
    fn test_run_step_3() {
        let mut exe = Executable {
            pc: 0,
            computer: Computer {
                A: 2024,
                B: 0,
                C: 0,
            },
            program: compile(parse_raw_program("0,1,5,4,3,0".to_string()).unwrap()).unwrap(),
        };
        let output = exe.execute();
        assert_eq!(output.join(","), "4,2,5,6,7,7,7,7,3,1,0");
        assert_eq!(exe.computer.A, 0);
    }

    /// If register B contains 29, the program 1,7 would set register B to 26.
    #[test]
    fn test_run_step_4() {
        let mut computer = Computer { A: 0, B: 29, C: 0 };
        let program = compile(parse_raw_program("1,7".to_string()).unwrap()).unwrap();
        let _ = run_step(&mut computer, &program[0]);
        assert_eq!(computer.B, 26);
    }

    /// If register B contains 2024 and register C contains 43690, the program 4,0 would set register B to 44354.
    #[test]
    fn test_run_step_5() {
        let mut computer = Computer {
            A: 0,
            B: 2024,
            C: 43690,
        };
        let program = compile(parse_raw_program("4,0".to_string()).unwrap()).unwrap();
        let _ = run_step(&mut computer, &program[0]);
        assert_eq!(computer.B, 44354);
    }

    #[test]
    fn execute_example() {
        let mut exe = Executable::new(&EXAMPLE_EXPECTED_COMPUTER, &EXAMPLE_EXPECTED_PROGRAM);
        let output = exe.execute();
        assert_eq!(output.join(","), "4,6,3,5,6,3,5,2,1,0");
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), "2,0,7,3,0,3,1,3,7");
    }

    ///////////////////////////////////////////////

    #[test]
    fn quine_example() {
        let raw = "0,3,5,4,3,0";
        let mut exe = Executable {
            pc: 0,
            computer: Computer {
                A: 117440,
                B: 0,
                C: 0,
            },
            program: compile(parse_raw_program(raw.to_string()).unwrap()).unwrap(),
        };
        let output = exe.execute().join(",");
        assert_eq!(output, raw);
    }

    #[test]
    fn search_for_quine() {
        let actual = minimum_register_a_for_quine(
            &Computer { A: 0, B: 0, C: 0 },
            &compile(parse_raw_program("0,3,5,4,3,0".to_string()).unwrap()).unwrap(),
            Algo::Special,
        )
        .unwrap();
        assert_eq!(actual, 117440);
    }

    #[ignore]
    #[test]
    fn pt2_soln_example() {
        // TODO
        assert_eq!(solution_pt2().unwrap(), 0)
    }
}
