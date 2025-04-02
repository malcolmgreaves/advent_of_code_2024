use std::fmt::format;

use crate::{io_help, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

type Register = u32;

#[allow(non_snake_case)]
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
#[derive(Clone, PartialEq, Eq)]
enum Instruction {
    adv(Operand),
    bxl(Operand),
    bst(Operand),
    jnz(Operand),
    bxc,
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
            4 => Ok(Self::bxc), // ignores operand: only there for "legacy" reasons :)
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
            Self::bxc => 4,
            Self::out(_) => 5,
            Self::bdv(_) => 6,
            Self::cdv(_) => 7,
        }
    }
}

struct Executable {
    pc: usize,
    computer: Computer,
    program: Program,
}

enum Step {
    Work,
    Output(String),
    Halt,
}

impl Executable {
    pub fn new(computer: &Computer, program: &Program) -> Executable {
        Executable {
            pc: 0,
            computer: computer.clone(),
            program: program.clone(),
        }
    }

    pub fn increment(mut self) {
        self.pc += 2;
    }

    pub fn jump(mut self, instruction_pointer: usize) {
        self.pc = instruction_pointer;
    }

    pub fn is_ready(&self) -> bool {
        self.pc < self.program.len()
    }

    /// None if program has halted.
    /// This occurs if the instruction pointer is past the end of the program.
    pub fn instruction(&self) -> Option<&Instruction> {
        if self.is_ready() {
            Some(&self.program[self.pc])
        } else {
            None
        }
    }

    pub fn run_step(mut self) -> Step {
        match self.instruction() {
            Some(instruction) => {
                panic!("Step::Output(_) or Step::Work")
            }
            None => Step::Halt,
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
                    bits.swap_remove(1).parse::<u32>().map_err(|e| format!("{e}"))
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
    let mut bits = line.split("Program: ").collect::<Vec<_>>();
    let (ops, errors) = collect_results(
        bits.swap_remove(1)
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

fn execute(c: Computer, p: Program) -> Vec<Output> {
    let mut exe = Executable::new(&c, &p);
    let mut output = Vec::new();
    while let Some(instruction) = exe.instruction() {}
    output
}

fn run_step(computer: &mut Computer, instruction: Instruction) -> Option<String> {
    match instruction {
        Instruction::adv(combo) => {
            let demoniator = 2_u32.pow(combo_operand_value(computer, combo));
            let result = computer.A / demoniator;
            computer.A = result;
        }
        Instruction::bxl(literal) => {
            // ^ is bitwise XOR: https://doc.rust-lang.org/std/ops/trait.BitXor.html
            let val = computer.B ^ literal as u32;
            computer.B = val;
        }
        Instruction::bst(combo) => todo!(),
        Instruction::jnz(literal) => todo!(),
        Instruction::bxc => todo!(),
        Instruction::out(combo) => {
            let val = combo_operand_value(computer, combo) % 8;
            return Some(format!("{val}"));
        }
        Instruction::bdv(combo) => todo!(),
        Instruction::cdv(combo) => todo!(),
    }
    None
}

fn combo_operand_value(computer: &Computer, combo: Operand) -> u32 {
    match combo {
        0..=3 => combo as u32,
        4 => computer.A,
        5 => computer.B,
        6 => computer.C,
        7 => panic!("Combo operand 7 is reserved and will not appear in valid programs."),
        _ => panic!("unrecognized Operand value: {combo}"),
    }
}

pub fn solution_pt1() -> Result<String, String> {
    let lines = io_help::read_lines("./inputs/???");
    let (computer, program) = construct(lines)?;
    Err(format!("part 1 is unimplemented!"))
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<String, String> {
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
