use std::fmt::format;

use crate::{io_help, utils::collect_results};

///////////////////////////////////////////////////////////////////////////////////////////////////

type Register = u32;

#[allow(non_snake_case)]
#[derive(Clone, PartialEq, Eq)]
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

impl Executable {
    pub fn new(computer: &Computer, program: &Program) -> Executable {
        Executable {
            pc: 0,
            computer: computer.clone(),
            program: program.clone(),
        }
    }

    pub fn increment(&mut self) {
        self.pc += 2;
    }

    pub fn jump(&mut self, instruction_pointer: usize) {
        self.pc = instruction_pointer;
    }

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
        while self.is_ready() {
            let (pc, maybe_output) = run_step(&mut self.computer, &self.program[self.pc]);
            match pc {
                Some(instruction_pointer) => self.jump(instruction_pointer),
                None => self.increment(),
            };
            match maybe_output {
                Some(o) => output.push(o),
                None => (),
            };
        }
        output
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
            let val = computer.B ^ *literal as u32;
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
        Instruction::bxc => {
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

fn div(computer: &Computer, combo: Operand) -> u32 {
    computer.A / 2_u32.pow(combo_operand_value(computer, combo))
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
    let mut exe = Executable::new(&computer, &program);
    let output = exe.execute();
    Ok(output.join(","))
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
