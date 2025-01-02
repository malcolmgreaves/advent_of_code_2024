use core::iter::Iterator;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn read_lines(filepath: &str) -> impl Iterator<Item = String> {
    let f = File::open(filepath).expect("Could not load file!");
    let reader = BufReader::new(f);
    reader
        .lines()
        .map(|l| l.expect("Could not read a line of text."))
}

pub fn read_lines_as_ints(separator: &str, filepath: &str) -> Vec<Vec<i32>> {
    read_lines(filepath)
        .map(|l| l.trim().split(separator).map(to_int).collect::<Vec<i32>>())
        .collect::<Vec<Vec<i32>>>()
}

pub fn to_int(s: &str) -> i32 {
    s.parse::<i32>()
        .expect(format!("Could not convert to i32! Original: '{s}'").as_str())
}
