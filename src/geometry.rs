use std::fmt::Display;

use crate::utils::Coordinate;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Slope {
    // NOTE: Origin (0,0) is ** TOP-LEFT**.
    // NOTE: A **POSITIVE** rise means **GOING DOWN**. A **NEGATIVE** rise means **GOING UP**.
    // NOTE: A **POSITIVE** run means **GOING RIGHT**. A **NEGATIVE** run means **GOING LEFT**.
    rise: i64,
    run: i64,
}

impl Display for Slope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.rise, self.run)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum AntinodeLoc {
    OutOfBounds,
    OneLocationSrc(Coordinate),
    OneLocationDst(Coordinate),
    TwoLocations { src: Coordinate, dst: Coordinate },
}

pub fn antinode_location(
    max_rows: usize,
    max_cols: usize,
    src: &Coordinate,
    dst: &Coordinate,
) -> AntinodeLoc {
    // 2x distance from antenna on line
    // so take each (src) and (dst) and find 2x distance on line

    let coord_at = {
        // let dist = distance_l1(src, dst);
        // let distance_times_2 = dist * 2;
        move |start: &Coordinate, direction: &Slope| {
            coordinate_at_l1(max_rows, max_cols, start, direction, 2)
        }
    };

    let antinode_for_src = coord_at(src, &line_slope(src, dst));

    let antinode_for_dst = coord_at(dst, &line_slope(dst, src));

    match (antinode_for_src, antinode_for_dst) {
        (Some(s), Some(d)) => AntinodeLoc::TwoLocations { src: s, dst: d },
        (Some(s), None) => AntinodeLoc::OneLocationSrc(s),
        (None, Some(d)) => AntinodeLoc::OneLocationDst(d),
        (None, None) => AntinodeLoc::OutOfBounds,
    }
}

#[allow(dead_code)]
pub fn distance_l1(a: &Coordinate, b: &Coordinate) -> u64 {
    // Calculate 2D distance between points: 𝑑=|𝑥1−𝑥0| + |𝑦1−𝑦0|
    let row_d = difference(a.row, b.row);
    let col_d = difference(a.col, b.col);
    row_d + col_d
}

#[allow(dead_code)]
pub fn distance_l2(a: &Coordinate, b: &Coordinate) -> f64 {
    // Calculate 2D distance between points: 𝑑=√[ (𝑥1−𝑥0)^2 + (𝑦1−𝑦0)^2 ]
    // (x1-x0)^2
    let row_d = difference(a.row, b.row).pow(2);
    // (y1-y0)^2
    let col_d = difference(a.col, b.col).pow(2);
    // (𝑥1−𝑥0)^2 + (𝑦1−𝑦0)^2
    let sum_d = (row_d + col_d) as f64;
    // 𝑑=√[ (𝑥1−𝑥0)^2 + (𝑦1−𝑦0)^2 ]
    sum_d.sqrt()
}

pub fn difference(a: usize, b: usize) -> u64 {
    // absolute value difference between two positive integers

    // lower -> higher OK
    let a = a as u64;
    let b = b as u64;
    if a > b {
        a - b
    } else if b > a {
        b - a
    } else {
        0
    }
}

pub fn line_slope(src: &Coordinate, dst: &Coordinate) -> Slope {
    let (src_row, src_col) = force_i64(src);
    let (dst_row, dst_col) = force_i64(dst);
    Slope {
        rise: dst_row - src_row,
        run: dst_col - src_col,
    }
}

fn force_i64(x: &Coordinate) -> (i64, i64) {
    let row = TryInto::<i64>::try_into(x.row).unwrap();
    let col = TryInto::<i64>::try_into(x.col).unwrap();
    (row, col)
}

fn coordinate_at_l1(
    max_rows: usize,
    max_cols: usize,
    start: &Coordinate,
    direction: &Slope,
    lengths_away: u16,
) -> Option<Coordinate> {
    // JUST MULTIPLY RISE AND RUN by lengths_away (2) !

    // (new_x, new_y) is FULL STEP on the line from START in DIRECTION that is LENGTHS_AWAY
    let (new_x, new_y) = (
        TryInto::<i64>::try_into(start.row).unwrap() + (direction.rise * lengths_away as i64),
        TryInto::<i64>::try_into(start.col).unwrap() + (direction.run * lengths_away as i64),
    );

    if new_x >= 0
        && new_x < TryInto::<i64>::try_into(max_rows).unwrap()
        && new_y >= 0
        && new_y < TryInto::<i64>::try_into(max_cols).unwrap()
    {
        // in-bounds!
        let row = TryInto::<usize>::try_into(new_x).unwrap();
        let col = TryInto::<usize>::try_into(new_y).unwrap();
        Some(Coordinate { row, col })
    } else {
        // out of bounds!
        None
    }
}

#[allow(dead_code)]
fn coordinate_at_l2(
    max_rows: usize,
    max_cols: usize,
    start: &Coordinate,
    direction: &Slope,
    distance: f64,
) -> Option<Coordinate> {
    // https://math.stackexchange.com/a/175906

    let (x0, y0) = (
        TryInto::<i64>::try_into(start.row).unwrap(),
        TryInto::<i64>::try_into(start.col).unwrap(),
    );
    // (x1, y1) is ONE STEP on the line from START in DIRECTION
    let (x1, y1) = (
        TryInto::<i64>::try_into(start.row).unwrap() + direction.rise,
        TryInto::<i64>::try_into(start.col).unwrap() + direction.run,
    );

    // Let 𝐯=(𝑥1,𝑦1)−(𝑥0,𝑦0). Normalize this to 𝐮=𝐯/||𝐯||.
    let (u_x, u_y) = {
        let norm_v = ((x1 - x0).abs() + (y1 - y0).abs()) as f64;
        (((x1 - x0) as f64) / norm_v, ((y1 - y0) as f64) / norm_v)
    };
    // The point along your line at a distance 𝑑 from (𝑥0,𝑦0) is then (𝑥0,𝑦0)+𝑑𝐮.
    // If you want it in the direction of (𝑥1,𝑦1), or (𝑥0,𝑦0)−𝑑𝐮, if you want it in the opposite direction.
    let new_x = (start.row as f64) + distance * u_x;
    let new_y = (start.col as f64) + distance * u_y;

    if new_x >= 0.0 && new_x < max_rows as f64 && new_y >= 0.0 && new_y < max_cols as f64 {
        match (custom_round(max_rows, new_x), custom_round(max_cols, new_y)) {
            (Some(row), Some(col)) => {
                // println!("\tA: ({},{}) [S: {direction}, D: {distance:0.2}] -> T: ({:0.2},{:0.2}) -> final ({row},{col})", start.row, start.col, new_x, new_y);
                Some(Coordinate { row, col })
            }
            _ => {
                // println!("\tA: ({},{}) [S: {direction}, D: {distance:0.2}] -> T: ({:0.2},{:0.2}) -> OOB (round, max: ({max_rows},{max_cols}))", start.row, start.col, new_x, new_y);
                // rounding pushed coordinate off of board!
                None
            }
        }
    } else {
        // out of bounds!
        // println!("\tA ({},{}) [S: {direction}, D: {distance:0.2}] -> T: ({:0.2},{:0.2}) -> OOB (max: ({max_rows},{max_cols}))", start.row, start.col, new_x, new_y);
        None
    }
}

#[allow(dead_code)]
fn custom_round(maximum: usize, x: f64) -> Option<usize> {
    if x < 0.0 {
        None
    } else if x >= maximum as f64 {
        None
    } else {
        let r = x.ceil() as usize;
        // let r = x.round() as usize;
        if r >= maximum {
            None
        } else {
            Some(r)
        }
    }
}

pub fn antinode_location_resonant_harmonics(
    max_rows: usize,
    max_cols: usize,
    src: &Coordinate,
    dst: &Coordinate,
) -> Vec<Coordinate> {
    // 2x distance from antenna on line
    // so take each (src) and (dst) and find 2x distance on line

    let coord_at = {
        // let dist = distance_l1(src, dst);
        // let distance_times_2 = dist * 2;
        move |start: &Coordinate, direction: &Slope| -> Vec<Coordinate> {
            let mut i = 1_u16;
            let mut coords = Vec::new();
            loop {
                let c = coordinate_at_l1(max_rows, max_cols, start, direction, i);
                i += 1;
                match c {
                    Some(coordinate) => coords.push(coordinate),
                    None => break,
                }
            }
            coords
        }
    };

    let mut all_antinodes = Vec::new();
    // get slope
    let slope_src_to_dst = line_slope(src, dst);
    // go from SRC to DST until exhausted
    all_antinodes.extend(
        coord_at(src, &slope_src_to_dst), //.into_iter().map(|a| AntinodeLoc::OneLocationSrc(a))
    );
    // invert slope and go from SRC until exhausted
    let slope_dst_to_src = line_slope(dst, src);
    all_antinodes.extend(
        coord_at(src, &slope_dst_to_src), //.into_iter().map(|a| AntinodeLoc::OneLocationSrc(a))
    );
    // don't forget that SRC is ALSO a resonant harmonic antinode! it will be one for DST
    all_antinodes.push(src.clone()); //AntinodeLoc::OneLocationDst(src.clone()));
    all_antinodes
}
