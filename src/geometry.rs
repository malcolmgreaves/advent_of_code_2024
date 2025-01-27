use crate::utils::{Coordinate, Matrix};

pub struct Slope {
    // NOTE: Origin (0,0) is ** TOP-LEFT**.
    // NOTE: A **POSITIVE** rise means **GOING DOWN**. A **NEGATIVE** rise means **GOING UP**.
    // NOTE: A **POSITIVE** run means **GOING RIGHT**. A **NEGATIVE** run means **GOING LEFT**.
    rise: i64,
    run: i64,
}

impl Slope {
    fn rise_over_run(&self) -> f64 {
        // cast: we will loose precision
        (self.rise as f64) / (self.run as f64)
    }
}

pub enum AntinodeLocation {
    OutOfBounds,
    OneLocation(Coordinate),
    TwoLocations(Coordinate, Coordinate),
}

pub fn anitnode_location(
    max_rows: usize,
    max_cols: usize,
    src: &Coordinate,
    dst: &Coordinate,
) -> AntinodeLocation {
    // 2x distance from antenna on line
    // so take each (src) and (dst) and find 2x distance on line

    let coord_at = {
        let dist = distance(src, dst);
        move |start: &Coordinate, direction: &Slope| {
            coordinate_at(max_rows, max_cols, start, direction, dist)
        }
    };

    let antinode_for_src = coord_at(src, &line_slope(src, dst));

    let antinode_for_dst = coord_at(dst, &line_slope(dst, src));

    match (antinode_for_src, antinode_for_dst) {
        (Some(s), Some(d)) => AntinodeLocation::TwoLocations(s, d),
        (Some(s), None) => AntinodeLocation::OneLocation(s),
        (None, Some(d)) => AntinodeLocation::OneLocation(d),
        (None, None) => AntinodeLocation::OutOfBounds,
    }
}

pub fn distance(a: &Coordinate, b: &Coordinate) -> f64 {
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

fn coordinate_at(
    max_rows: usize,
    max_cols: usize,
    start: &Coordinate,
    direction: &Slope,
    distance: f64,
) -> Option<Coordinate> {
    // https://math.stackexchange.com/a/175906
    // Let 𝐯=(𝑥1,𝑦1)−(𝑥0,𝑦0). Normalize this to 𝐮=𝐯||𝐯||.
    // The point along your line at a distance 𝑑 from (𝑥0,𝑦0) is then (𝑥0,𝑦0)+𝑑𝐮.
    // If you want it in the direction of (𝑥1,𝑦1), or (𝑥0,𝑦0)−𝑑𝐮, if you want it in the opposite direction.

    todo!()
}
