use crate::utils::{Coordinate, Matrix};

pub struct Slope {
    // NOTE: Origin (0,0) is ** TOP-LEFT**.
    // NOTE: A **POSITIVE** rise means **GOING DOWN**. A **NEGATIVE** rise means **GOING UP**.
    // NOTE: A **POSITIVE** run means **GOING RIGHT**. A **NEGATIVE** run means **GOING LEFT**.
    rise: i64,
    run: i64,
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
        let distance_times_2 = dist * 2.0;
        move |start: &Coordinate, direction: &Slope| {
            coordinate_at(max_rows, max_cols, start, direction, distance_times_2)
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
        let row = new_x.clamp(0.0, (max_rows - 1) as f64).round() as usize;
        let col = new_y.clamp(0.0, (max_cols - 1) as f64).round() as usize;
        assert!(row < max_rows, "new_x: {new_x} | new_y: {new_y}");
        assert!(col < max_cols, "new_x: {new_x} | new_y: {new_y}");
        Some(Coordinate { row, col })
    } else {
        // out of bounds!
        None
    }
}
