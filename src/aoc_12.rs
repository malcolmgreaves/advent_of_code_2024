use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
};

use crate::{
    io_help, log,
    matrix::{
        Coordinate, Coords, Direction, GridMovement, Matrix, cardinal_neighbors,
        exterior_perimiter, trace_perimiter,
    },
    utils::group_by,
};

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt1() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/12").collect::<Vec<String>>();
    let garden = construct(&lines);
    let regions = determine_regions(&garden);
    Ok(cost(&regions))
}

impl HasValue<char> for char {
    fn value(&self) -> char {
        *self
    }
}

type Garden = Matrix<char>;

fn construct(lines: &[String]) -> Garden {
    assert_ne!(lines.len(), 0);
    lines.iter().map(|l| l.chars().collect()).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
struct Region<T: Eq + Clone + Ord + Debug> {
    letter: T,
    area: u64,
    perimiter: u64,
    members: Vec<Coordinate>,
}

impl<T: Eq + Clone + Ord + Debug> Region<T> {
    fn price(&self) -> u64 {
        self.area * self.perimiter
    }

    fn new<A>(mat: &Matrix<A>, letter: T, members: Vec<Coordinate>) -> Self {
        let area = members.len() as u64;
        let perimiter = exterior_perimiter(mat, &members);
        Self {
            letter,
            area,
            perimiter,
            members,
        }
    }
}

#[allow(dead_code)]
fn compare_region_num_pairs(
    (a_reg, a): &(Region<char>, u64),
    (b_reg, b): &(Region<char>, u64),
) -> Ordering {
    compare_char_num_pairs(&(a_reg.letter, *a), &(b_reg.letter, *b))
}

#[allow(dead_code)]
fn compare_char_num_pairs((a_char, a): &(char, u64), (b_char, b): &(char, u64)) -> Ordering {
    match a_char.cmp(b_char) {
        Ordering::Equal => a.cmp(b),
        other => other,
    }
}

impl Display for Region<char> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Region {{")?;
        write!(
            f,
            "letter: '{}', area: {}, perimiter: {}, members: {}",
            self.letter,
            self.area,
            self.perimiter,
            Coords(&self.members)
        )?;
        write!(f, "}}")
    }
}

impl<T: Eq + Clone + Ord + Debug> Ord for Region<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.letter.cmp(&other.letter) {
            Ordering::Equal => self.members.cmp(&other.members),
            a => a,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum State<T: Eq + Clone + Ord + Debug> {
    Building(T),
    Finished(T),
}

impl<T: Eq + Clone + Ord + Debug> HasValue<T> for State<T> {
    fn value(&self) -> T {
        match self {
            Self::Building(c) | Self::Finished(c) => c.clone(),
        }
    }
}

fn determine_regions<T: Eq + Clone + Ord + Debug>(garden: &Matrix<T>) -> Vec<Region<T>> {
    assert_ne!(garden.len(), 0);

    let mut region_builder: Matrix<State<T>> = garden
        .iter()
        .map(|r| r.iter().map(|c| State::Building(c.clone())).collect())
        .collect();

    let mut regions = Vec::new();

    let mut unfinished_business = true;
    while unfinished_business {
        unfinished_business = false;
        for row in 0..garden.len() {
            for col in 0..garden[0].len() {
                let val = region_builder[row][col].value();
                match expanded_neighborhood(&region_builder, row, col) {
                    FloodFill::New(available) => {
                        unfinished_business = true;
                        for Coordinate { row: r, col: c } in available.iter() {
                            region_builder[*r][*c] = State::Finished(val.clone());
                        }
                        let region = Region::new(&garden, val, available);
                        regions.push(region);
                    }
                    FloodFill::Solo => {
                        region_builder[row][col] = State::Finished(val.clone());
                        let region = Region::new(&garden, val, vec![Coordinate { row, col }]);
                        regions.push(region)
                    }
                    FloodFill::Prefilled => {
                        // we have already included this position in a previously obtained FloodFill::New(..) result
                        // we kept track of this by marking this visisted position with a State::Finished
                        // when we call expanded_neighborhood(..) on a Finished, we get this Prefilled
                        continue;
                    }
                }
            }
        }
    }

    regions
}

trait HasValue<T: Eq> {
    fn value(&self) -> T;
}

/// Neighbors: up, below, left, and right of (row, col) while still being in-bounds.
fn immediate_neighbors<A: Eq, T: Eq + Clone + Ord + Debug + HasValue<A>>(
    region_builder: &Matrix<T>,
    row: usize,
    col: usize,
) -> Vec<Coordinate> {
    let center_character = region_builder[row][col].value();

    cardinal_neighbors(region_builder, row, col)
        .iter()
        .flat_map(|x| match x {
            (Some(new_row), Some(new_col)) => {
                // this character check ensures we're only considering compatible positions
                if region_builder[*new_row][*new_col].value() == center_character {
                    Some(Coordinate::new(*new_row, *new_col))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<Vec<_>>()
}

enum FloodFill {
    Solo,
    Prefilled,
    New(Vec<Coordinate>),
}

/// The entire contigious neighborhood from (row,col) that are all in-bounds and have the same character.
fn expanded_neighborhood<T: Eq + Clone + Ord + Debug>(
    region_builder: &Matrix<State<T>>,
    row: usize,
    col: usize,
) -> FloodFill {
    if let State::Finished(_) = region_builder[row][col] {
        return FloodFill::Prefilled;
    }

    let mut neighborhood = HashSet::<Coordinate>::new();
    neighborhood.extend(immediate_neighbors(region_builder, row, col));
    if neighborhood.len() == 0 {
        return FloodFill::Solo;
    }

    let mut inserted_once = true;
    while inserted_once {
        inserted_once = false;
        let current_neighbors = neighborhood.iter().map(|x| x.clone()).collect::<Vec<_>>();
        current_neighbors.into_iter().for_each(|n| {
            for neigh in immediate_neighbors(region_builder, n.row, n.col) {
                if !neighborhood.contains(&neigh) {
                    inserted_once = true;
                    neighborhood.insert(neigh);
                }
            }
        });
    }

    FloodFill::New(neighborhood.into_iter().collect())
}

fn cost<T: Eq + Clone + Ord + Debug>(regions: &[Region<T>]) -> u64 {
    regions.iter().fold(0, |s, r| s + r.price())
}

///////////////////////////////////////////////////////////////////////////////////////////////////

pub fn solution_pt2() -> Result<u64, String> {
    let lines = io_help::read_lines("./inputs/12").collect::<Vec<String>>();
    let garden = construct(&lines);
    Ok(cost_sides(&garden))
}

fn cost_sides(garden: &Garden) -> u64 {
    count_sides(garden).iter().fold(0, |s, (region, sides)| {
        let cost_region = sides * region.area;
        s + cost_region
    })
}

fn count_sides<T: Eq + Clone + Ord + Debug>(garden: &Matrix<T>) -> Vec<(Region<T>, u64)> {
    // scan
    //      top
    //      bottom
    //      left
    //      right
    determine_regions(garden)
        .into_iter()
        .map(|r| {
            let s = scan_count(garden, &r);
            log!("{s}");
            (r, s)
        })
        .collect()
}

fn scan_count<T: Eq + Clone + Ord + Debug>(garden: &Matrix<T>, r: &Region<T>) -> u64 {
    log!("region: {:?} - {}", r.letter, Coords(&r.members));
    let top = scan_sides(garden, r, Direction::Up);
    let bottom = scan_sides(garden, r, Direction::Down);
    let left = scan_sides(garden, r, Direction::Left);
    let right = scan_sides(garden, r, Direction::Right);
    top + bottom + left + right
}

fn scan_sides<T: Eq + Clone + Ord + Debug>(
    garden: &Matrix<T>,
    region: &Region<T>,
    d: Direction,
) -> u64 {
    log!("direction: {d:?}");
    let exterior = determine_exterior(region);

    // FOR EACH row
    //  for each square
    //      from SQUARE in DIRECTION -> is there another square one step away in region?
    //              if NO or OUT OF BOUNDS, then mark that side as a face in the FACES matrix
    //                  increment counter
    //
    // for each NON-ZERO element in FACES:
    //      for each neighbor (cardinal) that is also non-zero,
    //          set this square & that other square as the min number
    //
    // find max element in faces => # of sides in DIRECTION

    let MinMax {
        min: min_row,
        max: max_row,
    } = exterior.row_limit();
    let MinMax {
        min: min_col,
        max: max_col,
    } = exterior.col_limit();
    // log!("min_row: {min_row} max_row: {max_row} | min_col: {min_col} max_col: {max_col}");
    let mut marker = exterior.submatrix(0 as u32);
    let mut counter = 1 as u32;

    // log!("marker:\n{marker:?}");

    let g = GridMovement::new(garden);

    // mark each edge facing direction
    for i in min_row..=max_row {
        // log!("i: {i}");
        for j in min_col..=max_col {
            let point = Coordinate { row: i, col: j };
            if !exterior.inside(&point) {
                continue;
            }

            let is_edge = match g.next_advance(&point, &d) {
                Some(next) => {
                    if garden[next.row][next.col] == region.letter {
                        // same region => interior!
                        false
                    } else {
                        // other region => edge!
                        true
                    }
                }
                None => true,
            };

            if is_edge {
                let i_marker = i - min_row;
                let j_marker = j - min_col;
                marker[i_marker][j_marker] = counter;
                // log!("\tis edge!: {i},{j}");
                counter += 1;
            } else {
                // log!("\tnot an edge!: {i},{j}");
            }
        }
    }

    // coalese edges
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..marker.len() {
            for j in 0..marker[0].len() {
                if marker[i][j] == 0 {
                    continue;
                }
                cardinal_neighbors(&marker, i, j)
                    .into_iter()
                    .for_each(|(aa, bb)| {
                        match (aa, bb) {
                            (Some(a), Some(b)) => {
                                let neighbor_number = marker[a][b];
                                if neighbor_number != 0 {
                                    // nonzero -> (i,j) and one of its neighbors are both edges
                                    // make them the same edge ID by uniquely selecting 1
                                    // we use MIN to make this as we can incrementally grow it
                                    if neighbor_number != marker[i][j] {
                                        changed = true;
                                    }
                                    let new = std::cmp::min(neighbor_number, marker[i][j]);
                                    marker[i][j] = new;
                                    marker[a][b] = new;
                                    // log!("\t\tcoalese: {i},{j} & {a},{b} are {new}");
                                } else {
                                    // 0 means not an edge
                                    // --> do nothing
                                }
                            }
                            _ => {
                                // one part of the neigbhor coordinate is OOB
                                // other is not an edge --> do nothing
                            }
                        }
                    });
            }
        }
    }

    // now the max # is the highest edge
    let sides_as_regions = determine_regions(&marker)
        .into_iter()
        .filter(|r| r.letter != 0)
        .collect::<Vec<_>>();
    log!(
        "\tfound {} sides (as regions):\n\t\t{:?}",
        sides_as_regions.len(),
        sides_as_regions
    );
    sides_as_regions.len() as u64
}

#[allow(dead_code)]
fn count_sides_custom<T: Eq + Clone + Ord + Debug + Default + Display>(
    garden: &Matrix<T>,
) -> Vec<(Region<T>, u64)> {
    let regions = {
        let mut r = determine_regions(garden);
        r.sort();
        r
    };

    let expanded_garden: Matrix<T> = expand_to_intersection_points_grid(garden);
    let expanded_regions: Vec<Region<T>> = {
        let mut r = determine_regions(&expanded_garden);
        r.sort();
        r
    };

    assert_eq!(regions.len(), expanded_regions.len());
    // operate on expanded grid and region

    regions
        .into_iter()
        .zip(expanded_regions)
        .map(|(og_region, ex_region)| {
            assert_eq!(og_region.letter, ex_region.letter);

            log!("REGION: '{}'", og_region.letter);
            let sides_exterior = count_sides_permiter(&expanded_garden, &ex_region.members);
            log!(
                "\tfound {sides_exterior} exterior sides for region '{}'",
                og_region.letter
            );

            let sides_interior = {
                let interior_psuedo_regions = find_holes(&expanded_garden, &ex_region);
                log!("\tfound {} holes", interior_psuedo_regions.len());

                interior_psuedo_regions
                    .into_iter()
                    .enumerate()
                    .fold(0, |s, (i, int_reg)| {
                        let id_interior = i + 1;
                        log!("\tINTERIOR REGION #{}:", id_interior);
                        let int_s = count_sides_permiter(&expanded_garden, &int_reg.members);
                        log!("\tfound {int_s} sides of interior region # {}", id_interior);
                        s + int_s
                    })
            };
            log!("\tfound {sides_interior} total interior sides");

            let sides = sides_exterior + sides_interior;
            log!("Total of {sides} for region '{}'", og_region.letter);

            (og_region, sides)
        })
        .collect()
}

fn expand_to_intersection_points_grid<T: Default + Clone>(mat: &Matrix<T>) -> Matrix<T> {
    let mut expanded = vec![vec![Default::default(); mat[0].len() * 2]; mat.len() * 2];
    // 2x size
    // for each i,j turn into 8 NEW points: all neighbors + 1 original point
    mat.iter().enumerate().for_each(|(i, row)| {
        let i_expanded = i * 2;
        row.iter().enumerate().for_each(|(j, _x)| {
            // i,j in mat
            //  ==>
            // i*2, j*2 in expanded
            let j_expanded = j * 2;
            expanded[i_expanded][j_expanded] = mat[i][j].clone();
            expanded[i_expanded][j_expanded + 1] = mat[i][j].clone();
            expanded[i_expanded + 1][j_expanded] = mat[i][j].clone();
            expanded[i_expanded + 1][j_expanded + 1] = mat[i][j].clone();
        });
    });

    expanded
}

// Only works on **EXPANDED** matrices! Must use `expand_to_intersection_points_grid` first!
fn count_sides_permiter<T>(expanded_mat: &Matrix<T>, members: &[Coordinate]) -> u64 {
    let perimeter = {
        let mut p = trace_perimiter(expanded_mat, members);

        p.sort_by(|a, b| match a.row.cmp(&b.row) {
            Ordering::Equal => a.col.cmp(&b.col),
            x => x,
        });
        log!("\tPERMITER: {}", Coords(&p));
        p
    };

    let is_perimiter = {
        let membership = perimeter.iter().collect::<HashSet<&Coordinate>>();
        move |c: &Coordinate| -> bool { membership.contains(c) }
    };
    let g = GridMovement::new(expanded_mat);

    let mut current = perimeter.first().unwrap().clone();
    let start = current.clone();
    let mut direction = Direction::Right;
    let mut visisted = HashSet::new();
    // visisted.insert((current.clone(), direction.clone()));
    visisted.insert(current.clone());
    // let mut last = current;
    // let mut previous_dir = Direction::Right;
    // let mut sides = 1;

    let mut sides = 1;

    log!("\t\t[face] start: {current} facing {direction:?} (+{sides})");

    let mut changed = true;
    while changed {
        changed = false;
        for (next, facing) in g.clockwise_rotation(&current, &direction.opposite().clockwise()) {
            // let pair = (next.clone(), facing.clone());
            if !visisted.contains(&next) {
                // if !visisted.contains(&pair) {
                if is_perimiter(&next) {
                    if facing == direction {
                        log!("\t\t[face] continuing along existing face ({facing:?} -> {next})");
                    } else {
                        // let n = direction.calculate_clockwise_turns(&facing) as u64;
                        let n = 1;
                        log!(
                            "\t\t[turn] changing direction from {direction:?} to {facing:?} to reach {next} (+{n} -> {})",
                            sides + n
                        );
                        sides += n;
                    }
                    visisted.insert(next.clone());
                    // visisted.insert(pair);
                    current = next;
                    direction = facing;
                    changed = true;

                    // if current == start {
                    //     // rotate until it is in the original orietnation
                    //     let n = direction.calculate_clockwise_turns(&Direction::Right) as u64;
                    //     log!("\t\t[stop] reached starting point! It's done! (+{n})");
                    //     new_sides += n;
                    //     changed = false;
                    // }
                    break;
                } else {
                    log!(
                        "\t\t\t[fail] next step in {facing:?} is out of perimiter of region ({next})"
                    );
                }
            } else {
                if next == start {
                    if facing != direction {
                        // let n = direction.calculate_clockwise_turns(&facing) as u64;
                        let n = 1;
                        // log!("\t\t[turn] changing direction from {direction:?} to {facing:?} to reach {next} (+{n} -> {})", new_sides+n);
                        sides += n;
                        log!(
                            "\t\t[allow] have already visisted {next}, but it's original so make it once more! (+{n} -> {sides})"
                        );
                    } else {
                        log!(
                            "\t\t[allow] have already visisted {next}, but it's original so make it once more!"
                        );
                    }
                    current = next;
                    direction = facing;
                    changed = true;
                    break;
                } else {
                    log!("\t\t\t[fail] have already visisted {next}, {facing:?}");
                }
            }
        }
        if !changed {
            log!(
                "\t\t[STOP] could not find any viable place to step to from {current} facing {direction:?} --> END @ # sides: {sides}"
            );
        }
    }

    sides
}

fn find_holes<T: Eq + Clone + Ord + Debug>(
    expanded_garden: &Matrix<T>,
    region: &Region<T>,
) -> Vec<Region<bool>> {
    // calculate exterior =>
    //  for each row and col, find min and max
    //
    // for all points at or inside exterior
    //      collect points if they are not the same letter as {{region.letter}}
    //
    // for all collected points
    //      copy into new empty matrix
    //      relabel each as same letter
    //      relabel everything else as other letter
    //      run determine_regions on this copy
    //      discard region for other letter
    //      for each other region, return it with a new unique letter that doesn't
    //          match anything in expanded_garden

    let exterior = determine_exterior(region);

    let interior_points_not_in_region = expanded_garden
        .iter()
        .enumerate()
        .flat_map(|(i, row)| {
            row.iter()
                .enumerate()
                .flat_map(|(j, c)| {
                    let coord = Coordinate { row: i, col: j };
                    if *c != region.letter && exterior.inside(&coord) {
                        Some(coord)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let psuedo_label_interior = true;
    let psudeo_expanded_garden: Matrix<bool> = {
        let MinMax {
            min: min_row,
            max: max_row,
        } = exterior.row_limit();

        let row_range = max_row - min_row;

        let MinMax {
            min: min_col,
            max: max_col,
        } = exterior.col_limit();
        let col_range = max_col - min_col;

        let mut psudeo_expanded_garden = exterior.submatrix(false);

        interior_points_not_in_region.iter().for_each(|c| {
            let new_row = c.row - min_row;
            let new_col = c.col - min_col;
            if new_row >= row_range || new_col >= col_range {
                println!("ERROR: new: ({new_row}, {new_col}) is outside of bounds: ({row_range},{col_range}) using row min-max: {:?} and col min-max: {:?} -- original: {c}", exterior.row_limit(), exterior.col_limit());
            }
            psudeo_expanded_garden[new_row][new_col] = psuedo_label_interior;
        });
        psudeo_expanded_garden
    };

    let interior_regions = determine_regions(&psudeo_expanded_garden)
        .into_iter()
        .filter(|r| r.letter == psuedo_label_interior)
        .collect::<Vec<_>>();

    interior_regions
}

#[derive(Debug, PartialEq, Eq)]
struct MinMax<N>
where
    N: Ord + Sub<Output = N> + Add<Output = N> + Mul<Output = N> + Div<Output = N> + Clone,
{
    min: N,
    max: N,
}

impl<N> MinMax<N>
where
    N: Ord + Sub<Output = N> + Add<Output = N> + Mul<Output = N> + Div<Output = N> + Clone,
{
    fn inside(&self, x: N) -> bool {
        self.min <= x && x <= self.max
    }
}

struct Exterior {
    rows: HashMap<usize, MinMax<usize>>,
    cols: HashMap<usize, MinMax<usize>>,
}

impl Exterior {
    fn inside(&self, point: &Coordinate) -> bool {
        match (self.rows.get(&point.row), self.cols.get(&point.col)) {
            (Some(row_min_max), Some(col_min_max)) => {
                row_min_max.inside(point.col) && col_min_max.inside(point.row)
            }
            _ => false,
        }
    }

    fn row_limit(&self) -> MinMax<usize> {
        _min_max_of(self.cols.iter().map(|(_, r)| r).collect::<Vec<_>>())
    }

    fn col_limit(&self) -> MinMax<usize> {
        _min_max_of(self.rows.iter().map(|(_, c)| c).collect::<Vec<_>>())
    }

    fn submatrix<T: Clone>(&self, zero: T) -> Matrix<T> {
        let MinMax {
            min: min_col,
            max: max_col,
        } = self.col_limit();
        let col_range = (max_col + 1) - min_col;

        let MinMax {
            min: min_row,
            max: max_row,
        } = self.row_limit();
        let row_range = (max_row + 1) - min_row;

        vec![vec![zero; col_range]; row_range]
    }
}

fn _min_max_of<N>(min_maxes: Vec<&MinMax<N>>) -> MinMax<N>
where
    N: Ord + Sub<Output = N> + Add<Output = N> + Mul<Output = N> + Div<Output = N> + Clone,
{
    let min = min_maxes
        .iter()
        .map(|MinMax { min, max: _ }| min)
        .min()
        .unwrap();
    let max = min_maxes
        .iter()
        .map(|MinMax { min: _, max }| max)
        .max()
        .unwrap();
    MinMax {
        min: min.clone(),
        max: max.clone(),
    }
}

fn determine_exterior<T: Eq + Clone + Ord + Debug>(region: &Region<T>) -> Exterior {
    Exterior {
        rows: _determine_exterior(&region.members, true),
        cols: _determine_exterior(&region.members, false),
    }
}

fn _determine_exterior(members: &[Coordinate], rows: bool) -> HashMap<usize, MinMax<usize>> {
    // https://www.reddit.com/r/rust/comments/1cxv6kt/function_pointer_tuple_from_match_expression/
    let (key, other) = if rows {
        (
            row_select as fn(&Coordinate) -> usize,
            col_select as fn(&Coordinate) -> usize,
        )
    } else {
        (
            col_select as fn(&Coordinate) -> usize,
            row_select as fn(&Coordinate) -> usize,
        )
    };

    group_by(
        |(index_row_or_col, _)| *index_row_or_col,
        members.iter().map(|c| (key(c), c)).collect(),
    )
    .into_iter()
    .map(|(i_row, elements)| {
        let other_vals = elements.iter().map(|(_, c)| other(c)).collect::<Vec<_>>();
        (
            i_row,
            MinMax {
                min: *other_vals.iter().min().unwrap(),
                max: *other_vals.iter().max().unwrap(),
            },
        )
    })
    .collect()
}

fn row_select(c: &Coordinate) -> usize {
    c.row
}

fn col_select(c: &Coordinate) -> usize {
    c.col
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {

    use indoc::indoc;
    use lazy_static::lazy_static;

    use crate::{io_help::read_lines_in_memory, matrix::print_matrix};

    use super::*;

    const EXAMPLE_INPUT_STR_SM: &str = indoc! {"
        AAAA
        BBCD
        BBCC
        EEEC
    "};

    const EXAMPLE_INPUT_STR_2P: &str = indoc! {"
        OOOOO
        OXOXO
        OOOOO
        OXOXO
        OOOOO
    "};

    const EXAMPLE_INPUT_STR_LG: &str = indoc! {"
        RRRRIICCFF
        RRRRIICCCF
        VVRRRCCFFF
        VVRCCCJFFF
        VVVVCJJCFE
        VVIVCCJJEE
        VVIIICJJEE
        MIIIIIJJEE
        MIIISIJEEE
        MMMISSJEEE
    "};

    #[allow(dead_code)]
    const EXAMPLE_INPUT_STR_WEIRD: &str = indoc! {"
        AAIBBBB
        AAIIIBB
        AIIIIIB
        AIIIBIB
        AAAIBBB
    "};

    lazy_static! {
        static ref EXAMPLE_SM: Garden = vec![
            vec!['A', 'A', 'A', 'A'],
            vec!['B', 'B', 'C', 'D'],
            vec!['B', 'B', 'C', 'C'],
            vec!['E', 'E', 'E', 'C'],
        ];
        static ref EXAMPLE_2P: Garden = vec![
            vec!['O', 'O', 'O', 'O', 'O'],
            vec!['O', 'X', 'O', 'X', 'O'],
            vec!['O', 'O', 'O', 'O', 'O'],
            vec!['O', 'X', 'O', 'X', 'O'],
            vec!['O', 'O', 'O', 'O', 'O'],
        ];
        static ref EXAMPLE_LG: Garden = vec![
            vec!['R', 'R', 'R', 'R', 'I', 'I', 'C', 'C', 'F', 'F'],
            vec!['R', 'R', 'R', 'R', 'I', 'I', 'C', 'C', 'C', 'F'],
            vec!['V', 'V', 'R', 'R', 'R', 'C', 'C', 'F', 'F', 'F'],
            vec!['V', 'V', 'R', 'C', 'C', 'C', 'J', 'F', 'F', 'F'],
            vec!['V', 'V', 'V', 'V', 'C', 'J', 'J', 'C', 'F', 'E'],
            vec!['V', 'V', 'I', 'V', 'C', 'C', 'J', 'J', 'E', 'E'],
            vec!['V', 'V', 'I', 'I', 'I', 'C', 'J', 'J', 'E', 'E'],
            vec!['M', 'I', 'I', 'I', 'I', 'I', 'J', 'J', 'E', 'E'],
            vec!['M', 'I', 'I', 'I', 'S', 'I', 'J', 'E', 'E', 'E'],
            vec!['M', 'M', 'M', 'I', 'S', 'S', 'J', 'E', 'E', 'E'],
        ];
        static ref EXAMPLE_WEIRD: Garden = vec![
            vec!['A', 'A', 'I', 'B', 'B', 'B', 'B'],
            vec!['A', 'A', 'I', 'I', 'I', 'B', 'B'],
            vec!['A', 'I', 'I', 'I', 'I', 'I', 'B'],
            vec!['A', 'I', 'I', 'I', 'B', 'I', 'B'],
            vec!['A', 'A', 'A', 'I', 'B', 'B', 'B'],
        ];
    }

    fn into_lines(contents: &str) -> Vec<String> {
        read_lines_in_memory(contents).collect::<Vec<_>>()
    }

    #[test]
    fn construction() {
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_2P));
        let expected: &Garden = &EXAMPLE_2P;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_SM));
        let expected: &Garden = &EXAMPLE_SM;
        assert_eq!(actual, *expected);
        //
        let actual = construct(&into_lines(EXAMPLE_INPUT_STR_LG));
        let expected: &Garden = &EXAMPLE_LG;
        assert_eq!(actual, *expected);
    }

    #[test]
    fn regions() {
        regions_test(&EXAMPLE_2P, vec![('X', 1, 4), ('O', 21, 36)]);
        regions_test(
            &EXAMPLE_SM,
            vec![
                ('A', 4, 10),
                ('B', 4, 8),
                ('C', 4, 10),
                ('D', 1, 4),
                ('E', 3, 8),
            ],
        );
    }

    fn regions_test(garden: &Garden, expected_region_info: Vec<(char, u64, u64)>) {
        for r in determine_regions(garden) {
            log!(
                "REGION: '{}': area={} perimiter={} # members: {}",
                r.letter,
                r.area,
                r.perimiter,
                r.members.len()
            );
            let cap = (r.letter, r.area, r.perimiter);
            assert!(
                expected_region_info.iter().any(|expected| cap == *expected),
                "expecting {cap:?} to be one of {}: {expected_region_info:?}",
                expected_region_info.len(),
            )
        }
    }

    #[test]
    fn price() {
        price_test(
            &EXAMPLE_2P,
            vec![('X', 4), ('X', 4), ('X', 4), ('X', 4), ('O', 756)],
        );
        price_test(
            &EXAMPLE_SM,
            vec![('A', 40), ('B', 32), ('C', 40), ('D', 4), ('E', 24)],
        );
        price_test(
            &EXAMPLE_LG,
            vec![
                ('R', 216),
                ('I', 32),
                ('C', 392),
                ('F', 180),
                ('V', 260),
                ('J', 220),
                ('C', 4),
                ('E', 234),
                ('I', 308),
                ('M', 60),
                ('S', 24),
            ],
        );
    }

    fn price_test(garden: &Garden, expected_prices: Vec<(char, u64)>) {
        let expected_prices = {
            let mut e = expected_prices.clone();
            e.sort_by(compare_char_num_pairs);
            e
        };
        let expected_cost = expected_prices.iter().fold(0, |s, (_, p)| s + *p);

        let regions = determine_regions(garden);
        let actual_region_prices = {
            let mut r: Vec<(char, u64)> = determine_regions(garden)
                .iter()
                .map(|r| (r.letter, r.price()))
                .collect();
            r.sort_by(compare_char_num_pairs);
            r
        };
        let actual_cost = cost(&regions);

        assert_eq!(actual_region_prices, expected_prices);
        assert_eq!(actual_cost, expected_cost);
    }

    #[test]
    fn sides_simple() {
        _cost_sides_test(&vec![vec!['A']], None, None, None, Some(vec![('A', 4)]));
        _cost_sides_test(
            &vec![vec!['A', 'A']],
            None,
            None,
            None,
            Some(vec![('A', 4)]),
        );
        _cost_sides_test(
            &vec![vec!['A'], vec!['A']],
            None,
            None,
            None,
            Some(vec![('A', 4)]),
        );
        _cost_sides_test(
            &vec![vec!['A', 'A'], vec!['A', 'A']],
            None,
            None,
            None,
            Some(vec![('A', 4)]),
        );
    }

    #[test]
    fn debug_sides_0() {
        let garden: &Garden = &vec![vec!['A', 'A', 'A'], vec!['A', 'A', 'B']];
        // ------------------------------------
        // vec!['A', 'A', 'A'],
        // vec!['A', 'A', 'B'],
        // -----------------------------------
        //          0    1    2    3    4    5
        // 0: vec!['A', 'A', 'A', 'A', 'A', 'A'],
        // 1: vec!['A', 'A', 'A', 'A', 'A', 'A'],
        // 2: vec!['A', 'A', 'A', 'A', 'B', 'B'],
        // 3: vec!['A', 'A', 'A', 'A', 'B', 'B'],
        // -----------------------------------
        // let example: &Garden = &EXAMPLE_WEIRD;
        let expected = [('A', 6), ('B', 4)];
        _debug_sides(garden, &expected);
    }

    #[test]
    fn debug_sides_1() {
        let garden: &EXAMPLE_SM = &EXAMPLE_SM;
        /*
                    0    1    2    3    4    5    6    7
           0: vec!['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
           1: vec!['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
           2: vec!['B', 'B', 'B', 'B', 'C', 'C', 'D', 'D'],
           3: vec!['B', 'B', 'B', 'B', 'C', 'C', 'D', 'D'],
           4: vec!['B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
           5: vec!['B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
           6: vec!['E', 'E', 'E', 'E', 'E', 'E', 'C', 'C'],
           7: vec!['E', 'E', 'E', 'E', 'E', 'E', 'C', 'C'],
        */
        let expected = [('A', 4), ('B', 4), ('C', 8), ('D', 4), ('E', 4)];
        _debug_sides(garden, &expected);
    }

    #[test]
    fn debug_sides_2() {
        let garden = &EXAMPLE_2P;
        /*
                    0    1    2    3    4    5    6    7    8    9
           0: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
           1: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
           2: vec!['O', 'O', 'X', 'X', 'O', 'O', 'X', 'X', 'O', 'O'],
           3: vec!['O', 'O', 'X', 'X', 'O', 'O', 'X', 'X', 'O', 'O'],
           4: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
           5: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
           6: vec!['O', 'O', 'X', 'X', 'O', 'O', 'X', 'X', 'O', 'O'],
           7: vec!['O', 'O', 'X', 'X', 'O', 'O', 'X', 'X', 'O', 'O'],
           8: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
           9: vec!['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        */
        let expected = [('O', 20), ('X', 4), ('X', 4), ('X', 4), ('X', 4)];
        _debug_sides(garden, &expected);
    }

    fn _debug_sides(garden: &Garden, expected: &[(char, u64)]) {
        let region_sides = count_sides(garden);

        let rcs = {
            let mut rcs = region_sides
                .into_iter()
                .map(|(region, sides)| {
                    log!(
                        "region: '{}' has {} members --> {sides} sides",
                        region.letter,
                        region.members.len()
                    );
                    (region.letter, sides)
                })
                .collect::<Vec<_>>();
            rcs.sort_by(compare_char_num_pairs);
            rcs
        };

        let expected = {
            let mut e = expected.to_vec();
            e.sort_by(compare_char_num_pairs);
            e
        };

        assert_eq!(rcs, expected);
    }

    #[test]
    fn cost_sides_test_small() {
        _cost_sides_test(
            &EXAMPLE_SM,
            Some(80),
            Some(vec![('A', 16), ('B', 16), ('C', 32), ('D', 4), ('E', 12)]),
            None,
            Some(vec![('A', 4), ('B', 4), ('C', 8), ('D', 4), ('E', 4)]),
        );
    }

    #[test]
    fn cost_sides_test_large() {
        _cost_sides_test(
            &EXAMPLE_LG,
            Some(1206),
            Some(vec![
                ('R', 120),
                ('I', 16),
                ('C', 308),
                ('F', 120),
                ('V', 130),
                ('J', 132),
                ('C', 4),
                ('E', 104),
                ('I', 224),
                ('M', 30),
                ('S', 18),
            ]),
            None,
            None,
        );
        // _cost_sides_test(
        //     &EXAMPLE_WEIRD,
        //     0,
        //     Some(vec![
        //         ('A', 0),   // 8 sides
        //         ('B', 0),   // 12 sides
        //         ('I', 224), // 16 sides
        //     ]),
        //     None,
        // );
    }

    fn _cost_sides_test(
        garden: &Garden,
        expected_cost: Option<u64>,
        per_cost: Option<Vec<(char, u64)>>,
        expected_sides: Option<u64>,
        per_sides: Option<Vec<(char, u64)>>,
    ) {
        assert!(
            expected_cost.is_some()
                || expected_sides.is_some()
                || per_cost.is_some()
                || per_sides.is_some(),
            "must supply either expected cost, expected # sides, per-region cost or sides!\n\texpected cost: {:?}\n\texpected sides: {:?}\n\t per region expected cost: {:?}\n\tper region expected # sides: {:?}",
            expected_cost,
            expected_sides,
            per_cost,
            per_sides
        );

        let check = |actuals: &[(Region<char>, u64)], specific: &[(char, u64)], msg: &str| {
            let mut specific = specific.to_vec();
            specific.sort_by(compare_char_num_pairs);
            let mut actuals = actuals.to_vec();
            actuals.sort_by(compare_region_num_pairs);

            let diffs = specific
                .iter()
                .zip(actuals)
                .filter(|((s_letter, s_val), (a_region, a_val))| {
                    let test = a_region.letter != *s_letter || s_val != a_val;
                    if test {
                        log!("MISMATCH: A region of {} plants with {msg} {a_val} should have matched {s_letter} with {msg} {s_val}:\n{a_region}\n", a_region.letter);
                    }
                    test
                })
                .collect::<Vec<_>>();

            assert_eq!(
                diffs.len(),
                0,
                "There were {} specific differences:\n{:?}",
                diffs.len(),
                diffs
            );
        };

        let regions_sides = count_sides(garden);
        match per_sides {
            Some(specific) => check(&regions_sides, &specific, "sides"),
            None => (),
        }

        let regions_costs = regions_sides
            .iter()
            .map(|(r, s)| {
                let cost = r.area * s;
                (r.clone(), cost)
            })
            .collect::<Vec<_>>();

        match per_cost {
            Some(specific) => check(&regions_costs, &specific, "cost"),
            None => (),
        }

        match expected_cost {
            Some(e_cost) => assert_eq!(
                regions_costs.iter().fold(0, |s, (_, c)| s + c),
                e_cost,
                "actual cost != expected"
            ),
            None => (),
        }

        match expected_sides {
            Some(e_sides) => assert_eq!(
                regions_sides.iter().fold(0, |u, (_, s)| { u + s }),
                e_sides,
                "actual # sides != expected",
            ),
            None => (),
        }

        log!("-------------------------------------------------");
    }

    #[test]
    fn expansion() {
        [
            (
                vec![vec!['A', 'A'], vec!['A', 'B']],
                vec![
                    vec!['A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'B', 'B'],
                    vec!['A', 'A', 'B', 'B'],
                ],
            ),
            (
                vec![vec!['A', 'A', 'A'], vec!['A', 'A', 'B']],
                vec![
                    vec!['A', 'A', 'A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'A', 'A', 'B', 'B'],
                    vec!['A', 'A', 'A', 'A', 'B', 'B'],
                ],
            ),
            (
                vec![
                    vec!['A', 'A', 'A'],
                    vec!['A', 'A', 'B'],
                    vec!['A', 'C', 'B'],
                ],
                vec![
                    vec!['A', 'A', 'A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'A', 'A', 'A', 'A'],
                    vec!['A', 'A', 'A', 'A', 'B', 'B'],
                    vec!['A', 'A', 'A', 'A', 'B', 'B'],
                    vec!['A', 'A', 'C', 'C', 'B', 'B'],
                    vec!['A', 'A', 'C', 'C', 'B', 'B'],
                ],
            ),
        ]
        .into_iter()
        .for_each(|(original, expected)| {
            let actual = expand_to_intersection_points_grid(&original);
            log!("actual:");
            print_matrix(&actual);
            log!("expected:");
            print_matrix(&expected);
            assert_eq!(actual, expected);
        })
    }

    #[test]
    fn exterior_limits() {
        let garden: &Garden = &EXAMPLE_LG;
        for region in determine_regions(garden) {
            let exterior = determine_exterior(&region);

            let MinMax {
                min: min_row,
                max: max_row,
            } = exterior.row_limit();
            let MinMax {
                min: min_col,
                max: max_col,
            } = exterior.col_limit();

            let _rows = region.members.iter().map(|c| c.row).collect::<Vec<_>>();
            assert_eq!(*_rows.iter().min().unwrap(), min_row);
            assert_eq!(*_rows.iter().max().unwrap(), max_row);

            let _cols = region.members.iter().map(|c| c.col).collect::<Vec<_>>();
            assert_eq!(*_cols.iter().min().unwrap(), min_col);
            assert_eq!(*_cols.iter().max().unwrap(), max_col);
        }
    }

    #[test]
    fn pt1_soln_example() {
        assert_eq!(solution_pt1().unwrap(), 1473620);
    }

    #[test]
    fn pt2_soln_example() {
        // assert_eq!(solution_pt2().unwrap(), )
    }
}
