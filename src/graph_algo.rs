pub trait Search: Clone + PartialEq + Eq + PartialOrd + Ord {
    fn cost(&self) -> u64;
}

// impl Ord for Search {
//     // just use cost
// }

pub fn lowest_cost_path_dijkstras(
    memory: &Memory,
    start: &Coordinate,
    end: &Coordinate,
) -> Result<u64, String> {
    validate(memory, start, end)?;

    // create graph
    let g = GridMovement::new(memory);

    // create priority queue
    let mut priority_queue: BinaryHeap<_> = BinaryHeap::<Search>::new();
    priority_queue.push(Search {
        loc: start.clone(),
        cost: 0,
    });

    // create cost ("distance") map from start -> each vertex (empty space)
    let mut distance: HashMap<Coordinate, u64> = GridMovement::new(memory)
        .coordinates()
        .map(|c| {
            let cost_from_start = if &c == start { 0 } else { u64::MAX };
            (c, cost_from_start)
        })
        .collect();

    let mut prev_for_best_path: HashMap<Coordinate, Option<Coordinate>> = GridMovement::new(memory)
        .coordinates()
        .map(|c| (c, None))
        .collect();

    // while queue is not empty
    //      v = take lowest cost from queue
    //      if v is end: return cost(v)

    let mut hit_end_once = false;
    while let Some(Search { loc, cost }) = priority_queue.pop() {
        if &loc == end {
            hit_end_once = true;
            continue;
        }
        if cost > *distance.get(&loc).unwrap() {
            // lower-cose path to loc has already been found
            continue;
        }

        // graph.neighbors(node)
        for (neighbor, new_dir) in g.cardinal_neighbor_directions(&loc) {
            if memory[neighbor.row][neighbor.col] == true {
                // we can't go into an occupied memory space!
                continue;
            }

            let next_cost = 1;

            let considering_next = Search {
                loc: neighbor,
                cost: cost + next_cost,
            };

            let previous_min_cost = distance.get_mut(&considering_next.loc).unwrap();

            if considering_next.cost < *previous_min_cost {
                // the path we took to get here is lower than the minimum cost of
                // some other path we took to get here!
                let new_min_cost = considering_next.cost.clone();
                let neighbor = considering_next.loc.clone();
                priority_queue.push(considering_next);
                // update distance (cost) for the location
                *previous_min_cost = new_min_cost;
                // update: we got to <neighbor> via <loc>
                prev_for_best_path.insert(neighbor, Some(loc.clone()));
            }
        }
    }

    match distance
        .get(&end)
        .filter(|_| hit_end_once)
        .map(|c| c.clone())
    {
        Some(min_steps) => Ok(min_steps),
        None => Err(format!(
            "no valid path found from start={start} to end={end}"
        )),
    }
}
