pub trait Appendable {
    type Item;
    fn append(self, element: Self::Item) -> Self;
}

impl<T> Appendable for Vec<T> {
    type Item = T;

    fn append(mut self, element: Self::Item) -> Self {
        self.push(element);
        self
    }
}

// pub trait ListOps {
//     type Item;
//     fn len(&self) -> usize;
//     fn to_vec(self) -> Vec<Self::Item>;
//     // fn into_iter(self) -> impl Iterator<Item=Self::Item>;
// }

// pub trait GrowableList: ListOps {
//     fn append(self, element: Self::Item) -> Self;
// }

// #[derive(Debug, PartialEq, Eq, Clone)]
// pub enum List<T> {
//     Empty,
//     Cons(T, Box<List<T>>),
// }

// impl<T> ListOps for List<T> {
//     type Item = T;

//     fn len(&self) -> usize {
//         match self {
//             List::Empty => 0,
//             List::Cons(_, remaining) => 1 + remaining.len(),
//         }
//     }

//     fn to_vec(self) -> Vec<Self::Item> {
//         let mut v = Vec::with_capacity(self.len());
//         unsafe {
//             v.set_len(self.len());
//         }
//         let mut i = self.len();
//         let mut ptr = self;
//         loop {
//             match ptr {
//                 List::Empty => break,
//                 List::Cons(element, remaining) => {
//                     i -= 1;
//                     v.as_mut_slice()[i] = element;
//                     ptr = *remaining;
//                 }
//             }
//         }
//         v
//     }

//     // fn into_iter(self) -> impl Iterator<Item=Self::Item> {
//     //     ListIter{list: self}
//     // }
// }

// // struct ListIter<T> {
// //     list: Box<List<T>>
// // }

// // impl<T> IntoIterator for List<T> {
// //     type Item=T;

// //     type IntoIter = ListIter<T>;

// //     fn into_iter(self) -> Self::IntoIter {
// //         ListIter{list:Box::new(self)}
// //     }
// // }

// // impl<T> Iterator for ListIter<T> {

// //     type Item=T;

// //     fn next(&mut self) -> Option<Self::Item> {
// //         // https://stackoverflow.com/a/63354217/362021
// //         let (e, r) = match self.list { //match &(*(self.list)) {
// //             List::Empty => return None,
// //             List::Cons(element, remaining ) => {
// //                 (*element, **remaining)
// //             },
// //         };
// //         self.list = Box::new(r);
// //         Some(e)
// //     }
// // }
// // impl<T> ListIter<T> {

// //     fn next_(mut self) -> Option<T> {
// //         // https://stackoverflow.com/a/63354217/362021
// //         let (e, new_list): (T, List<T>) = match self.list {
// //             List::Cons(element, remaining) => {
// //                 // let r: &List<T> = remaining;
// //                 let r = *remaining;
// //                 (element, r)
// //             },
// //             // List::Empty => (None, &List::Empty),
// //             List::Empty => return None,
// //         };

// //         self.list = new_list;

// //         Some(e)
// //     }
// // }

// impl<T> GrowableList for List<T> {
//     fn append(self, element: Self::Item) -> Self {
//         List::Cons(element, Box::new(self))
//     }
// }
