
//
// Copyright (C) 2015 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// Test vertex_to_cell_map for 3D problem with hanging nodes and periodic
// boundary.

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <vector>

#include "../tests.h"

void
test()
{
  Triangulation<3>          tria;
  Point<3>                  a(0., 0., 0.);
  Point<3>                  b(1., 1., 1.);
  std::vector<unsigned int> repetitions(3);
  repetitions[0] = 2;
  repetitions[1] = 2;
  repetitions[2] = 1;
  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, a, b);

  // std::vector<GridTools::PeriodicFacePair<typename
  // parallel::distributed::Triangulation<dim>::cell_iterator> >
  // periodicity_vector;

  //     GridTools::collect_periodic_faces
  //     ( coarse_grid, 0, 1,
  //       0, periodicity_vector);

  //   coarse_grid.add_periodicity (periodicity_vector);

  Triangulation<3>::active_cell_iterator cell = tria.begin_active();
  cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  std::vector<std::set<Triangulation<3>::active_cell_iterator>> vertex_to_cell =
    GridTools::vertex_to_cell_map(tria);

  AssertThrow(tria.n_vertices() == vertex_to_cell.size(),
              ExcMessage("Wrong number of vertices"));

  for (unsigned int i = 0; i < vertex_to_cell.size(); ++i)
    deallog << "Number of cells for vertex " << std::to_string(i) << ": "
            << vertex_to_cell[i].size() << std::endl;
}

int
main(int argc, char *argv[])
{
  initlog();
  deallog << std::setprecision(4);

  test();

  deallog << "OK" << std::endl;

  return 0;
}
