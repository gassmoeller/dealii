// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2020 by the deal.II authors
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



// test the FEValues views and extractor classes. these tests use a primitive
// finite element and scalar extractors

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/vector.h>

#include "../tests.h"



template <int dim>
void
test(const Triangulation<dim> &tr, const FiniteElement<dim> &fe)
{
  TimerOutput timer(std::cout, TimerOutput::summary, TimerOutput::wall_times);
  MappingQ<dim> mapping(4);

  {
    DoFHandler<dim> dof(tr);
    dof.distribute_dofs(fe);

        const QGauss<dim> quadrature_temp(2);
        std::vector<Point<dim>> quadrature_points(quadrature_temp.size());
        std::vector<double> shape_values(quadrature_temp.size());

    TimerOutput::Scope(timer,"Regenerate FEValues");

    unsigned int i=0;
    for (const auto &cell : tr.active_cell_iterators())
      {
        const QGauss<dim> quadrature(2);
        FEValues<dim>     fe_values(
          mapping, fe, quadrature, update_values | update_gradients | update_hessians);
        fe_values.reinit(cell);

        if (i==3)
{
        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
        {
          quadrature_points[q] = fe_values.quadrature_point(q);
          shape_values[q] = fe_values.value(q,i);
        }
        ++i;
}
      }
  }

  {
    DoFHandler<dim> dof(tr);
    dof.distribute_dofs(fe);

    TimerOutput::Scope(timer,"Replace Quadrature");
        FEValues<dim>     fe_values(
          mapping, fe, QGauss<dim>(2), update_values | update_gradients | update_hessians);
    for (const auto &cell : tr.active_cell_iterators())
      {
        const QGauss<dim> quadrature(2);
        fe_values.replace_quadrature(quadrature);
        fe_values.reinit(cell);

                if (i==3)
{
        for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
        {
          quadrature_points[q] = fe_values.quadrature_point(q);
          shape_values[q] = fe_values.value(q,i);
        }
        ++i;
}
      }
  }
}



template <int dim>
void
test_hyper_sphere()
{
  Triangulation<dim> tr;
  GridGenerator::hyper_ball(tr);

  static const SphericalManifold<dim> boundary;
  tr.set_manifold(0, boundary);

  tr.refine_global(3);

  FESystem<dim> fe(FE_Q<dim>(1),
                   1,
                   FE_Q<dim>(2),
                   2,
                   FE_DGQArbitraryNodes<dim>(QIterated<1>(QTrapez<1>(), 3)),
                   dim);
  test(tr, fe);
}


int
main()
{
  initlog();
  deallog << std::setprecision(2);

  test_hyper_sphere<2>();
  test_hyper_sphere<3>();
}
