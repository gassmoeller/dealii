/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: XXX
 */


// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/data_out.h>

#include <fstream>

using namespace dealii;


namespace Step66
{
  namespace BoundaryIds
  {
    constexpr types::boundary_id open    = 101;
    constexpr types::boundary_id cathode = 102;
    constexpr types::boundary_id anode   = 103;
  } // namespace BoundaryIds


  template <int dim>
  class CathodeRaySimulator
  {
  public:
    CathodeRaySimulator();

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve_field();
    void refine_grid();

    void create_particles();
    void move_particles();

    void update_timestep_size();
    void output_results() const;

    Triangulation<dim>        triangulation;
    MappingQ<dim>             mapping;
    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;
    AffineConstraints<double> constraints;

    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;

    Vector<double> solution;
    Vector<double> system_rhs;

    Particles::ParticleHandler<dim> particle_handler;

    DiscreteTime time;
  };



  // @sect3{The <code>CathodeRaySimulator</code> class implementation}

  // @sect4{The <code>CathodeRaySimulator</code> constructor}

  template <int dim>
  CathodeRaySimulator<dim>::CathodeRaySimulator()
    : mapping(1)
    , fe(2)
    , dof_handler(triangulation)
    , particle_handler(triangulation, mapping, /*n_properties=*/dim)
    , time(0, 0.2)
  {}



  // @sect4{The <code>CathodeRaySimulator::make_grid</code> function}

  template <int dim>
  void CathodeRaySimulator<dim>::make_grid()
  {
    static_assert(dim == 2,
                  "This function is currently only implemented for 2d.");

    /*
     *   *---*---*---*---*
     *   |   |   |   |   |
     *   **--*---*---*---*
     *   |   |   |   |   |
     *   *---*---*---*---*
     */

    const double       delta                                 = 0.1;
    const unsigned int nx                                    = 5;
    const unsigned int ny                                    = 3;
    const Point<dim>   vertices[nx * ny + 1]                 = {{0, 0},
                                              {1, 0},
                                              {2, 0},
                                              {3, 0},
                                              {4, 0},
                                              {0, 1 - delta},
                                              {1, 1},
                                              {2, 1},
                                              {3, 1},
                                              {4, 1},
                                              {0, 2},
                                              {1, 2},
                                              {2, 2},
                                              {3, 2},
                                              {4, 2},
                                              {0, 1 + delta}};
    const int          cell_vertices[(nx - 1) * (ny - 1)][4] = {
      {0, 1, nx + 0, nx + 1},
      {1, 2, nx + 1, nx + 2},
      {2, 3, nx + 2, nx + 3},
      {3, 4, nx + 3, nx + 4},

      {15, nx + 1, 2 * nx + 0, 2 * nx + 1},
      {nx + 1, nx + 2, 2 * nx + 1, 2 * nx + 2},
      {nx + 2, nx + 3, 2 * nx + 2, 2 * nx + 3},
      {nx + 3, nx + 4, 2 * nx + 3, 2 * nx + 4}};
    std::vector<CellData<dim>> cells((nx - 1) * (ny - 1), CellData<dim>());
    for (unsigned int i = 0; i < cells.size(); ++i)
      {
        for (unsigned int j = 0; j < 4; ++j)
          cells[i].vertices[j] = cell_vertices[i][j];
        cells[i].material_id = 0;
      }

    triangulation.create_triangulation(
      {std::begin(vertices), std::end(vertices)},
      cells,
      SubCellData()); // no boundary information

    triangulation.refine_global(2);

    for (auto &cell : triangulation.active_cell_iterators())
      for (auto &face : cell->face_iterators())
        if (face->at_boundary())
          {
            if ((face->center()[0] > 0) && (face->center()[0] < 1) &&
                (face->center()[1] > 1 - delta) &&
                (face->center()[1] < 1 + delta))
              face->set_boundary_id(BoundaryIds::cathode);
            else if ((face->center()[0] > 4 - 1e-12) &&
                     ((face->center()[1] > 1.25) || (face->center()[1] < 0.75)))
              face->set_boundary_id(BoundaryIds::anode);
            else
              face->set_boundary_id(BoundaryIds::open);
          }

    triangulation.refine_global(1);
  }


  // @sect4{The <code>CathodeRaySimulator::setup_system</code> function}

  template <int dim>
  void CathodeRaySimulator<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             BoundaryIds::cathode,
                                             Functions::ConstantFunction<dim>(
                                               -1),
                                             constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             BoundaryIds::anode,
                                             Functions::ConstantFunction<dim>(
                                               +1),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
  }


  // @sect4{The <code>CathodeRaySimulator::assemble_system</code> function}

  template <int dim>
  void CathodeRaySimulator<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              {
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));           // dx

                // need to replace by point sources
                cell_rhs(i) +=
                  (0 *                                 // f(x)
                   fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   fe_values.JxW(q_index));            // dx
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }


  // @sect4{CathodeRaySimulator::solve}

  template <int dim>
  void CathodeRaySimulator<dim>::solve_field()
  {
    SolverControl            solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);
  }


  // @sect4{CathodeRaySimulator::refine_grid}

  template <int dim>
  void CathodeRaySimulator<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.2,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();
  }


  // @sect4{CathodeRaySimulator::create_particles}

  template <int dim>
  void CathodeRaySimulator<dim>::create_particles()
  {
    FEFaceValues<dim>           fe_face_values(fe,
                                     QGauss<dim - 1>(3),
                                     update_quadrature_points |
                                       update_gradients |
                                       update_normal_vectors);
    std::vector<Tensor<1, dim>> solution_gradients(
      fe_face_values.n_quadrature_points);
    const FEValuesExtractors::Scalar electric_potential(0);

    types::particle_index n_current_particles =
      particle_handler.n_global_particles();

    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() && face->boundary_id() == BoundaryIds::cathode)
          {
            // compute the electric field strength at each quadrature point
            fe_face_values.reinit(cell, face);
            fe_face_values[electric_potential].get_function_gradients(
              solution, solution_gradients);
            for (const unsigned int q_point :
                 fe_face_values.quadrature_point_indices())
              {
                const Tensor<1, dim> E =
                  solution_gradients[q_point]; // need times dielectric constant

                // electrons can only escape the cathode if the electric field
                // strength exceeds a threshold (arbitrarily chosen here) and,
                // crucially, if the electric field points *into* the domain
                const double E_threshold = 1.;
                if ((E * fe_face_values.normal_vector(q_point) < 0) &&
                    (E.norm() > E_threshold))
                  {
                    const Point<dim> location =
                      fe_face_values.quadrature_point(q_point);
                    // create a particle at 'location' and insert it into the
                    // ParticleHandler

                    Particles::Particle<dim> new_particle;
                    new_particle.set_location(location);
                    new_particle.set_id(n_current_particles);
                    // probably need to set the reference location as well?
                    auto it =
                      particle_handler.insert_particle(new_particle, cell);

                    const std::vector<double> initial_velocity(dim, 0.);
                    it->set_properties(initial_velocity);


                    ++n_current_particles;
                  }
              }
          }

    particle_handler.update_cached_numbers();
  }


  // @sect4{CathodeRaySimulator::move_particles}

  template <int dim>
  void CathodeRaySimulator<dim>::move_particles()
  {
    // advance all particles
    const double dt = time.get_next_step_size();

    for (const auto cell : dof_handler.active_cell_iterators())
      if (particle_handler.n_particles_in_cell(cell) > 0)
        {
          std::vector<Point<dim>> reference_points;
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            reference_points.push_back(particle.get_reference_location());

          Quadrature<dim> quad(reference_points);

          FEValues<dim> fe_values(mapping, fe, quad, update_gradients);
          fe_values.reinit(cell);

          std::vector<Tensor<1, dim>> solution_gradients(quad.size());
          fe_values.get_function_gradients(solution, solution_gradients);

          unsigned int particle_index = 0;
          for (auto &particle : particle_handler.particles_in_cell(cell))
            {
              const Tensor<1, dim> E = solution_gradients[particle_index];

              const Tensor<1, dim> acceleration =
                E; // todo: should actually be e*E/m

              const auto     particle_properties = particle.get_properties();
              Tensor<1, dim> old_velocity =
                (dim == 2 ?
                   Point<dim>(particle_properties[0], particle_properties[1]) :
                   Point<dim>(particle_properties[0],
                              particle_properties[1],
                              particle_properties[2]));
              const Tensor<1, dim> new_velocity =
                old_velocity + dt * acceleration;

              const Tensor<1, dim> dx = dt * (old_velocity + new_velocity) / 2;
              const Point<dim>     new_position = particle.get_location() + dx;

              particle.set_location(new_position);

              particle.set_properties(make_array_view(new_velocity));

              // TODO: set new reference location

              ++particle_index;
            }
        }
  }


  // @sect4{CathodeRaySimulator::update_timestep_size}

  template <int dim>
  void CathodeRaySimulator<dim>::update_timestep_size()
  {
    // We need to respect a CFL condition whereby particles can not move further
    // than one cell. Need to compute their speed here and divide the cell size
    // by that speed for all particles, then take the minimum.

    const double cfl = 0.5; // TODO

    double dt = std::numeric_limits<double>::max();

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (particle_handler.n_particles_in_cell(cell) > 0)
        {
          const double minimum_vertex_distance =
            cell->minimum_vertex_distance();

          std::vector<Point<dim>> reference_points;
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            reference_points.push_back(particle.get_reference_location());

          Quadrature<dim> quad(reference_points);

          FEValues<dim> fe_values(mapping, fe, quad, update_gradients);
          fe_values.reinit(cell);

          std::vector<Tensor<1, dim>> solution_gradients(quad.size());
          fe_values.get_function_gradients(solution, solution_gradients);

          double       max_speed      = 0.0;
          unsigned int particle_index = 0;
          for (const auto &particle : particle_handler.particles_in_cell(cell))
            {
              const Tensor<1, dim> E = solution_gradients[particle_index];

              const Tensor<1, dim> acceleration =
                E; // todo: should actually be e*E/m

              const auto     particle_properties = particle.get_properties();
              Tensor<1, dim> old_velocity =
                (dim == 2 ?
                   Point<dim>(particle_properties[0], particle_properties[1]) :
                   Point<dim>(particle_properties[0],
                              particle_properties[1],
                              particle_properties[2]));
              const Tensor<1, dim> new_velocity =
                old_velocity + dt * acceleration;

              max_speed =
                std::max(max_speed, ((old_velocity + new_velocity) / 2).norm());

              ++particle_index;
            }

          if (max_speed > 0.0)
            dt = std::min(dt, cfl * minimum_vertex_distance / max_speed);
        }

    Assert(
      dt < std::numeric_limits<double>::max(),
      ExcMessage(
        "There are no particles from which one could compute the appropriate time step size."));

    time.set_desired_next_step_size(dt);
  }



  // @sect4{The <code>CathodeRaySimulator::output_results()</code> function}
  template <int dim>
  class ElectricFieldPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    ElectricFieldPostprocessor()
      : DataPostprocessorVector<dim>("electric_field", update_gradients)
    {}

    virtual void evaluate_scalar_field(
      const DataPostprocessorInputs::Scalar<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());

      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
          AssertDimension(computed_quantities[p].size(), dim);
          for (unsigned int d = 0; d < dim; ++d)
            computed_quantities[p][d] = /* coefficient */
              1.0 * input_data.solution_gradients[p][d];
        }
    }
  };

  template <int dim>
  void CathodeRaySimulator<dim>::output_results() const
  {
    {
      ElectricFieldPostprocessor<dim> electric_field;
      DataOut<dim>                    data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.add_data_vector(solution, electric_field);
      data_out.build_patches();

      data_out.set_flags(
        DataOutBase::VtkFlags(time.get_current_time(), time.get_step_number()));

      std::ofstream output("solution-" +
                           Utilities::int_to_string(time.get_step_number(), 4) +
                           ".vtu");
      data_out.write_vtu(output);
    }

    {
      Particles::DataOut<dim, dim> particle_out;
      particle_out.build_patches(particle_handler);

      particle_out.set_flags(
        DataOutBase::VtkFlags(time.get_current_time(), time.get_step_number()));

      std::ofstream output("particles-" +
                           Utilities::int_to_string(time.get_step_number(), 4) +
                           ".vtu");
      particle_out.write_vtu(output);
    }
    // TODO: also output particles and their properties
  }


  // @sect4{CathodeRaySimulator::run}

  template <int dim>
  void CathodeRaySimulator<dim>::run()
  {
    make_grid();

    // do a few refinement cycles up front
    const unsigned int n_pre_refinement_cycles = 4;
    for (unsigned int refinement_cycle = 0;
         refinement_cycle < n_pre_refinement_cycles;
         ++refinement_cycle)
      {
        setup_system();
        assemble_system();
        solve_field();
        refine_grid();
      }


    // Now do the loop over time:
    setup_system();
    do
      {
        std::cout << "Timestep " << time.get_step_number() + 1 << std::endl;
        std::cout << "  Field degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve_field();

        create_particles();
        std::cout << "  Number of particles:      "
                  << particle_handler.n_global_particles() << std::endl;

        update_timestep_size();
        move_particles();

        time.advance_time();
        output_results();

        std::cout << std::endl
                  << "  Now at t=" << time.get_current_time()
                  << ", dt=" << time.get_previous_step_size() << '.'
                  << std::endl
                  << std::endl;
      }
    while (time.is_at_end() == false);
  }
} // namespace Step66

// @sect3{The <code>main</code> function}

int main()
{
  try
    {
      Step66::CathodeRaySimulator<2> cathode_ray_simulator_2d;
      cathode_ray_simulator_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
