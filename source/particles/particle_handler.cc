// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2021 by the deal.II authors
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

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle_handler.h>

#include <memory>
#include <utility>

DEAL_II_NAMESPACE_OPEN

namespace Particles
{
  namespace
  {
    template <int dim, int spacedim>
    std::vector<char>
    pack_particles(std::vector<ParticleIterator<dim, spacedim>> &particles)
    {
      std::vector<char> buffer;

      if (particles.size() == 0)
        return buffer;

      buffer.resize(particles.size() *
                    particles.front()->serialized_size_in_bytes());
      void *current_data = buffer.data();

      for (const auto &particle : particles)
        {
          current_data = particle->write_particle_data_to_memory(current_data);
        }

      return buffer;
    }
  } // namespace

  template <int dim, int spacedim>
  ParticleHandler<dim, spacedim>::ParticleHandler()
    : triangulation()
    , mapping()
    , property_pool(std::make_unique<PropertyPool<dim, spacedim>>(0))
    , particles()
    , global_number_of_particles(0)
    , local_number_of_particles(0)
    , global_max_particles_per_cell(0)
    , next_free_particle_index(0)
    , size_callback()
    , store_callback()
    , load_callback()
    , handle(numbers::invalid_unsigned_int)
    , tria_listeners()
  {}



  template <int dim, int spacedim>
  ParticleHandler<dim, spacedim>::ParticleHandler(
    const Triangulation<dim, spacedim> &triangulation,
    const Mapping<dim, spacedim> &      mapping,
    const unsigned int                  n_properties)
    : triangulation(&triangulation, typeid(*this).name())
    , mapping(&mapping, typeid(*this).name())
    , property_pool(std::make_unique<PropertyPool<dim, spacedim>>(n_properties))
    , particles(triangulation.n_active_cells())
    , global_number_of_particles(0)
    , local_number_of_particles(0)
    , global_max_particles_per_cell(0)
    , next_free_particle_index(0)
    , size_callback()
    , store_callback()
    , load_callback()
    , handle(numbers::invalid_unsigned_int)
    , tria_listeners()
  {
    triangulation_cache =
      std::make_unique<GridTools::Cache<dim, spacedim>>(triangulation, mapping);

    connect_to_triangulation_signals();
  }



  template <int dim, int spacedim>
  ParticleHandler<dim, spacedim>::~ParticleHandler()
  {
    clear_particles();

    for (const auto &connection : tria_listeners)
      connection.disconnect();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::initialize(
    const Triangulation<dim, spacedim> &new_triangulation,
    const Mapping<dim, spacedim> &      new_mapping,
    const unsigned int                  n_properties)
  {
    for (const auto &connection : tria_listeners)
      connection.disconnect();

    tria_listeners.clear();

    triangulation = &new_triangulation;
    mapping       = &new_mapping;

    connect_to_triangulation_signals();

    // Create the memory pool that will store all particle properties
    property_pool = std::make_unique<PropertyPool<dim, spacedim>>(n_properties);

    // Create the grid cache to cache the information about the triangulation
    // that is used to locate the particles into subdomains and cells
    triangulation_cache =
      std::make_unique<GridTools::Cache<dim, spacedim>>(new_triangulation,
                                                        new_mapping);

    particles.resize(triangulation->n_active_cells());
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::copy_from(
    const ParticleHandler<dim, spacedim> &particle_handler)
  {
    // clear this object before copying particles
    clear();

    const unsigned int n_properties =
      particle_handler.property_pool->n_properties_per_slot();
    initialize(*particle_handler.triangulation,
               *particle_handler.mapping,
               n_properties);

    property_pool = std::make_unique<PropertyPool<dim, spacedim>>(
      *(particle_handler.property_pool));

    // copy static members
    global_number_of_particles = particle_handler.global_number_of_particles;
    local_number_of_particles  = particle_handler.local_number_of_particles;
    global_max_particles_per_cell =
      particle_handler.global_max_particles_per_cell;
    next_free_particle_index = particle_handler.next_free_particle_index;
    particles                = particle_handler.particles;

    ghost_particles_cache.ghost_particles_by_domain =
      particle_handler.ghost_particles_cache.ghost_particles_by_domain;
    handle = particle_handler.handle;
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::clear()
  {
    clear_particles();
    global_number_of_particles    = 0;
    local_number_of_particles     = 0;
    next_free_particle_index      = 0;
    global_max_particles_per_cell = 0;
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::clear_particles()
  {
    for (auto &particles_in_cell : particles)
      for (auto &particle : particles_in_cell)
        if (particle != PropertyPool<dim, spacedim>::invalid_handle)
          property_pool->deregister_particle(particle);

    particles.clear();
    if (triangulation != nullptr)
      particles.resize(triangulation->n_active_cells());

    // the particle properties have already been deleted by their destructor,
    // but the memory is still allocated. Return the memory as well.
    property_pool->clear();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::update_cached_numbers()
  {
    local_number_of_particles                          = 0;
    types::particle_index local_max_particle_index     = 0;
    types::particle_index local_max_particles_per_cell = 0;

    for (const auto &particles_in_cell : particles)
      {
        const types::particle_index n_particles_in_cell =
          particles_in_cell.size();

        local_max_particles_per_cell =
          std::max(local_max_particles_per_cell, n_particles_in_cell);
        local_number_of_particles += n_particles_in_cell;

        for (const auto &particle : particles_in_cell)
          {
            local_max_particle_index =
              std::max(local_max_particle_index,
                       property_pool->get_id(particle));
          }
      }

    if (const auto parallel_triangulation =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &*triangulation))
      {
        global_number_of_particles = dealii::Utilities::MPI::sum(
          local_number_of_particles,
          parallel_triangulation->get_communicator());

        if (global_number_of_particles == 0)
          {
            next_free_particle_index      = 0;
            global_max_particles_per_cell = 0;
          }
        else
          {
            types::particle_index local_max_values[2] = {
              local_max_particle_index, local_max_particles_per_cell};
            types::particle_index global_max_values[2];

            Utilities::MPI::max(local_max_values,
                                parallel_triangulation->get_communicator(),
                                global_max_values);

            next_free_particle_index      = global_max_values[0] + 1;
            global_max_particles_per_cell = global_max_values[1];
          }
      }
    else
      {
        global_number_of_particles = local_number_of_particles;
        next_free_particle_index =
          global_number_of_particles == 0 ? 0 : local_max_particle_index + 1;
        global_max_particles_per_cell = local_max_particles_per_cell;
      }
  }



  template <int dim, int spacedim>
  types::particle_index
  ParticleHandler<dim, spacedim>::n_particles_in_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    if (particles.size() == 0)
      return 0;

    if (cell->is_artificial() == false)
      {
        return particles[cell->active_cell_index()].size();
      }
    else
      AssertThrow(false,
                  ExcMessage("You can't ask for the particles on an artificial "
                             "cell since we don't know what exists on these "
                             "kinds of cells."));

    return numbers::invalid_unsigned_int;
  }



  template <int dim, int spacedim>
  typename ParticleHandler<dim, spacedim>::particle_iterator_range
  ParticleHandler<dim, spacedim>::particles_in_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
    const
  {
    return (const_cast<ParticleHandler<dim, spacedim> *>(this))
      ->particles_in_cell(cell);
  }



  template <int dim, int spacedim>
  typename ParticleHandler<dim, spacedim>::particle_iterator_range
  ParticleHandler<dim, spacedim>::particles_in_cell(
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    const unsigned int active_cell_index = cell->active_cell_index();

    if (cell->is_artificial() == false)
      {
        if (particles[active_cell_index].size() == 0)
          {
            return boost::make_iterator_range(
              particle_iterator(particles, *property_pool, cell, 0),
              particle_iterator(particles, *property_pool, cell, 0));
          }
        else
          {
            particle_iterator begin(particles, *property_pool, cell, 0);
            particle_iterator end(particles,
                                  *property_pool,
                                  cell,
                                  particles[active_cell_index].size() - 1);
            // end needs to point to the particle after the last one in the
            // cell.
            ++end;

            return boost::make_iterator_range(begin, end);
          }
      }
    else
      AssertThrow(false,
                  ExcMessage("You can't ask for the particles on an artificial "
                             "cell since we don't know what exists on these "
                             "kinds of cells."));

    return {};
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::remove_particle(
    const ParticleHandler<dim, spacedim>::particle_iterator &particle)
  {
    const unsigned int active_cell_index = particle->active_cell_index;

    particle_container &container = *(particle->particles);

    // if the particle has an invalid handle (e.g. because it has
    // been duplicated before calling this function) do not try
    // to deallocate its memory again
    auto handle = particle->get_handle();
    if (handle != PropertyPool<dim, spacedim>::invalid_handle)
      {
        property_pool->deregister_particle(handle);
      }

    if (container[active_cell_index].size() > 1)
      {
        container[active_cell_index][particle->particle_index_within_cell] =
          std::move(particles[active_cell_index].back());
        container[active_cell_index].resize(
          particles[active_cell_index].size() - 1);
      }
    else
      {
        container[active_cell_index].clear();
      }

    --local_number_of_particles;
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::remove_particles(
    const std::vector<ParticleHandler<dim, spacedim>::particle_iterator>
      &particles_to_remove)
  {
    unsigned int n_particles_removed = 0;

    for (unsigned int cell_index = 0; cell_index < particles.size();
         ++cell_index)
      {
        // If there is nothing left to remove, break and return
        if (n_particles_removed == particles_to_remove.size())
          break;

        // Skip cells where there is nothing to remove
        if (particles_to_remove[n_particles_removed]->active_cell_index !=
            cell_index)
          continue;

        const unsigned int n_particles_in_cell = particles[cell_index].size();
        unsigned int       move_to             = 0;
        for (unsigned int move_from = 0; move_from < n_particles_in_cell;
             ++move_from)
          {
            if (n_particles_removed != particles_to_remove.size() &&
                particles_to_remove[n_particles_removed]->active_cell_index ==
                  cell_index &&
                particles_to_remove[n_particles_removed]
                    ->particle_index_within_cell == move_from)
              {
                auto handle =
                  particles_to_remove[n_particles_removed]->get_handle();

                // if some particles have invalid handles (e.g. because they are
                // multiple times in the vector, or have been moved from
                // before), do not try to deallocate their memory again
                if (handle != PropertyPool<dim, spacedim>::invalid_handle)
                  property_pool->deregister_particle(handle);
                ++n_particles_removed;
                continue;
              }
            else
              {
                particles[cell_index][move_to] =
                  std::move(particles[cell_index][move_from]);
                ++move_to;
              }
          }
        particles[cell_index].resize(move_to);
      }

    update_cached_numbers();
  }



  template <int dim, int spacedim>
  typename ParticleHandler<dim, spacedim>::particle_iterator
  ParticleHandler<dim, spacedim>::insert_particle(
    const Particle<dim, spacedim> &                                    particle,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    return insert_particle(particle.get_location(),
                           particle.get_reference_location(),
                           particle.get_id(),
                           cell,
                           particle.get_properties());
  }



  template <int dim, int spacedim>
  typename ParticleHandler<dim, spacedim>::particle_iterator
  ParticleHandler<dim, spacedim>::insert_particle(
    const void *&                                                      data,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell)
  {
    Assert(triangulation != nullptr, ExcInternalError());
    Assert(particles.size() == triangulation->n_active_cells(),
           ExcInternalError());
    Assert(
      cell->is_locally_owned(),
      ExcMessage(
        "You can't insert particles in a cell that is not locally owned."));

    const unsigned int active_cell_index = cell->active_cell_index();
    particles[active_cell_index].push_back(property_pool->register_particle());
    particle_iterator particle_it(particles,
                                  *property_pool,
                                  cell,
                                  particles[active_cell_index].size() - 1);

    data = particle_it->read_particle_data_from_memory(data);

    ++local_number_of_particles;

    return particle_it;
  }



  template <int dim, int spacedim>
  typename ParticleHandler<dim, spacedim>::particle_iterator
  ParticleHandler<dim, spacedim>::insert_particle(
    const Point<spacedim> &     position,
    const Point<dim> &          reference_position,
    const types::particle_index particle_index,
    const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
    const ArrayView<const double> &properties)
  {
    Assert(triangulation != nullptr, ExcInternalError());
    Assert(particles.size() == triangulation->n_active_cells(),
           ExcInternalError());
    Assert(cell.state() == IteratorState::valid, ExcInternalError());
    Assert(
      cell->is_locally_owned(),
      ExcMessage(
        "You tried to insert a particle into a cell that is not locally owned. This is not supported."));

    const unsigned int active_cell_index = cell->active_cell_index();
    particles[active_cell_index].push_back(property_pool->register_particle());
    particle_iterator particle_it(particles,
                                  *property_pool,
                                  cell,
                                  particles[active_cell_index].size() - 1);

    particle_it->set_location(position);
    particle_it->set_reference_location(reference_position);
    particle_it->set_id(particle_index);

    if (properties.size() != 0)
      particle_it->set_properties(properties);

    ++local_number_of_particles;

    return particle_it;
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::insert_particles(
    const std::multimap<
      typename Triangulation<dim, spacedim>::active_cell_iterator,
      Particle<dim, spacedim>> &new_particles)
  {
    for (const auto &cell_and_particle : new_particles)
      insert_particle(cell_and_particle.second, cell_and_particle.first);

    update_cached_numbers();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::insert_particles(
    const std::vector<Point<spacedim>> &positions)
  {
    Assert(triangulation != nullptr, ExcInternalError());

    update_cached_numbers();

    // Determine the starting particle index of this process, which
    // is the highest currently existing particle index plus the sum
    // of the number of newly generated particles of all
    // processes with a lower rank if in a parallel computation.
    const types::particle_index local_next_particle_index =
      get_next_free_particle_index();
    types::particle_index local_start_index = 0;

#ifdef DEAL_II_WITH_MPI
    if (const auto parallel_triangulation =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &*triangulation))
      {
        types::particle_index particles_to_add_locally = positions.size();
        const int             ierr = MPI_Scan(&particles_to_add_locally,
                                  &local_start_index,
                                  1,
                                  DEAL_II_PARTICLE_INDEX_MPI_TYPE,
                                  MPI_SUM,
                                  parallel_triangulation->get_communicator());
        AssertThrowMPI(ierr);
        local_start_index -= particles_to_add_locally;
      }
#endif

    local_start_index += local_next_particle_index;

    auto point_locations =
      GridTools::compute_point_locations_try_all(*triangulation_cache,
                                                 positions);

    auto &cells           = std::get<0>(point_locations);
    auto &local_positions = std::get<1>(point_locations);
    auto &index_map       = std::get<2>(point_locations);
    auto &missing_points  = std::get<3>(point_locations);
    // If a point was not found, throwing an error, as the old
    // implementation of compute_point_locations would have done
    AssertThrow(missing_points.size() == 0,
                VectorTools::ExcPointNotAvailableHere());

    (void)missing_points;

    if (cells.size() == 0)
      return;

    for (unsigned int i = 0; i < cells.size(); ++i)
      for (unsigned int p = 0; p < local_positions[i].size(); ++p)
        insert_particle(positions[index_map[i][p]],
                        local_positions[i][p],
                        local_start_index + index_map[i][p],
                        cells[i]);

    update_cached_numbers();
  }



  template <int dim, int spacedim>
  std::map<unsigned int, IndexSet>
  ParticleHandler<dim, spacedim>::insert_global_particles(
    const std::vector<Point<spacedim>> &positions,
    const std::vector<std::vector<BoundingBox<spacedim>>>
      &                                       global_bounding_boxes,
    const std::vector<std::vector<double>> &  properties,
    const std::vector<types::particle_index> &ids)
  {
    if (!properties.empty())
      {
        AssertDimension(properties.size(), positions.size());
#ifdef DEBUG
        for (const auto &p : properties)
          AssertDimension(p.size(), n_properties_per_particle());
#endif
      }

    if (!ids.empty())
      AssertDimension(ids.size(), positions.size());

    const auto comm = triangulation->get_communicator();

    const auto n_mpi_processes = Utilities::MPI::n_mpi_processes(comm);

    // Compute the global number of properties
    const auto n_global_properties =
      Utilities::MPI::sum(properties.size(), comm);

    // Gather the number of points per processor
    const auto n_particles_per_proc =
      Utilities::MPI::all_gather(comm, positions.size());

    // Calculate all starting points locally
    std::vector<unsigned int> particle_start_indices(n_mpi_processes);

    unsigned int particle_start_index = get_next_free_particle_index();
    for (unsigned int process = 0; process < particle_start_indices.size();
         ++process)
      {
        particle_start_indices[process] = particle_start_index;
        particle_start_index += n_particles_per_proc[process];
      }

    // Get all local information
    const auto cells_positions_and_index_maps =
      GridTools::distributed_compute_point_locations(*triangulation_cache,
                                                     positions,
                                                     global_bounding_boxes);

    // Unpack the information into several vectors:
    // All cells that contain at least one particle
    const auto &local_cells_containing_particles =
      std::get<0>(cells_positions_and_index_maps);

    // The reference position of every particle in the local part of the
    // triangulation.
    const auto &local_reference_positions =
      std::get<1>(cells_positions_and_index_maps);
    // The original index in the positions vector for each particle in the
    // local part of the triangulation
    const auto &original_indices_of_local_particles =
      std::get<2>(cells_positions_and_index_maps);
    // The real spatial position of every particle in the local part of the
    // triangulation.
    const auto &local_positions = std::get<3>(cells_positions_and_index_maps);
    // The MPI process that inserted each particle
    const auto &calling_process_indices =
      std::get<4>(cells_positions_and_index_maps);

    // Create the map of cpu to indices, indicating who sent us what particle
    std::map<unsigned int, std::vector<unsigned int>>
      original_process_to_local_particle_indices_tmp;
    for (unsigned int i_cell = 0;
         i_cell < local_cells_containing_particles.size();
         ++i_cell)
      {
        for (unsigned int i_particle = 0;
             i_particle < local_positions[i_cell].size();
             ++i_particle)
          {
            const unsigned int local_id_on_calling_process =
              original_indices_of_local_particles[i_cell][i_particle];
            const unsigned int calling_process =
              calling_process_indices[i_cell][i_particle];

            original_process_to_local_particle_indices_tmp[calling_process]
              .push_back(local_id_on_calling_process);
          }
      }
    std::map<unsigned int, IndexSet> original_process_to_local_particle_indices;
    for (auto &process_and_particle_indices :
         original_process_to_local_particle_indices_tmp)
      {
        const unsigned int calling_process = process_and_particle_indices.first;
        original_process_to_local_particle_indices.insert(
          {calling_process, IndexSet(n_particles_per_proc[calling_process])});
        std::sort(process_and_particle_indices.second.begin(),
                  process_and_particle_indices.second.end());
        original_process_to_local_particle_indices[calling_process].add_indices(
          process_and_particle_indices.second.begin(),
          process_and_particle_indices.second.end());
        original_process_to_local_particle_indices[calling_process].compress();
      }

    // A map from mpi process to properties, ordered as in the IndexSet.
    // Notice that this ordering may be different from the ordering in the
    // vectors above, since no local ordering is guaranteed by the
    // distribute_compute_point_locations() call.
    // This is only filled if n_global_properties is > 0
    std::map<unsigned int, std::vector<std::vector<double>>>
      locally_owned_properties_from_other_processes;

    // A map from mpi process to ids, ordered as in the IndexSet.
    // Notice that this ordering may be different from the ordering in the
    // vectors above, since no local ordering is guaranteed by the
    // distribute_compute_point_locations() call.
    // This is only filled if ids.size() is > 0
    std::map<unsigned int, std::vector<types::particle_index>>
      locally_owned_ids_from_other_processes;

    if (n_global_properties > 0 || !ids.empty())
      {
        // Gather whom I sent my own particles to, to decide whom to send
        // the particle properties or the ids
        auto send_to_cpu = Utilities::MPI::some_to_some(
          comm, original_process_to_local_particle_indices);

        // Prepare the vector of properties to send
        if (n_global_properties > 0)
          {
            std::map<unsigned int, std::vector<std::vector<double>>>
              non_locally_owned_properties;

            for (const auto &it : send_to_cpu)
              {
                std::vector<std::vector<double>> properties_to_send(
                  it.second.n_elements(),
                  std::vector<double>(n_properties_per_particle()));
                unsigned int index = 0;
                for (const auto el : it.second)
                  properties_to_send[index++] = properties[el];
                non_locally_owned_properties.insert(
                  {it.first, properties_to_send});
              }

            // Send the non locally owned properties to each mpi process
            // that needs them
            locally_owned_properties_from_other_processes =
              Utilities::MPI::some_to_some(comm, non_locally_owned_properties);

            AssertDimension(
              locally_owned_properties_from_other_processes.size(),
              original_process_to_local_particle_indices.size());
          }

        if (!ids.empty())
          {
            std::map<unsigned int, std::vector<types::particle_index>>
              non_locally_owned_ids;
            for (const auto &it : send_to_cpu)
              {
                std::vector<types::particle_index> ids_to_send(
                  it.second.n_elements());
                unsigned int index = 0;
                for (const auto el : it.second)
                  ids_to_send[index++] = ids[el];
                non_locally_owned_ids.insert({it.first, ids_to_send});
              }

            // Send the non locally owned ids to each mpi process
            // that needs them
            locally_owned_ids_from_other_processes =
              Utilities::MPI::some_to_some(comm, non_locally_owned_ids);

            AssertDimension(locally_owned_ids_from_other_processes.size(),
                            original_process_to_local_particle_indices.size());
          }
      }

    // Now fill up the actual particles
    for (unsigned int i_cell = 0;
         i_cell < local_cells_containing_particles.size();
         ++i_cell)
      {
        for (unsigned int i_particle = 0;
             i_particle < local_positions[i_cell].size();
             ++i_particle)
          {
            const unsigned int local_id_on_calling_process =
              original_indices_of_local_particles[i_cell][i_particle];

            const unsigned int calling_process =
              calling_process_indices[i_cell][i_particle];

            const unsigned int index_within_set =
              original_process_to_local_particle_indices[calling_process]
                .index_within_set(local_id_on_calling_process);

            const unsigned int particle_id =
              ids.empty() ?
                local_id_on_calling_process +
                  particle_start_indices[calling_process] :
                locally_owned_ids_from_other_processes[calling_process]
                                                      [index_within_set];

            auto particle_it =
              insert_particle(local_positions[i_cell][i_particle],
                              local_reference_positions[i_cell][i_particle],
                              particle_id,
                              local_cells_containing_particles[i_cell]);

            if (n_global_properties > 0)
              {
                particle_it->set_properties(
                  locally_owned_properties_from_other_processes
                    [calling_process][index_within_set]);
              }
          }
      }

    update_cached_numbers();

    return original_process_to_local_particle_indices;
  }



  template <int dim, int spacedim>
  std::map<unsigned int, IndexSet>
  ParticleHandler<dim, spacedim>::insert_global_particles(
    const std::vector<Particle<dim, spacedim>> &particles,
    const std::vector<std::vector<BoundingBox<spacedim>>>
      &global_bounding_boxes)
  {
    // Store the positions in a vector of points, the ids in a vector of ids,
    // and the properties, if any, in a vector of vector of properties.
    std::vector<Point<spacedim>>       positions;
    std::vector<std::vector<double>>   properties;
    std::vector<types::particle_index> ids;
    positions.resize(particles.size());
    ids.resize(particles.size());
    if (n_properties_per_particle() > 0)
      properties.resize(particles.size(),
                        std::vector<double>(n_properties_per_particle()));

    unsigned int i = 0;
    for (const auto &p : particles)
      {
        positions[i] = p.get_location();
        ids[i]       = p.get_id();
        if (p.has_properties())
          properties[i] = {p.get_properties().begin(),
                           p.get_properties().end()};
        ++i;
      }

    return insert_global_particles(positions,
                                   global_bounding_boxes,
                                   properties,
                                   ids);
  }



  template <int dim, int spacedim>
  types::particle_index
  ParticleHandler<dim, spacedim>::n_global_particles() const
  {
    return global_number_of_particles;
  }



  template <int dim, int spacedim>
  types::particle_index
  ParticleHandler<dim, spacedim>::n_global_max_particles_per_cell() const
  {
    return global_max_particles_per_cell;
  }



  template <int dim, int spacedim>
  types::particle_index
  ParticleHandler<dim, spacedim>::n_locally_owned_particles() const
  {
    return local_number_of_particles;
  }



  template <int dim, int spacedim>
  unsigned int
  ParticleHandler<dim, spacedim>::n_properties_per_particle() const
  {
    return property_pool->n_properties_per_slot();
  }



  template <int dim, int spacedim>
  types::particle_index
  ParticleHandler<dim, spacedim>::get_next_free_particle_index() const
  {
    return next_free_particle_index;
  }



  template <int dim, int spacedim>
  IndexSet
  ParticleHandler<dim, spacedim>::locally_owned_particle_ids() const
  {
    IndexSet set(get_next_free_particle_index());
    for (const auto &p : *this)
      set.add_index(p.get_id());
    set.compress();
    return set;
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::get_particle_positions(
    std::vector<Point<spacedim>> &positions,
    const bool                    add_to_output_vector)
  {
    // There should be one point per particle to gather
    AssertDimension(positions.size(), n_locally_owned_particles());

    unsigned int i = 0;
    for (auto it = begin(); it != end(); ++it, ++i)
      {
        if (add_to_output_vector)
          positions[i] = positions[i] + it->get_location();
        else
          positions[i] = it->get_location();
      }
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::set_particle_positions(
    const std::vector<Point<spacedim>> &new_positions,
    const bool                          displace_particles)
  {
    // There should be one point per particle to fix the new position
    AssertDimension(new_positions.size(), n_locally_owned_particles());

    unsigned int i = 0;
    for (auto it = begin(); it != end(); ++it, ++i)
      {
        if (displace_particles)
          it->set_location(it->get_location() + new_positions[i]);
        else
          it->set_location(new_positions[i]);
      }
    sort_particles_into_subdomains_and_cells();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::set_particle_positions(
    const Function<spacedim> &function,
    const bool                displace_particles)
  {
    // The function should have sufficient components to displace the
    // particles
    AssertDimension(function.n_components, spacedim);

    Vector<double> new_position(spacedim);
    for (auto &particle : *this)
      {
        Point<spacedim> particle_location = particle.get_location();
        function.vector_value(particle_location, new_position);
        if (displace_particles)
          for (unsigned int d = 0; d < spacedim; ++d)
            particle_location[d] += new_position[d];
        else
          for (unsigned int d = 0; d < spacedim; ++d)
            particle_location[d] = new_position[d];
        particle.set_location(particle_location);
      }
    sort_particles_into_subdomains_and_cells();
  }



  template <int dim, int spacedim>
  PropertyPool<dim, spacedim> &
  ParticleHandler<dim, spacedim>::get_property_pool() const
  {
    return *property_pool;
  }



  namespace
  {
    /**
     * This function is used as comparison argument to std::sort to sort the
     * vector of tensors @p center_directions by its scalar product with the
     * @p particle_direction tensor. The sorted indices allow to
     * loop over @p center_directions with increasing angle between
     * @p particle_direction and @p center_directions. This function assumes
     * that @p particle_direction and @p center_directions are normalized
     * to length one before calling this function.
     */
    template <int dim>
    bool
    compare_particle_association(
      const unsigned int                 a,
      const unsigned int                 b,
      const Tensor<1, dim> &             particle_direction,
      const std::vector<Tensor<1, dim>> &center_directions)
    {
      const double scalar_product_a = center_directions[a] * particle_direction;
      const double scalar_product_b = center_directions[b] * particle_direction;

      // The function is supposed to return if a is before b. We are looking
      // for the alignment of particle direction and center direction,
      // therefore return if the scalar product of a is larger.
      return (scalar_product_a > scalar_product_b);
    }
  } // namespace



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::sort_particles_into_subdomains_and_cells()
  {
    Assert(triangulation != nullptr, ExcInternalError());
    Assert(particles.size() == triangulation->n_active_cells(),
           ExcInternalError());

    // TODO: The current algorithm only works for particles that are in
    // the local domain or in ghost cells, because it only knows the
    // subdomain_id of ghost cells, but not of artificial cells. This
    // can be extended using the distributed version of compute point
    // locations.
    // TODO: Extend this function to allow keeping particles on other
    // processes around (with an invalid cell).

    std::vector<particle_iterator> particles_out_of_cell;

    // Reserve some space for particles that need sorting to avoid frequent
    // re-allocation. Guess 25% of particles need sorting. Balance memory
    // overhead and performance.
    particles_out_of_cell.reserve(n_locally_owned_particles() / 4);

    // Now update the reference locations of the moved particles
    std::vector<Point<spacedim>> real_locations;
    std::vector<Point<dim>>      reference_locations;
    real_locations.reserve(global_max_particles_per_cell);
    reference_locations.reserve(global_max_particles_per_cell);

    for (const auto &cell : triangulation->active_cell_iterators())
      {
        const unsigned int active_cell_index = cell->active_cell_index();
        const unsigned int n_pic = particles[active_cell_index].size();

        // Particles can be inserted into arbitrary cells, e.g. if their cell is
        // not known. However, for artificial cells we can not evaluate
        // the reference position of particles. Do not sort particles that are
        // not locally owned, because they will be sorted by the process that
        // owns them.
        if (cell->is_locally_owned() == false)
          {
            continue;
          }

        auto pic = particles_in_cell(cell);

        real_locations.clear();
        for (const auto &particle : pic)
          real_locations.push_back(particle.get_location());

        reference_locations.resize(n_pic);
        mapping->transform_points_real_to_unit_cell(cell,
                                                    real_locations,
                                                    reference_locations);

        auto particle = pic.begin();
        for (const auto &p_unit : reference_locations)
          {
            if (p_unit[0] == std::numeric_limits<double>::infinity() ||
                !GeometryInfo<dim>::is_inside_unit_cell(p_unit))
              particles_out_of_cell.push_back(particle);
            else
              particle->set_reference_location(p_unit);

            ++particle;
          }
      }

    // There are three reasons why a particle is not in its old cell:
    // It moved to another cell, to another subdomain or it left the mesh.
    // Particles that moved to another cell are updated and moved inside the
    // particles vector, particles that moved to another domain are
    // collected in the moved_particles_domain vector. Particles that left
    // the mesh completely are ignored and removed.
    std::map<types::subdomain_id, std::vector<particle_iterator>>
      moved_particles;
    std::map<
      types::subdomain_id,
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
      moved_cells;

    // We do not know exactly how many particles are lost, exchanged between
    // domains, or remain on this process. Therefore we pre-allocate
    // approximate sizes for these vectors. If more space is needed an
    // automatic and relatively fast (compared to other parts of this
    // algorithm) re-allocation will happen.
    using vector_size = typename std::vector<particle_iterator>::size_type;

    std::set<types::subdomain_id> ghost_owners;
    if (const auto parallel_triangulation =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &*triangulation))
      ghost_owners = parallel_triangulation->ghost_owners();

    // Reserve some space for particles that need communication to avoid
    // frequent re-allocation. Guess 25% of particles out of their old cell need
    // communication. Balance memory overhead and performance.
    for (const auto &ghost_owner : ghost_owners)
      moved_particles[ghost_owner].reserve(particles_out_of_cell.size() / 4);
    for (const auto &ghost_owner : ghost_owners)
      moved_cells[ghost_owner].reserve(particles_out_of_cell.size() / 4);

    {
      // Create a map from vertices to adjacent cells using grid cache
      const std::vector<
        std::set<typename Triangulation<dim, spacedim>::active_cell_iterator>>
        &vertex_to_cells = triangulation_cache->get_vertex_to_cell_map();

      // Create a corresponding map of vectors from vertex to cell center
      // using grid cache
      const std::vector<std::vector<Tensor<1, spacedim>>>
        &vertex_to_cell_centers =
          triangulation_cache->get_vertex_to_cell_centers_directions();

      std::vector<unsigned int> neighbor_permutation;

      // Find the cells that the particles moved to.
      for (auto &out_particle : particles_out_of_cell)
        {
          // make a copy of the current cell, since we will modify the
          // variable current_cell in the following but we need the original in
          // the case the particle is not found
          auto current_cell = out_particle->get_surrounding_cell();

          // The cell the particle is in
          Point<dim> current_reference_position;
          bool       found_cell = false;

          // Check if the particle is in one of the old cell's neighbors
          // that are adjacent to the closest vertex
          const unsigned int closest_vertex =
            GridTools::find_closest_vertex_of_cell<dim, spacedim>(
              current_cell, out_particle->get_location(), *mapping);
          Tensor<1, spacedim> vertex_to_particle =
            out_particle->get_location() - current_cell->vertex(closest_vertex);
          vertex_to_particle /= vertex_to_particle.norm();

          const unsigned int closest_vertex_index =
            current_cell->vertex_index(closest_vertex);
          const unsigned int n_neighbor_cells =
            vertex_to_cells[closest_vertex_index].size();

          neighbor_permutation.resize(n_neighbor_cells);
          for (unsigned int i = 0; i < n_neighbor_cells; ++i)
            neighbor_permutation[i] = i;

          const auto &cell_centers =
            vertex_to_cell_centers[closest_vertex_index];
          std::sort(neighbor_permutation.begin(),
                    neighbor_permutation.end(),
                    [&vertex_to_particle, &cell_centers](const unsigned int a,
                                                         const unsigned int b) {
                      return compare_particle_association(a,
                                                          b,
                                                          vertex_to_particle,
                                                          cell_centers);
                    });

          // Search all of the cells adjacent to the closest vertex of the
          // previous cell Most likely we will find the particle in them.
          for (unsigned int i = 0; i < n_neighbor_cells; ++i)
            {
              try
                {
                  typename std::set<typename Triangulation<dim, spacedim>::
                                      active_cell_iterator>::const_iterator
                    cell = vertex_to_cells[closest_vertex_index].begin();

                  std::advance(cell, neighbor_permutation[i]);
                  const Point<dim> p_unit =
                    mapping->transform_real_to_unit_cell(
                      *cell, out_particle->get_location());
                  if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                    {
                      current_cell               = *cell;
                      current_reference_position = p_unit;
                      found_cell                 = true;
                      break;
                    }
                }
              catch (typename Mapping<dim>::ExcTransformationFailed &)
                {}
            }

          if (!found_cell)
            {
              // The particle is not in a neighbor of the old cell.
              // Look for the new cell in the whole local domain.
              // This case is rare.
              const std::pair<const typename Triangulation<dim, spacedim>::
                                active_cell_iterator,
                              Point<dim>>
                current_cell_and_position =
                  GridTools::find_active_cell_around_point<>(
                    *triangulation_cache, out_particle->get_location());
              current_cell               = current_cell_and_position.first;
              current_reference_position = current_cell_and_position.second;

              if (current_cell.state() != IteratorState::valid)
                {
                  // We can find no cell for this particle. It has left the
                  // domain due to an integration error or an open boundary.
                  // Signal the loss and move on.
                  signals.particle_lost(out_particle,
                                        out_particle->get_surrounding_cell());
                  continue;
                }
            }

          // If we are here, we found a cell and reference position for this
          // particle
          out_particle->set_reference_location(current_reference_position);

          // Reinsert the particle into our domain if we own its cell.
          // Mark it for MPI transfer otherwise
          if (current_cell->is_locally_owned())
            {
              const unsigned int active_cell_index =
                current_cell->active_cell_index();

              particles[active_cell_index].push_back(
                particles[out_particle->active_cell_index]
                         [out_particle->particle_index_within_cell]);

              // Avoid deallocating the memory of this particle
              particles[out_particle->active_cell_index]
                       [out_particle->particle_index_within_cell] =
                         PropertyPool<dim, spacedim>::invalid_handle;
            }
          else
            {
              moved_particles[current_cell->subdomain_id()].push_back(
                out_particle);
              moved_cells[current_cell->subdomain_id()].push_back(current_cell);
            }
        }
    }

    // Exchange particles between processors if we have more than one process
#ifdef DEAL_II_WITH_MPI
    if (const auto parallel_triangulation =
          dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
            &*triangulation))
      {
        if (dealii::Utilities::MPI::n_mpi_processes(
              parallel_triangulation->get_communicator()) > 1)
          send_recv_particles(moved_particles, particles, moved_cells);
      }
#endif

    // remove_particles also calls update_cached_numbers()
    remove_particles(particles_out_of_cell);
  } // namespace Particles



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::exchange_ghost_particles(
    const bool enable_cache)
  {
    // Nothing to do in serial computations
    const auto parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &*triangulation);
    if (parallel_triangulation != nullptr)
      {
        if (dealii::Utilities::MPI::n_mpi_processes(
              parallel_triangulation->get_communicator()) == 1)
          return;
      }
    else
      return;

#ifndef DEAL_II_WITH_MPI
    (void)enable_cache;
#else
    // Clear ghost particles and their properties
    for (const auto &cell : triangulation->active_cell_iterators())
      if (cell->is_ghost())
        {
          // Clear particle properties
          for (auto &ghost_particle : particles[cell->active_cell_index()])
            property_pool->deregister_particle(ghost_particle);

          // Clear particles themselves
          particles[cell->active_cell_index()].clear();
        }

    // Clear ghost particles cache and invalidate it
    ghost_particles_cache.ghost_particles_by_domain.clear();
    ghost_particles_cache.valid = false;


    const std::set<types::subdomain_id> ghost_owners =
      parallel_triangulation->ghost_owners();
    for (const auto ghost_owner : ghost_owners)
      ghost_particles_cache.ghost_particles_by_domain[ghost_owner].reserve(
        static_cast<typename std::vector<particle_iterator>::size_type>(
          local_number_of_particles * 0.25));

    const std::vector<std::set<unsigned int>> vertex_to_neighbor_subdomain =
      triangulation_cache->get_vertex_to_neighbor_subdomain();

    for (const auto &cell : triangulation->active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            std::set<unsigned int> cell_to_neighbor_subdomain;
            for (const unsigned int v : cell->vertex_indices())
              {
                cell_to_neighbor_subdomain.insert(
                  vertex_to_neighbor_subdomain[cell->vertex_index(v)].begin(),
                  vertex_to_neighbor_subdomain[cell->vertex_index(v)].end());
              }

            if (cell_to_neighbor_subdomain.size() > 0)
              {
                const particle_iterator_range particle_range =
                  particles_in_cell(cell);

                for (const auto domain : cell_to_neighbor_subdomain)
                  {
                    for (typename particle_iterator_range::iterator particle =
                           particle_range.begin();
                         particle != particle_range.end();
                         ++particle)
                      ghost_particles_cache.ghost_particles_by_domain[domain]
                        .push_back(particle);
                  }
              }
          }
      }

    send_recv_particles(
      ghost_particles_cache.ghost_particles_by_domain,
      particles,
      std::map<
        types::subdomain_id,
        std::vector<
          typename Triangulation<dim, spacedim>::active_cell_iterator>>(),
      enable_cache);
#endif
  }

  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::update_ghost_particles()
  {
    // Nothing to do in serial computations
    const auto parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &*triangulation);
    if (parallel_triangulation == nullptr ||
        dealii::Utilities::MPI::n_mpi_processes(
          parallel_triangulation->get_communicator()) == 1)
      {
        return;
      }


#ifdef DEAL_II_WITH_MPI
    // First clear the current ghost_particle information
    // ghost_particles.clear();
    Assert(ghost_particles_cache.valid,
           ExcMessage(
             "Ghost particles cannot be updated if they first have not been "
             "exchanged at least once with the cache enabled"));


    send_recv_particles_properties_and_location(
      ghost_particles_cache.ghost_particles_by_domain, particles);
#endif
  }



#ifdef DEAL_II_WITH_MPI
  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::send_recv_particles(
    const std::map<types::subdomain_id, std::vector<particle_iterator>>
      &                 particles_to_send,
    particle_container &received_particles,
    const std::map<
      types::subdomain_id,
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>>
      &        send_cells,
    const bool build_cache)
  {
    Assert(triangulation != nullptr, ExcInternalError());
    Assert(received_particles.size() == triangulation->n_active_cells(),
           ExcInternalError());

    ghost_particles_cache.valid = build_cache;

    const auto parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &*triangulation);
    Assert(
      parallel_triangulation,
      ExcMessage(
        "This function is only implemented for parallel::TriangulationBase "
        "objects."));

    // Determine the communication pattern
    const std::set<types::subdomain_id> ghost_owners =
      parallel_triangulation->ghost_owners();
    const std::vector<types::subdomain_id> neighbors(ghost_owners.begin(),
                                                     ghost_owners.end());
    const unsigned int                     n_neighbors = neighbors.size();

    if (send_cells.size() != 0)
      Assert(particles_to_send.size() == send_cells.size(), ExcInternalError());

    // If we do not know the subdomain this particle needs to be send to,
    // throw an error
    Assert(particles_to_send.find(numbers::artificial_subdomain_id) ==
             particles_to_send.end(),
           ExcInternalError());

    // TODO: Implement the shipping of particles to processes that are not
    // ghost owners of the local domain
    for (auto send_particles = particles_to_send.begin();
         send_particles != particles_to_send.end();
         ++send_particles)
      Assert(ghost_owners.find(send_particles->first) != ghost_owners.end(),
             ExcNotImplemented());

    std::size_t n_send_particles = 0;
    for (auto send_particles = particles_to_send.begin();
         send_particles != particles_to_send.end();
         ++send_particles)
      n_send_particles += send_particles->second.size();

    const unsigned int cellid_size = sizeof(CellId::binary_type);

    // Containers for the amount and offsets of data we will send
    // to other processors and the data itself.
    std::vector<unsigned int> n_send_data(n_neighbors, 0);
    std::vector<unsigned int> send_offsets(n_neighbors, 0);
    std::vector<char>         send_data;

    Particle<dim, spacedim> test_particle;
    test_particle.set_property_pool(*property_pool);

    const unsigned int individual_particle_data_size =
      test_particle.serialized_size_in_bytes() +
      (size_callback ? size_callback() : 0);

    const unsigned int individual_total_particle_data_size =
      individual_particle_data_size + cellid_size;

    // Only serialize things if there are particles to be send.
    // We can not return early even if no particles
    // are send, because we might receive particles from other processes
    if (n_send_particles > 0)
      {
        // Allocate space for sending particle data
        send_data.resize(n_send_particles *
                         individual_total_particle_data_size);

        void *data = static_cast<void *>(&send_data.front());

        // Serialize the data sorted by receiving process
        for (unsigned int i = 0; i < n_neighbors; ++i)
          {
            send_offsets[i] = reinterpret_cast<std::size_t>(data) -
                              reinterpret_cast<std::size_t>(&send_data.front());

            const unsigned int n_particles_to_send =
              particles_to_send.at(neighbors[i]).size();

            Assert(static_cast<std::size_t>(n_particles_to_send) *
                       individual_total_particle_data_size ==
                     static_cast<std::size_t>(
                       n_particles_to_send *
                       individual_total_particle_data_size),
                   ExcMessage("Overflow when trying to send particle "
                              "data"));

            for (unsigned int j = 0; j < n_particles_to_send; ++j)
              {
                // If no target cells are given, use the iterator
                // information
                typename Triangulation<dim, spacedim>::active_cell_iterator
                  cell;
                if (send_cells.size() == 0)
                  cell = particles_to_send.at(neighbors[i])[j]
                           ->get_surrounding_cell();
                else
                  cell = send_cells.at(neighbors[i])[j];

                const CellId::binary_type cellid =
                  cell->id().template to_binary<dim>();
                memcpy(data, &cellid, cellid_size);
                data = static_cast<char *>(data) + cellid_size;

                data = particles_to_send.at(neighbors[i])[j]
                         ->write_particle_data_to_memory(data);
                if (store_callback)
                  data =
                    store_callback(particles_to_send.at(neighbors[i])[j], data);
              }
            n_send_data[i] = n_particles_to_send;
          }
      }

    // Containers for the data we will receive from other processors
    std::vector<unsigned int> n_recv_data(n_neighbors);
    std::vector<unsigned int> recv_offsets(n_neighbors);

    {
      const int mpi_tag = Utilities::MPI::internal::Tags::
        particle_handler_send_recv_particles_setup;

      std::vector<MPI_Request> n_requests(2 * n_neighbors);
      for (unsigned int i = 0; i < n_neighbors; ++i)
        {
          const int ierr = MPI_Irecv(&(n_recv_data[i]),
                                     1,
                                     MPI_UNSIGNED,
                                     neighbors[i],
                                     mpi_tag,
                                     parallel_triangulation->get_communicator(),
                                     &(n_requests[2 * i]));
          AssertThrowMPI(ierr);
        }
      for (unsigned int i = 0; i < n_neighbors; ++i)
        {
          const int ierr = MPI_Isend(&(n_send_data[i]),
                                     1,
                                     MPI_UNSIGNED,
                                     neighbors[i],
                                     mpi_tag,
                                     parallel_triangulation->get_communicator(),
                                     &(n_requests[2 * i + 1]));
          AssertThrowMPI(ierr);
        }
      const int ierr =
        MPI_Waitall(2 * n_neighbors, n_requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

    // Determine how many particles and data we will receive
    unsigned int total_recv_data = 0;
    for (unsigned int neighbor_id = 0; neighbor_id < n_neighbors; ++neighbor_id)
      {
        recv_offsets[neighbor_id] = total_recv_data;
        total_recv_data +=
          n_recv_data[neighbor_id] * individual_total_particle_data_size;
      }

    // Set up the space for the received particle data
    std::vector<char> recv_data(total_recv_data);

    // Exchange the particle data between domains
    {
      std::vector<MPI_Request> requests(2 * n_neighbors);
      unsigned int             send_ops = 0;
      unsigned int             recv_ops = 0;

      const int mpi_tag = Utilities::MPI::internal::Tags::
        particle_handler_send_recv_particles_send;

      for (unsigned int i = 0; i < n_neighbors; ++i)
        if (n_recv_data[i] > 0)
          {
            const int ierr =
              MPI_Irecv(&(recv_data[recv_offsets[i]]),
                        n_recv_data[i] * individual_total_particle_data_size,
                        MPI_CHAR,
                        neighbors[i],
                        mpi_tag,
                        parallel_triangulation->get_communicator(),
                        &(requests[send_ops]));
            AssertThrowMPI(ierr);
            send_ops++;
          }

      for (unsigned int i = 0; i < n_neighbors; ++i)
        if (n_send_data[i] > 0)
          {
            const int ierr =
              MPI_Isend(&(send_data[send_offsets[i]]),
                        n_send_data[i] * individual_total_particle_data_size,
                        MPI_CHAR,
                        neighbors[i],
                        mpi_tag,
                        parallel_triangulation->get_communicator(),
                        &(requests[send_ops + recv_ops]));
            AssertThrowMPI(ierr);
            recv_ops++;
          }
      const int ierr =
        MPI_Waitall(send_ops + recv_ops, requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

    // Put the received particles into the domain if they are in the
    // triangulation
    const void *recv_data_it = static_cast<const void *>(recv_data.data());

    // Store the particle iterators in the cache
    auto &ghost_particles_iterators =
      ghost_particles_cache.ghost_particles_iterators;

    if (build_cache)
      {
        ghost_particles_iterators.clear();

        auto &send_pointers_particles = ghost_particles_cache.send_pointers;
        send_pointers_particles.assign(n_neighbors + 1, 0);

        for (unsigned int i = 0; i < n_neighbors; ++i)
          send_pointers_particles[i + 1] =
            send_pointers_particles[i] +
            n_send_data[i] * individual_particle_data_size;

        auto &recv_pointers_particles = ghost_particles_cache.recv_pointers;
        recv_pointers_particles.assign(n_neighbors + 1, 0);

        for (unsigned int i = 0; i < n_neighbors; ++i)
          recv_pointers_particles[i + 1] =
            recv_pointers_particles[i] +
            n_recv_data[i] * individual_particle_data_size;

        ghost_particles_cache.neighbors = neighbors;

        ghost_particles_cache.send_data.resize(
          ghost_particles_cache.send_pointers.back());
        ghost_particles_cache.recv_data.resize(
          ghost_particles_cache.recv_pointers.back());
      }

    while (reinterpret_cast<std::size_t>(recv_data_it) -
             reinterpret_cast<std::size_t>(recv_data.data()) <
           total_recv_data)
      {
        CellId::binary_type binary_cellid;
        memcpy(&binary_cellid, recv_data_it, cellid_size);
        const CellId id(binary_cellid);
        recv_data_it = static_cast<const char *>(recv_data_it) + cellid_size;

        const typename Triangulation<dim, spacedim>::active_cell_iterator cell =
          triangulation->create_cell_iterator(id);

        const unsigned int active_cell_index = cell->active_cell_index();
        received_particles[active_cell_index].emplace_back(
          property_pool->register_particle());
        particle_iterator particle_it(
          received_particles,
          *property_pool,
          cell,
          received_particles[active_cell_index].size() - 1);

        recv_data_it =
          particle_it->read_particle_data_from_memory(recv_data_it);

        if (load_callback)
          recv_data_it = load_callback(particle_it, recv_data_it);

        if (build_cache) // TODO: is this safe?
          ghost_particles_iterators.push_back(particle_it);
      }

    AssertThrow(recv_data_it == recv_data.data() + recv_data.size(),
                ExcMessage(
                  "The amount of data that was read into new particles "
                  "does not match the amount of data sent around."));
  }
#endif



#ifdef DEAL_II_WITH_MPI
  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::send_recv_particles_properties_and_location(
    const std::map<types::subdomain_id, std::vector<particle_iterator>>
      &                 particles_to_send,
    particle_container &updated_particles)
  {
    const auto parallel_triangulation =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &*triangulation);
    Assert(
      parallel_triangulation,
      ExcMessage(
        "This function is only implemented for parallel::TriangulationBase "
        "objects."));

    Assert(updated_particles.size() == parallel_triangulation->n_active_cells(),
           ExcInternalError());

    const auto &neighbors     = ghost_particles_cache.neighbors;
    const auto &send_pointers = ghost_particles_cache.send_pointers;
    const auto &recv_pointers = ghost_particles_cache.recv_pointers;

    std::vector<char> &send_data = ghost_particles_cache.send_data;

    // Fill data to send
    if (send_pointers.back() > 0)
      {
        void *data = static_cast<void *>(&send_data.front());

        // Serialize the data sorted by receiving process
        for (const auto i : neighbors)
          for (const auto &p : particles_to_send.at(i))
            {
              data = p->write_particle_data_to_memory(data);
              if (store_callback)
                data = store_callback(p, data);
            }
      }

    std::vector<char> &recv_data = ghost_particles_cache.recv_data;

    // Exchange the particle data between domains
    {
      std::vector<MPI_Request> requests(2 * neighbors.size());
      unsigned int             send_ops = 0;
      unsigned int             recv_ops = 0;

      const int mpi_tag = Utilities::MPI::internal::Tags::
        particle_handler_send_recv_particles_send;

      for (unsigned int i = 0; i < neighbors.size(); ++i)
        if ((recv_pointers[i + 1] - recv_pointers[i]) > 0)
          {
            const int ierr =
              MPI_Irecv(recv_data.data() + recv_pointers[i],
                        recv_pointers[i + 1] - recv_pointers[i],
                        MPI_CHAR,
                        neighbors[i],
                        mpi_tag,
                        parallel_triangulation->get_communicator(),
                        &(requests[send_ops]));
            AssertThrowMPI(ierr);
            send_ops++;
          }

      for (unsigned int i = 0; i < neighbors.size(); ++i)
        if ((send_pointers[i + 1] - send_pointers[i]) > 0)
          {
            const int ierr =
              MPI_Isend(send_data.data() + send_pointers[i],
                        send_pointers[i + 1] - send_pointers[i],
                        MPI_CHAR,
                        neighbors[i],
                        mpi_tag,
                        parallel_triangulation->get_communicator(),
                        &(requests[send_ops + recv_ops]));
            AssertThrowMPI(ierr);
            recv_ops++;
          }
      const int ierr =
        MPI_Waitall(send_ops + recv_ops, requests.data(), MPI_STATUSES_IGNORE);
      AssertThrowMPI(ierr);
    }

    // Put the received particles into the domain if they are in the
    // triangulation
    const void *recv_data_it = static_cast<const void *>(recv_data.data());

    // Gather ghost particle iterators from the cache
    auto &ghost_particles_iterators =
      ghost_particles_cache.ghost_particles_iterators;

    for (auto &recv_particle : ghost_particles_iterators)
      {
        // Update particle data using previously allocated memory space
        // for efficiency reasons
        recv_data_it =
          recv_particle->read_particle_data_from_memory(recv_data_it);

        if (load_callback)
          recv_data_it = load_callback(
            particle_iterator(updated_particles,
                              *property_pool,
                              recv_particle->cell,
                              recv_particle->particle_index_within_cell),
            recv_data_it);
      }

    AssertThrow(recv_data_it == recv_data.data() + recv_data.size(),
                ExcMessage(
                  "The amount of data that was read into new particles "
                  "does not match the amount of data sent around."));
  }
#endif

  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::register_additional_store_load_functions(
    const std::function<std::size_t()> &                            size_callb,
    const std::function<void *(const particle_iterator &, void *)> &store_callb,
    const std::function<const void *(const particle_iterator &, const void *)>
      &load_callb)
  {
    size_callback  = size_callb;
    store_callback = store_callb;
    load_callback  = load_callb;
  }


  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::connect_to_triangulation_signals()
  {
    tria_listeners.push_back(triangulation->signals.create.connect([&]() {
      this->initialize(*(this->triangulation),
                       *(this->mapping),
                       this->property_pool->n_properties_per_slot());
    }));

    this->tria_listeners.push_back(
      this->triangulation->signals.clear.connect([&]() { this->clear(); }));

    // for distributed triangulations, connect to distributed signals
    if (dynamic_cast<const parallel::DistributedTriangulationBase<dim, spacedim>
                       *>(&(*triangulation)) != nullptr)
      {
        tria_listeners.push_back(
          triangulation->signals.post_distributed_refinement.connect(
            [&]() { this->post_mesh_change_action(); }));
        tria_listeners.push_back(
          triangulation->signals.post_distributed_repartition.connect(
            [&]() { this->post_mesh_change_action(); }));
        tria_listeners.push_back(
          triangulation->signals.post_distributed_load.connect(
            [&]() { this->post_mesh_change_action(); }));
      }
    else
      {
        tria_listeners.push_back(triangulation->signals.post_refinement.connect(
          [&]() { this->post_mesh_change_action(); }));
      }
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::post_mesh_change_action()
  {
    Assert(triangulation != nullptr, ExcInternalError());

    const bool distributed_triangulation =
      dynamic_cast<
        const parallel::DistributedTriangulationBase<dim, spacedim> *>(
        &(*triangulation)) != nullptr;
    (void)distributed_triangulation;

    Assert(
      distributed_triangulation || local_number_of_particles == 0,
      ExcMessage(
        "Mesh refinement in a non-distributed triangulation is not supported by the ParticleHandler class. Either insert particles after mesh creation, or use a distributed triangulation."))

      // Resize the container if it is possible without
      // transferring particles
      if (local_number_of_particles == 0)
        particles.resize(triangulation->n_active_cells());
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::prepare_for_coarsening_and_refinement()
  {
    register_store_callback();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::prepare_for_serialization()
  {
    register_store_callback();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::register_store_callback_function()
  {
    register_store_callback();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::register_store_callback()
  {
    parallel::distributed::Triangulation<dim, spacedim>
      *distributed_triangulation =
        const_cast<parallel::distributed::Triangulation<dim, spacedim> *>(
          dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim>
                         *>(&(*triangulation)));
    (void)distributed_triangulation;

    Assert(distributed_triangulation != nullptr, dealii::ExcNotImplemented());

#ifdef DEAL_II_WITH_P4EST
    // Only save and load particles if there are any, we might get here for
    // example if somebody created a ParticleHandler but generated 0
    // particles.
    update_cached_numbers();

    if (global_max_particles_per_cell > 0)
      {
        const auto callback_function =
          [this](const typename Triangulation<dim, spacedim>::cell_iterator
                   &cell_iterator,
                 const typename Triangulation<dim, spacedim>::CellStatus
                   cell_status) {
            return this->store_particles(cell_iterator, cell_status);
          };

        handle = distributed_triangulation->register_data_attach(
          callback_function, /*returns_variable_size_data=*/true);
      }
#endif
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::transfer_after_coarsening_and_refinement()
  {
    register_load_callback(false);
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::deserialize()
  {
    register_load_callback(true);
  }


  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::register_load_callback_function(
    const bool serialization)
  {
    if (serialization == true)
      deserialize();
    else
      transfer_after_coarsening_and_refinement();
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::register_load_callback(
    const bool serialization)
  {
    parallel::distributed::Triangulation<dim, spacedim>
      *distributed_triangulation =
        const_cast<parallel::distributed::Triangulation<dim, spacedim> *>(
          dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim>
                         *>(&(*triangulation)));
    (void)distributed_triangulation;

    Assert(distributed_triangulation != nullptr, dealii::ExcNotImplemented());

    // First prepare container for insertion
    clear_particles();

#ifdef DEAL_II_WITH_P4EST
    // If we are resuming from a checkpoint, we first have to register the
    // store function again, to set the triangulation in the same state as
    // before the serialization. Only by this it knows how to deserialize the
    // data correctly. Only do this if something was actually stored.
    if (serialization && (global_max_particles_per_cell > 0))
      {
        const auto callback_function =
          [this](const typename Triangulation<dim, spacedim>::cell_iterator
                   &cell_iterator,
                 const typename Triangulation<dim, spacedim>::CellStatus
                   cell_status) {
            return this->store_particles(cell_iterator, cell_status);
          };

        handle = distributed_triangulation->register_data_attach(
          callback_function, /*returns_variable_size_data=*/true);
      }

    // Check if something was stored and load it
    if (handle != numbers::invalid_unsigned_int)
      {
        const auto callback_function =
          [this](
            const typename Triangulation<dim, spacedim>::cell_iterator
              &cell_iterator,
            const typename Triangulation<dim, spacedim>::CellStatus cell_status,
            const boost::iterator_range<std::vector<char>::const_iterator>
              &range_iterator) {
            this->load_particles(cell_iterator, cell_status, range_iterator);
          };

        distributed_triangulation->notify_ready_to_unpack(handle,
                                                        callback_function);

        // Reset handle and update global number of particles. The number
        // can change because of discarded or newly generated particles
        handle = numbers::invalid_unsigned_int;
        update_cached_numbers();
      }
#else
    (void)serialization;
#endif
  }


  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::serialize_particles_pre_local_refinement()
  {
    serialized_particle_handles.reserve(local_number_of_particles);

    for (const auto particle : *this)
      serialized_particle_handles.push_back(
        std::make_pair(particle.cell->id(), particle.get_handle()));
  }

  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::deserialize_particles_post_local_refinement()
  {
    clear_particles();

    for (const auto &particle : serialized_particle_handles)
      {
        const typename Triangulation<dim, spacedim>::cell_iterator cell =
          triangulation->create_cell_iterator(particle.first);
        if (cell->is_active())
          {
            particles[cell->active_cell_index()].push_back(particle.second);
          }
        else if (cell->child(0)->is_active())
          {
            // refined case
            for (unsigned int child_index = 0;
                 child_index < GeometryInfo<dim>::max_children_per_cell;
                 ++child_index)
              {
                const typename Triangulation<dim,
                                             spacedim>::active_cell_iterator
                  child = cell->child(child_index);

                try
                  {
                    const Point<dim> p_unit =
                      mapping->transform_real_to_unit_cell(
                        child, property_pool->get_location(particle.second));
                    if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                      {
                        property_pool->set_reference_location(particle.second,
                                                              p_unit);
                        particles[child->active_cell_index()].push_back(
                          particle.second);
                      }
                  }
                catch (typename Mapping<dim>::ExcTransformationFailed &)
                  {}
              }
          }
        else if (cell->parent()->is_active())
          {
            // coarsened case
            particles[cell->parent()->active_cell_index()].push_back(
              particle.second);

            const Point<dim> p_unit = mapping->transform_real_to_unit_cell(
              cell->parent(), property_pool->get_location(particle.second));
            property_pool->set_reference_location(particle.second, p_unit);
          }
      }
  }


  template <int dim, int spacedim>
  std::vector<char>
  ParticleHandler<dim, spacedim>::store_particles(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const typename Triangulation<dim, spacedim>::CellStatus     status) const
  {
    std::vector<particle_iterator> stored_particles_on_cell;

    switch (status)
      {
        case parallel::TriangulationBase<dim, spacedim>::CELL_PERSIST:
        case parallel::TriangulationBase<dim, spacedim>::CELL_REFINE:
          // If the cell persist or is refined store all particles of the
          // current cell.
          {
            const unsigned int n_particles =
              particles[cell->active_cell_index()].size();
            stored_particles_on_cell.reserve(n_particles);

            for (unsigned int i = 0; i < n_particles; ++i)
              stored_particles_on_cell.push_back(
                particle_iterator(particles, *property_pool, cell, i));
          }
          break;

        case parallel::TriangulationBase<dim, spacedim>::CELL_COARSEN:
          // If this cell is the parent of children that will be coarsened,
          // collect the particles of all children.
          {
            for (const auto &child : cell->child_iterators())
              {
                const unsigned int n_particles = n_particles_in_cell(child);

                stored_particles_on_cell.reserve(
                  stored_particles_on_cell.size() + n_particles);

                for (unsigned int i = 0; i < n_particles; ++i)
                  stored_particles_on_cell.push_back(
                    particle_iterator(particles, *property_pool, child, i));
              }
          }
          break;

        default:
          Assert(false, ExcInternalError());
          break;
      }

    return pack_particles(stored_particles_on_cell);
  }



  template <int dim, int spacedim>
  void
  ParticleHandler<dim, spacedim>::load_particles(
    const typename Triangulation<dim, spacedim>::cell_iterator &    cell,
    const typename Triangulation<dim, spacedim>::CellStatus         status,
    const boost::iterator_range<std::vector<char>::const_iterator> &data_range)
  {
    const auto cell_to_store_particles =
      (status != parallel::TriangulationBase<dim, spacedim>::CELL_REFINE) ?
        cell :
        cell->child(0);

    // deserialize particles and insert into local storage
    {
      const void *data = static_cast<const void *>(&(*data_range.begin()));

      while (data < &(*data_range.end()))
        insert_particle(data, cell_to_store_particles);

      Assert(
        data == &(*data_range.end()),
        ExcMessage(
          "The particle data could not be deserialized successfully. "
          "Check that when deserializing the particles you expect the same "
          "number of properties that were serialized."));
    }

    auto loaded_particles_on_cell = particles_in_cell(cell_to_store_particles);

    // now update particle storage location and properties if necessary
    switch (status)
      {
        case parallel::TriangulationBase<dim, spacedim>::CELL_PERSIST:
          {
            // all particles are correctly inserted
          }
          break;

        case parallel::TriangulationBase<dim, spacedim>::CELL_COARSEN:
          {
            // all particles are in correct cell, but their reference location
            // has changed
            for (auto &particle : loaded_particles_on_cell)
              {
                const Point<dim> p_unit =
                  mapping->transform_real_to_unit_cell(cell_to_store_particles,
                                                       particle.get_location());
                particle.set_reference_location(p_unit);
              }
          }
          break;

        case parallel::TriangulationBase<dim, spacedim>::CELL_REFINE:
          {
            // we need to find the correct child to store the particles and
            // their reference location has changed
            const unsigned int stored_index =
              cell_to_store_particles->active_cell_index();

            // Cannot use range-based loop, because number of particles in cell
            // is going to change
            auto particle = loaded_particles_on_cell.begin();
            for (unsigned int i = 0; i < particles[stored_index].size();)
              {
                for (unsigned int child_index = 0;
                     child_index < GeometryInfo<dim>::max_children_per_cell;
                     ++child_index)
                  {
                    const typename Triangulation<dim, spacedim>::cell_iterator
                      child = cell->child(child_index);

                    try
                      {
                        const Point<dim> p_unit =
                          mapping->transform_real_to_unit_cell(
                            child, particle->get_location());
                        if (GeometryInfo<dim>::is_inside_unit_cell(p_unit))
                          {
                            particle->set_reference_location(p_unit);

                            // if the particle is not in child 0, we stored the
                            // handle in the wrong place; move the handle and
                            // redo the loop; otherwise move on to next particle
                            if (child_index != 0)
                              {
                                particles[child->active_cell_index()].push_back(
                                  particles[stored_index][i]);

                                particles[stored_index][i] =
                                  particles[stored_index].back();
                                particles[stored_index].resize(
                                  particles[stored_index].size() - 1);
                              }
                            else
                              {
                                ++i;
                                ++particle;
                              }
                            break;
                          }
                      }
                    catch (typename Mapping<dim>::ExcTransformationFailed &)
                      {}
                  }
              }
          }
          break;

        default:
          Assert(false, ExcInternalError());
          break;
      }
  }
} // namespace Particles

#include "particle_handler.inst"

DEAL_II_NAMESPACE_CLOSE
