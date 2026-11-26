// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2026 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// Check batched extraction from Tpetra block vectors, including ghost entries,
// empty blocks, and unsorted and repeated indices.

#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/trilinos_tpetra_block_vector.h>

#include "../tests.h"


template <typename MemorySpace>
void
check(const LinearAlgebra::TpetraWrappers::BlockVector<double, MemorySpace> &v,
      const std::vector<types::global_dof_index> &indices)
{
  std::vector<double> entries(indices.size(), -1.);
  v.extract_subvector_to(make_array_view(indices), make_array_view(entries));
  for (unsigned int i = 0; i < indices.size(); ++i)
    AssertThrow(entries[i] == indices[i] + 1., ExcInternalError());

  // Also check virtual dispatch through the vector interface.
  const ReadVector<double> &base = v;
  std::fill(entries.begin(), entries.end(), -1.);
  base.extract_subvector_to(make_array_view(indices), make_array_view(entries));
  for (unsigned int i = 0; i < indices.size(); ++i)
    AssertThrow(entries[i] == indices[i] + 1., ExcInternalError());

  // The other overloads must remain accessible on the derived class.
  v.extract_subvector_to(indices, entries);
  v.extract_subvector_to(indices.begin(), indices.end(), entries.begin());
}


template <typename MemorySpace>
void
test()
{
  using BlockVector =
    LinearAlgebra::TpetraWrappers::BlockVector<double, MemorySpace>;

  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  const std::vector<types::global_dof_index> sizes = {0, 7, 0, 5, 3, 0};
  std::vector<IndexSet>                      owned, relevant;
  for (const auto size : sizes)
    {
      IndexSet partitioning(size);
      partitioning.add_range(size * rank / n_processes,
                             size * (rank + 1) / n_processes);
      owned.push_back(partitioning);
      relevant.push_back(complete_index_set(size));
    }

  BlockVector             v(owned, MPI_COMM_WORLD);
  types::global_dof_index offset = 0;
  for (unsigned int block = 0; block < v.n_blocks(); ++block)
    {
      LinearAlgebra::ReadWriteVector<double> values(owned[block]);
      for (const auto index : owned[block])
        values[index] = offset + index + 1.;
      if (sizes[block] != 0)
        v.block(block).import_elements(values, VectorOperation::insert);
      offset += sizes[block];
    }

  std::vector<types::global_dof_index> indices;
  for (const auto index : v.locally_owned_elements())
    indices.push_back(index);
  check(v, indices);

  std::reverse(indices.begin(), indices.end());
  indices.push_back(indices.front());
  check(v, indices);
  check(v, {});
  check(BlockVector(), {});

  BlockVector ghosted(owned, relevant, MPI_COMM_WORLD);
  ghosted = v;
  check(ghosted, {14, 0, 7, 6, 12, 11, 7, 1, 14});
  check(ghosted, {7, 11, 8, 7});

  deallog << "OK" << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;

  test<MemorySpace::Host>();
  test<MemorySpace::Default>();
}
