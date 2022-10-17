// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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


// check MPI::CollectiveMutex when interacting with exceptions

#include <deal.II/base/mpi.h>

#include <csignal>

#include "../tests.h"

// hook into SIGABRT/SIGFPE and kill off the program
void
signal_handler(int signal)
{
  if (signal == SIGABRT)
    {
      std::cerr << "SIGABRT received\n";
    }
  else if (signal == SIGFPE)
    {
      std::cerr << "SIGFPE received\n";
    }
  else
    {
      std::cerr << "Unexpected signal " << signal << " received\n";
    }
#if DEAL_II_USE_CXX11
  // Kill the program without performing any other cleanup, which is likely to
  // lead to a deadlock
  std::cerr << "Calling _Exit (good)\n";
  std::_Exit(EXIT_FAILURE);
#else
  // Kill the program, or at least try to. The problem when we get here is
  // that calling std::exit invokes at_exit() functions that may still hang
  // the MPI system
  std::cerr << "Calling exit (bad)\n";
  std::exit(1);
#endif
}

void
unguarded(MPI_Comm comm)
{
  int        tag     = 12345;
  const auto my_rank = Utilities::MPI::this_mpi_process(comm);
  const auto n_ranks = Utilities::MPI::n_mpi_processes(comm);

  if (my_rank == 0)
    {
      std::set<int> received_from;
      MPI_Status    status;

      for (unsigned int n = 1; n < n_ranks; ++n)
        {
          unsigned int value;
          MPI_Recv(&value, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, tag, comm, &status);

          AssertThrow(received_from.count(status.MPI_SOURCE) == 0,
                      ExcMessage("oh no!"));
          received_from.insert(status.MPI_SOURCE);
        }
    }
  else
    {
      unsigned int value = 123;
      int          dest  = 0;
      MPI_Send(&value, 1, MPI_UNSIGNED, dest, tag, comm);
    }
}



void
test(MPI_Comm comm)
{
  try
    {
      static Utilities::MPI::CollectiveMutex      mutex;
      Utilities::MPI::CollectiveMutex::ScopedLock lock(mutex, comm);

      const auto my_rank = Utilities::MPI::this_mpi_process(comm);

      if (my_rank == 0)
        {
          Assert(false, ExcInternalError());
        }

      unsigned int value = 123;
      int          dest  = 0;
      MPI_Send(&value, 1, MPI_UNSIGNED, dest, 1, comm);
    }
  catch (::dealii::StandardExceptions::ExcInternalError &exc)
    {
      throw exc;
    }
}



int
main(int argc, char **argv)
{
  std::signal(SIGABRT, signal_handler);
  std::signal(SIGFPE, signal_handler);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  MPILogInitAll();

  test(MPI_COMM_WORLD);
}
