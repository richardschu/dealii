/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// C/C++
#include <filesystem>
#include <fstream>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

// // boost
// #include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/text_oarchive.hpp>

using namespace dealii;

template <int dim>
class sample_function : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double sample_function<dim>::value(const Point<dim> &p,
                                   const unsigned int) const
{
  if constexpr (dim == 2)
    {
      return std::sin(p[0] * p[1]);
    }
  else if constexpr (dim == 3)
    {
      return std::sin(p[0] * p[1] * p[2]);
    }
  else
    {
      return std::sin(p[0]);
    }
}

template <int dim>
class ArchiveVector
{
public:
  using VectorType        = LinearAlgebra::distributed::Vector<double>;
  using TriangulationType = parallel::distributed::Triangulation<dim>;
  using SolutionTransferType =
    parallel::distributed::SolutionTransfer<dim, VectorType>;

  ArchiveVector();

  void run();

private:
  void setup_and_serialize(const unsigned int fe_degree,
                           const unsigned int n_refine_global) const;

  template <int n_components>
  void output_vector(const DoFHandler<dim> &dof_handler,
                     const Mapping<dim>    &mapping,
                     const VectorType      &vector,
                     const std::string     &filename_basis) const;

  void
  deserialize_and_check_hp_conversion(const unsigned int fe_degree,
                                      const unsigned int n_refine_global) const;

  void deserialize_and_check_remote_point_evaluation(
    const unsigned int fe_degree,
    const unsigned int n_refine_global) const;

  MPI_Comm            mpi_comm;
  ConditionalOStream  pcout;
  std::string const   filename_reference        = "checkpoint_reference";
  const unsigned int  fe_degree_reference       = 2;
  const unsigned int  n_refine_global_reference = 4;
  const MappingQ<dim> mapping;
};

template <int dim>
ArchiveVector<dim>::ArchiveVector()
  : mpi_comm(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0))
  , mapping(1)
{}

template <int dim>
template <int n_components>
void ArchiveVector<dim>::output_vector(const DoFHandler<dim> &dof_handler,
                                       const Mapping<dim>    &mapping,
                                       const VectorType      &vector,
                                       const std::string &filename_basis) const
{
  pcout << "Exporting vector to vtu.\n";

  // Write higher order output.
  DataOut<dim>          data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);

  // Get vector with locally relevant entries.
  VectorType rel_vector;
  IndexSet   rel_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, rel_dofs);
  rel_vector.reinit(dof_handler.locally_owned_dofs(), rel_dofs, mpi_comm);
  rel_vector = vector;

  // Vector entries are to be interpreted as components of a vector.
  if constexpr (n_components > 1)
    {
      std::vector<
        dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim,
          dealii::DataComponentInterpretation::component_is_part_of_vector);
      std::vector<std::string> solution_names(n_components, "vector");
      data_out.add_data_vector(rel_vector,
                               "vector",
                               dealii::DataOut<dim>::type_dof_data,
                               data_component_interpretation);
    }
  else
    {
      data_out.add_data_vector(rel_vector, "vector");
    }

  const auto &triangulation = dof_handler.get_triangulation();

  // Add vector indicating subdomain.
  Vector<float> subdomain;
  if constexpr (true)
    {
      subdomain.reinit(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");
    }

  // Build patches.
  unsigned int n_subdivisions =
    triangulation.all_reference_cells_are_hyper_cube() ? 3 : 1;
  data_out.build_patches(mapping,
                         n_subdivisions,
                         dealii::DataOut<dim>::curved_inner_cells);

  // Create vtu files + pvtu record.
  std::string filename =
    "./" + filename_basis + "_p" +
    dealii::Utilities::int_to_string(triangulation.locally_owned_subdomain(),
                                     4);
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  // Combine outputs using mpi-thread 0.
  if (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < dealii::Utilities::MPI::n_mpi_processes(mpi_comm);
           ++i)
        {
          filenames.push_back(filename_basis + "_p" +
                              dealii::Utilities::int_to_string(i, 4) + ".vtu");
        }

      // Combine outputs of individual threads.
      std::ofstream master_output(("./" + filename_basis + ".pvtu").c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
}

template <int dim>
void ArchiveVector<dim>::setup_and_serialize(
  const unsigned int fe_degree,
  const unsigned int n_refine_global) const
{
  pcout << "Setting up and filling vector.\n";
  
  // Create and serialize coarse triangulation.
  Triangulation<dim> coarse_triangulation;
  GridGenerator::hyper_cube(coarse_triangulation);
  if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      coarse_triangulation.save("coarse_triangulation");
    }

  // Create triangulation based on coarse triangulation.
  TriangulationType triangulation(mpi_comm);
  triangulation.copy_triangulation(coarse_triangulation);
  coarse_triangulation.clear();
  triangulation.refine_global(n_refine_global);

  // Setup DoFs and fill vector with function interpolation.
  DoFHandler<dim> dof_handler(triangulation);
  const FE_Q<dim> fe(fe_degree);
  dof_handler.distribute_dofs(fe);

  IndexSet rel_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, rel_dofs);
  VectorType vector(dof_handler.locally_owned_dofs(), rel_dofs, mpi_comm);

  VectorTools::interpolate(dof_handler, sample_function<dim>(), vector);

  vector.update_ghost_values(); // turned out to be required

  // Output the vector.
  pcout << "output vector.l2_norm() = " << vector.l2_norm() << "\n";
  output_vector<1>(dof_handler, mapping, vector, "reference");

  // Serialize the vector using SolutionTransferType.
  SolutionTransferType solution_transfer(dof_handler);
  solution_transfer.prepare_for_serialization(vector);

  pcout << "Serializing vector with "
        << "fe_degree = " << fe_degree << ", "
        << "n_refine_global = " << n_refine_global << ".\n";
  triangulation.save(filename_reference);
}

template <int dim>
void ArchiveVector<dim>::deserialize_and_check_hp_conversion(
  const unsigned int fe_degree,
  const unsigned int n_refine_global) const
{
  pcout << "Deserializing and checking vector with "
        << "fe_degree = " << fe_degree << ", "
        << "n_refine_global = " << n_refine_global << ".\n";

  // Deserialize the coarse triangulation.
  Triangulation<dim> coarse_triangulation;
  coarse_triangulation.load("coarse_triangulation");

  // Deserialize the triangulation. That is, recreate the coarse mesh 
  // in some way, and then call `load()` to deserialize the triangulation. 
  TriangulationType triangulation(mpi_comm);
  triangulation.copy_triangulation(coarse_triangulation);
  triangulation.load(filename_reference);

  // Deserialize the vector as stored in the triangulation.
  DoFHandler<dim> dof_handler(triangulation);
  const FE_Q<dim> fe(fe_degree_reference);
  dof_handler.distribute_dofs(fe);

  IndexSet rel_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, rel_dofs);
  VectorType vector(dof_handler.locally_owned_dofs(), rel_dofs, mpi_comm);
  
  SolutionTransferType solution_transfer(dof_handler);
  solution_transfer.deserialize(vector);
  
  vector.update_ghost_values(); // turned out to be required (?)

  // Output the vector.
  pcout << "output vector.l2_norm() = " << vector.l2_norm() << "\n";
  output_vector<1>(dof_handler, mapping, vector, "comparison");

  // Perform here the hp-refinement/coarsening.
  // OR
  // Perform some grid-to-grid interpolation using RemotePointEvaluation.

  // Target has similar domain but can use entirely different FE space.
  // GridGenerator::hyper_cube(triangulation);
  // triangulation.refine_global(n_refine_global);
  // DoFHandler<dim> dof_handler(triangulation);
  // const FE_Q<dim> fe(fe_degree);
  // dof_handler.distribute_dofs(fe);
}

template <int dim>
void ArchiveVector<dim>::deserialize_and_check_remote_point_evaluation(
  const unsigned int fe_degree,
  const unsigned int n_refine_global) const
{
  // this will use remote point evaluation to interpolate the solution on the
  // serialized grid.
  (void)fe_degree;
  (void)n_refine_global;
}

template <int dim>
void ArchiveVector<dim>::run()
{
  setup_and_serialize(fe_degree_reference, n_refine_global_reference);
  
  deserialize_and_check_hp_conversion(3, 3);
  
  // deserialize_and_check_remote_point_evaluation(3, 3);
}

int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      ArchiveVector<3> archive_vector;
      archive_vector.run();
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
