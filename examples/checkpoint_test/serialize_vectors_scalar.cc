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
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

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

namespace ExaDG // exadg/include/exacg/operators/solution_interpolation_between_triangulations.h
{
  /**
   * Class to transfer solutions between different DoFHandlers via
   * interpolation. This class requires, that the destination DoFHandler has
   * generalized support points.
   */
  template <int dim, int n_components>
  class SolutionInterpolationBetweenTriangulations
  {
  public:
    /**
     * Set source and target triangulations on which solution transfer is
     * performed.
     *
     * @param[in] dof_handler_dst_in Target DofHandler.
     * @param[in] mapping_dst_in Target Mapping.
     * @param[in] dof_handler_src_in Source DofHandler.
     * @param[in] mapping_src_in Source Mapping.
     */
    void reinit(const dealii::DoFHandler<dim> &dof_handler_dst_in,
                const dealii::Mapping<dim>    &mapping_dst_in,
                const dealii::DoFHandler<dim> &dof_handler_src_in,
                const dealii::Mapping<dim>    &mapping_src_in)
    {
      AssertThrow(
        dof_handler_dst_in.get_fe().has_generalized_support_points(),
        dealii::ExcMessage(
          "Solution can only be interpolated to finite elements that have support points."));

      dof_handler_dst = &dof_handler_dst_in;
      dof_handler_src = &dof_handler_src_in;

      rpe.reinit(collect_mapped_support_points(dof_handler_dst_in,
                                               mapping_dst_in),
                 dof_handler_src_in.get_triangulation(),
                 mapping_src_in);
    }

    /**
     * Interpolate the solution from a source to a target triangulation.
     * At support points which are not overlapping the source triangulation,
     * corresponding degrees of freedom are set to 0.
     *
     * @param[in] dst Target DoF Vector.
     * @param[in] src Source DoF Vector.
     */
    template <typename VectorType1, typename VectorType2>
    void interpolate_solution(
      VectorType1                                                &dst,
      const VectorType2                                          &src,
      dealii::VectorTools::EvaluationFlags::EvaluationFlags const flags =
        dealii::VectorTools::EvaluationFlags::avg) const
    {
      AssertThrow(src.size() == dof_handler_src->n_dofs(),
                  dealii::ExcMessage("Dimensions do not fit."));
      AssertThrow(dst.size() == dof_handler_dst->n_dofs(),
                  dealii::ExcMessage("Dimensions do not fit."));

      src.update_ghost_values();
      const auto values = dealii::VectorTools::point_values<n_components>(
        rpe, *dof_handler_src, src, flags);
      src.zero_out_ghost_values();

      fill_dof_vector_with_values(dst, *dof_handler_dst, values);
    }

  private:
    std::vector<dealii::Point<dim>>
    collect_mapped_support_points(const dealii::DoFHandler<dim> &dof_handler,
                                  const dealii::Mapping<dim>    &mapping) const
    {
      std::vector<dealii::Point<dim>> support_points;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              auto cellwise_support_points =
                cell->get_fe().get_generalized_support_points();

              for (auto &csp : cellwise_support_points)
                csp = mapping.transform_unit_to_real_cell(cell, csp);

              support_points.insert(support_points.end(),
                                    cellwise_support_points.begin(),
                                    cellwise_support_points.end());
            }
        }

      return support_points;
    }

    template <typename VectorType, typename value_type>
    void
    fill_dof_vector_with_values(VectorType                    &dst,
                                const dealii::DoFHandler<dim> &dof_handler,
                                const std::vector<value_type> &values) const
    {
      auto ptr = values.begin();
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              const auto &fe = cell->get_fe();

              std::vector<double> dof_values(fe.n_dofs_per_cell());
              const unsigned int  n_support_points =
                fe.get_generalized_support_points().size();
              std::vector<dealii::Vector<double>> component_dof_values(
                n_support_points, dealii::Vector<double>(n_components));

              for (unsigned int i = 0; i < n_support_points; ++i)
                {
                  if constexpr (n_components == 1)
                    {
                      component_dof_values[i] = *ptr;
                    }
                  else
                    {
                      component_dof_values[i] =
                        std::move(dealii::Vector<double>(ptr->begin_raw(),
                                                         ptr->end_raw()));
                    }

                  ++ptr;
                }

              fe.convert_generalized_support_point_values_to_dof_values(
                component_dof_values, dof_values);

              cell->set_dof_values(
                dealii::Vector<typename VectorType::value_type>(
                  dof_values.begin(), dof_values.end()),
                dst);
            }
        }
    }

    dealii::SmartPointer<const dealii::DoFHandler<dim>> dof_handler_dst;
    dealii::SmartPointer<const dealii::DoFHandler<dim>> dof_handler_src;
    dealii::Utilities::MPI::RemotePointEvaluation<dim>  rpe;
  };

} // namespace ExaDG

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

  void deserialize(TriangulationType &triangulation,
                   DoFHandler<dim>   &dof_handler,
                   VectorType        &vector) const;

  void hp_conversion(const VectorType   &vector_in,
                     Triangulation<dim> &triangulation,
                     const unsigned int  fe_degree_in,
                     const unsigned int  n_refine_global_in,
                     const unsigned int  fe_degree_out,
                     const unsigned int  n_refine_global_out) const;

  template <int fe_degree>
  void deserialize_and_check_remote_point_evaluation(
    const unsigned int n_refine_global) const;

  void output_points(const Triangulation<dim>              &triangulation,
                     const std::vector<dealii::Point<dim>> &points,
                     const std::string                     &filename) const;

  template <int fe_degree>
  std::vector<Point<dim>>
  collect_integration_points(const DoFHandler<dim> &dof_handler,
                             const Quadrature<dim> &quadrature) const;

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
  if (Utilities::MPI::this_mpi_process(mpi_comm) == 0)
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
  FE_Q<dim> const fe(fe_degree);
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
void ArchiveVector<dim>::deserialize(TriangulationType &triangulation,
                                     DoFHandler<dim>   &dof_handler,
                                     VectorType        &vector) const
{
  pcout << "Deserializing and checking vector.\n";

  // Deserialize the coarse triangulation.
  Triangulation<dim> coarse_triangulation;
  coarse_triangulation.load("coarse_triangulation");

  // Deserialize the triangulation. That is, recreate the coarse mesh
  // in some way, and then call `load()` to deserialize the triangulation.
  triangulation.copy_triangulation(coarse_triangulation);
  triangulation.load(filename_reference);

  // Deserialize the vector as stored in the triangulation.
  dof_handler.clear();
  dof_handler.reinit(triangulation);
  FE_Q<dim> const fe(fe_degree_reference);
  dof_handler.distribute_dofs(fe);

  IndexSet rel_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, rel_dofs);
  vector.reinit(dof_handler.locally_owned_dofs(), rel_dofs, mpi_comm);

  SolutionTransferType solution_transfer(dof_handler);
  solution_transfer.deserialize(vector);

  // Output the vector.
  pcout << "output vector.l2_norm() = " << vector.l2_norm() << "\n";
  output_vector<1>(dof_handler, mapping, vector, "reference_read_back");
}

template <int dim>
void ArchiveVector<dim>::deserialize_and_check_hp_conversion(
  const unsigned int fe_degree,
  const unsigned int n_refine_global) const
{
  // Serialization based on a common coarse grid and FE_Q space.
  // Here specifically, we assume that we are using FE_Q elements,
  // possibly different polynomial orders and different refinement levels.
  pcout << "Deserializing and checking vector with "
        << "fe_degree = " << fe_degree << ", "
        << "n_refine_global = " << n_refine_global << ".\n";

  TriangulationType triangulation(mpi_comm);
  VectorType        vector;
  DoFHandler<dim>   dof_handler;
  deserialize(triangulation, dof_handler, vector);

  hp_conversion(vector,
                triangulation,
                fe_degree_reference,
                n_refine_global_reference,
                fe_degree,
                n_refine_global);
}

// Utility function to perform hp conversion in the same grid using FE_Q
// elements.
template <int dim>
void ArchiveVector<dim>::hp_conversion(
  const VectorType   &vector_in,
  Triangulation<dim> &triangulation,
  const unsigned int  fe_degree_in,
  const unsigned int  n_refine_global_in,
  const unsigned int  fe_degree_out,
  const unsigned int  n_refine_global_out) const
{
  // Setup new DoFHandler with FE_Q elements.
  DoFHandler<dim> dof_handler_combined(triangulation);

  FE_Q<dim> const       fe_in(fe_degree_in);
  FE_Q<dim> const       fe_out(fe_degree_out);
  hp::FECollection<dim> fe_collection(fe_in, fe_out);

  for (const auto &cell : dof_handler_combined.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell->set_active_fe_index(0);
        }
    }

  dof_handler_combined.distribute_dofs(fe_collection);

  VectorType vector_out(dof_handler_combined.locally_owned_dofs(), mpi_comm);
  vector_out = vector_in;

  dealii::IndexSet relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(
    dof_handler_combined, relevant_dofs); // RELEVANT DOFS EXTRACTED HERE

  unsigned int current_refinement_lvl = n_refine_global_in;
  bool         p_conversion_done      = false;
  while (not p_conversion_done and
         current_refinement_lvl != n_refine_global_out)
    {
      VectorType rel_vector_out;
      rel_vector_out.reinit(dof_handler_combined.locally_owned_dofs(),
                            relevant_dofs,
                            mpi_comm);
      rel_vector_out = vector_out;

      // Set FUTURE FE index such that more expensive conversions are done last.
      bool p_conversion_done_in_this_cycle = false;
      if (not p_conversion_done)
        {
          if (fe_degree_out <= fe_degree_in)
            {
              pcout << "  performing p conversion in first cycle since "
                    << fe_degree_out << " <= "
                    << "fe_degree_in"
                    << " \n";

              // Conversion in the first cycle renders later cycles cheaper.
              for (const auto &cell :
                   dof_handler_combined.active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    {
                      cell->set_future_fe_index(1);
                    }
                }
              p_conversion_done_in_this_cycle = true;
            }
          else
            {
              // Conversion in the last cycle renders earlier cycles cheaper.
              if (std::abs(static_cast<int>(n_refine_global_out) -
                           static_cast<int>(current_refinement_lvl)) == 1)
                {
                  pcout << "  performing p conversion in last cycle since "
                        << fe_degree_out << " > " << fe_degree_in << " \n";

                  for (const auto &cell :
                       dof_handler_combined.active_cell_iterators())
                    {
                      if (cell->is_locally_owned())
                        {
                          cell->set_future_fe_index(1);
                        }
                    }
                  p_conversion_done_in_this_cycle = true;
                }
            }
        }

      // Flag for h refinement/coarsening.
      if (current_refinement_lvl < n_refine_global_out)
        {
          pcout << "  flags for refining in space\n";
          triangulation.set_all_refine_flags();
          current_refinement_lvl += 1;
        }
      else if (current_refinement_lvl > n_refine_global_out)
        {
          pcout << "  flags coarsening in space\n";
          for (const auto &cell : triangulation.active_cell_iterators())
            {
              cell->clear_refine_flag();
              cell->set_coarsen_flag();
            }
          current_refinement_lvl -= 1;
        }

      // Prepare for coarsening and refinement.
      pcout << "  preparing for coarsening and refinement\n";
      triangulation.prepare_coarsening_and_refinement();
      SolutionTransferType solution_transfer(dof_handler_combined,
                                             true /* average_values */);
      solution_transfer.prepare_for_coarsening_and_refinement(rel_vector_out);

      pcout << "executing coarsening and refinement\n";
      triangulation.execute_coarsening_and_refinement();

      // Set ACTIVE FE index such that more expensive conversions are done last.
      // Same logic as above, we convert all FEs in the same cycle, so we simply
      // use one bool instead of repeating the logic here.
      if (p_conversion_done_in_this_cycle)
        {
          for (const auto &cell : dof_handler_combined.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  cell->set_active_fe_index(1);
                }
            }
          p_conversion_done = true;
        }

      dof_handler_combined.distribute_dofs(fe_collection);
      relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_combined);
      rel_vector_out.reinit(dof_handler_combined.locally_owned_dofs(),
                            relevant_dofs,
                            mpi_comm);

      pcout << "-> interpolating solution in new FE space\n";
      solution_transfer.interpolate(rel_vector_out);

      vector_out.reinit(dof_handler_combined.locally_owned_dofs(), mpi_comm);
      vector_out = rel_vector_out;
    }

  // Output the vector.
  pcout << "output vector.l2_norm() = " << vector_out.l2_norm() << "\n";
  output_vector<1>(dof_handler_combined,
                   mapping,
                   vector_out,
                   "comparison_p_" + std::to_string(fe_degree_out) + "_lvl_" +
                     std::to_string(n_refine_global_out));
}

template <int dim>
void ArchiveVector<dim>::output_points(
  const Triangulation<dim>              &triangulation,
  const std::vector<dealii::Point<dim>> &points,
  const std::string                     &filename) const
{
  Particles::ParticleHandler<dim, dim> particle_handler(triangulation, mapping);

  particle_handler.insert_particles(points);

  Particles::DataOut<dim, dim> particle_output;
  particle_output.build_patches(particle_handler);
  particle_output.write_vtu_with_pvtu_record("./",
                                             filename,
                                             0 /* counter */,
                                             mpi_comm);
}

template <int dim>
template <int fe_degree>
std::vector<Point<dim>> ArchiveVector<dim>::collect_integration_points(
  const DoFHandler<dim> &dof_handler,
  const Quadrature<dim> &quadrature) const
{
  const Triangulation<dim> &triangulation = dof_handler.get_triangulation();

  using VectorizedArrayType = VectorizedArray<double>;
  typename MatrixFree<dim, double, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double, VectorizedArrayType>::AdditionalData::none;
  additional_data.mapping_update_flags =
    update_quadrature_points; // update_values | update_JxW_values;

  MatrixFree<dim, double, VectorizedArrayType> matrix_free;
  AffineConstraints<double>                    empty_constraints;
  matrix_free.reinit(
    mapping, dof_handler, empty_constraints, quadrature, additional_data);
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1 /* n_components */, double>
    fe_eval(matrix_free);

  std::vector<Point<dim>> points;
  points.reserve(triangulation.n_active_cells() *
                 quadrature.size()); // conservative estimate

  for (unsigned int cell_batch_idx = 0;
       cell_batch_idx < matrix_free.n_cell_batches();
       ++cell_batch_idx)
    {
      fe_eval.reinit(cell_batch_idx);
      for (const unsigned int q : fe_eval.quadrature_point_indices())
        {
          const Point<dim, VectorizedArrayType> cell_batch_points =
            fe_eval.quadrature_point(q);
          for (unsigned int i = 0; i < VectorizedArrayType::size(); ++i)
            {
              Point<dim> p;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  p[d] = cell_batch_points[d][i];
                }
              points.push_back(p);
            }
        }
    }

  // Output the integration points to a file.
  output_points(triangulation, points, "integration_points_target");

  return points;
}

template <int dim>
template <int fe_degree>
void ArchiveVector<dim>::deserialize_and_check_remote_point_evaluation(
  const unsigned int n_refine_global) const
{
  // Interpolation onto a non-matching grid with arbitrary FE space.
  pcout << "Mesh to mesh interpolation with arbitrary FE space.\n";

  // Deserialize the grid and vector.
  VectorType        source_vector;
  TriangulationType source_triangulation(mpi_comm);
  DoFHandler<dim>   dof_handler_source;
  deserialize(source_triangulation, dof_handler_source, source_vector);

  // Create target grid: ball within the source triangulation.
  Point<dim> center;
  for (unsigned int i = 0; i < dim; ++i)
    {
      center[i] = 0.5;
    }
  const double      radius = 0.5;
  TriangulationType target_triangulation(mpi_comm);
  GridGenerator::hyper_ball_balanced(target_triangulation, center, radius);
  target_triangulation.refine_global(n_refine_global);

  DoFHandler<dim> dof_handler_target(target_triangulation);
  FE_Q<dim> const fe(fe_degree);
  dof_handler_target.distribute_dofs(fe);

  if constexpr (false)
    {
      // Collect integration points from target grid.
      QGauss<dim>                   quadrature(fe_degree + 1);
      const std::vector<Point<dim>> integration_points =
        collect_integration_points<fe_degree>(dof_handler_target, quadrature);

      // Setup RemotePointEvaluation.
      typename Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData
        rpe_data(1e-12 /* tolerance in reference cell */,
                 true /* enforce_unique_mapping */,
                 0 /* rtree_level */,
                 {} /* marked_vertices */);
      Utilities::MPI::RemotePointEvaluation<dim> rpe(rpe_data);

      // Missing: One could do a projection using RemotePointEvaluation,
      // but we will use the SolutionInterpolationBetweenTriangulations class
      // instead.

      // Solve projection on the target grid querying RemotePointEvaluation.
      VectorType vector_target(dof_handler_target.locally_owned_dofs(),
                               mpi_comm);
      AffineConstraints<double> empty_constraints;
      empty_constraints.close();
      VectorTools::project(
        mapping,
        dof_handler_target,
        empty_constraints,
        quadrature,
        sample_function<dim>(), // uses the analytical function
        vector_target);
    }
  else
    {
      pcout
        << "Using SolutionInterpolationBetweenTriangulations::interpolate_solution.\n";
      ExaDG::SolutionInterpolationBetweenTriangulations<dim,
                                                        1 /* n_components */>
        interp;
      interp.reinit(dof_handler_target, mapping, dof_handler_source, mapping);

      IndexSet rel_dofs_target;
      DoFTools::extract_locally_relevant_dofs(dof_handler_target,
                                              rel_dofs_target);
      VectorType vector_target(dof_handler_target.locally_owned_dofs(),
                               rel_dofs_target,
                               mpi_comm);

      interp.interpolate_solution(vector_target, source_vector);

      VectorType vector_out(dof_handler_target.locally_owned_dofs(),
                            rel_dofs_target,
                            mpi_comm);
      vector_out = vector_target;

      output_vector<1>(dof_handler_target, mapping, vector_out, "target");
    }
}

template <int dim>
void ArchiveVector<dim>::run()
{
  // Serialization and deserialization can be run using a different
  // number of MPI ranks. To test this, uncomment the following, run
  // the serialization, and then run the deserialization with a different
  // number of MPI ranks.

  // setup_and_serialize(fe_degree_reference, n_refine_global_reference);

  deserialize_and_check_hp_conversion(4 /* fe_degree */,
                                      2 /* n_refine_global */);

  deserialize_and_check_remote_point_evaluation<4 /* fe_degree */>(
    2 /* n_refine_global */);
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
