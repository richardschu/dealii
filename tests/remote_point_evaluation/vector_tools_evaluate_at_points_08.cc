// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

// Evaluate solution vector along a line.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_point_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

#include "../tests.h"

template<int dim>
class Displacement : public Function<dim>
{
public:
	Displacement() : Function<dim>(dim), factor(100) {}

	virtual double value(dealii::Point<dim> const & point,
	         	 unsigned int const         component) const override
	{
		  if(component == 0)
		    return (factor * std::pow(point[2], 2) * std::pow(point[1] ,2));
		  else
			  return 0;
	}

private:
	const double factor;
};

template <int dim>
void test()
{
	Triangulation<dim> tria;

	double const L_F = 0.7;
	double const B_F = 1.0;
	double const H_F = 0.5;

	double const T_S = 0.05;
	double const B_S = 0.6;
	double const H_S = 0.4;

	double const L_IN = 0.6;

	unsigned int const N_CELLS_X_OUTFLOW = 1;
	unsigned int const N_CELLS_Y_LOWER   = 2;
	unsigned int const N_CELLS_Z_MIDDLE  = 2;

    std::vector<dealii::Triangulation<3>> tria_vec;
    tria_vec.resize(4);

    // middle part (in terms of z-coordinates)

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[0],
      std::vector<unsigned int>({N_CELLS_X_OUTFLOW, 1, N_CELLS_Z_MIDDLE}),
      dealii::Point<3>(L_IN + T_S, H_S - H_F / 2.0, -B_S / 2.0),
      dealii::Point<3>(L_F, H_F / 2.0, B_S / 2.0));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[1],
      std::vector<unsigned int>({N_CELLS_X_OUTFLOW, N_CELLS_Y_LOWER, N_CELLS_Z_MIDDLE}),
      dealii::Point<3>(L_IN + T_S, -H_F / 2.0, -B_S / 2.0),
      dealii::Point<3>(L_F, H_S - H_F / 2.0, B_S / 2.0));

    // negative z-part

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[2],
      std::vector<unsigned int>({N_CELLS_X_OUTFLOW, N_CELLS_Y_LOWER, 1}),
      dealii::Point<3>(L_IN + T_S, -H_F / 2.0, -B_F / 2.0),
      dealii::Point<3>(L_F, H_S - H_F / 2.0, -B_S / 2.0));

    dealii::GridGenerator::subdivided_hyper_rectangle(
      tria_vec[3],
      std::vector<unsigned int>({1, N_CELLS_Y_LOWER, 1}),
      dealii::Point<3>(L_IN, -H_F / 2.0, -B_F / 2.0),
      dealii::Point<3>(L_IN + T_S, H_S - H_F / 2.0, -B_S / 2.0));

    std::vector<dealii::Triangulation<3> const *> tria_vec_ptr(tria_vec.size());
    for(unsigned int i = 0; i < tria_vec.size(); ++i)
      tria_vec_ptr[i] = &tria_vec[i];

    dealii::GridGenerator::merge_triangulations(tria_vec_ptr, tria, 1.e-10);

	FESystem<dim> fe(FE_Q<dim>(2), dim);
	DoFHandler<dim> dof_handler(tria);
	dof_handler.distribute_dofs(fe);
	Vector<double> displacements(dof_handler.n_dofs());

	Quadrature<dim> quad(fe.base_element(0).get_unit_support_points());
	FEValues<dim> fe_values(fe, quad, update_quadrature_points | update_values);
	MappingQCache<dim> mapping(2);
	mapping.initialize(
      tria,
      [&](const typename dealii::Triangulation<dim>::cell_iterator & cell_tria)
        -> std::vector<dealii::Point<dim>> {
        std::vector<dealii::Point<dim>> grid_coordinates(quad.size());

        fe_values.reinit(cell_tria);
        // extract displacement and add to original position
        for(unsigned int i = 0; i < grid_coordinates.size(); ++i)
          {
            grid_coordinates[i] =
              fe_values.quadrature_point(i);
            grid_coordinates[i][0] += Displacement<dim>().value(grid_coordinates[i], 0);
          }
        return grid_coordinates;
      });

	DataOut<dim> data_out;
	  DataOutBase::VtkFlags flags;
	  flags.write_higher_order_cells = true;
	  data_out.set_flags(flags);
	  data_out.attach_triangulation(tria);
	  data_out.build_patches(mapping, 2, DataOut<dim>::curved_inner_cells);
	  std::ofstream stream("file.vtu");
	  data_out.write_vtu(stream);

	constexpr unsigned int n_points = 100;
	std::vector<Point<dim>> points(n_points * n_points);
	unsigned int point_idx = 0;
	for(unsigned int i = 0; i < n_points; ++i)
	{
	  for(unsigned int j = 0; j < n_points; ++j)
	  {
		points[point_idx][0] = 0.65;
		points[point_idx][1] = -0.25 + 0.4 * (1.0/(n_points - 1) * j);
		points[point_idx][2] = -0.3 + 0.6 * (1.0/(n_points - 1) * i);

        points[point_idx][0] += Displacement<dim>().value(points[point_idx], 0) + 1e-6 * (-0.5+(rand() / (double)RAND_MAX));

        point_idx += 1;
	  }
	}

	dealii::BoundingBox<dim> bounding_box(points);
	auto const               boundary_points = bounding_box.create_extended_relative(1e-3).get_boundary_points();

	dealii::Triangulation<dim> particle_dummy_tria;
	dealii::GridGenerator::hyper_rectangle(particle_dummy_tria,
	                                       boundary_points.first,
	                                       boundary_points.second);

	dealii::MappingQGeneric<dim> particle_dummy_mapping(1 /* mapping_degree */);

	{
	  dealii::Particles::ParticleHandler<dim, dim> particle_handler(particle_dummy_tria, particle_dummy_mapping);
	  particle_handler.insert_particles(points);
	  dealii::Particles::DataOut<dim, dim> particle_output;
	  particle_output.build_patches(particle_handler);
	  std::ofstream filestream("points.vtu");
	  particle_output.write_vtu(filestream);
	}

	constexpr double tol = 1e-5;
	dealii::Utilities::MPI::RemotePointEvaluation<dim> rpe (tol);
	rpe.reinit(points, tria, mapping);

	unsigned int n_points_not_found_rpe = 0;
	if(!rpe.all_points_found())
	{
	  // get vector of points not found
	  std::vector<dealii::Point<dim>> points_not_found;
	  points_not_found.reserve(points.size());

	  for(unsigned int i = 0; i < points.size(); ++i)
	  {
		if(!rpe.point_found(i))
		{
		  n_points_not_found_rpe += 1;
		  points_not_found.push_back(points[i]);
		}
	  }
		{
		  dealii::Particles::ParticleHandler<dim, dim> particle_handler(particle_dummy_tria, particle_dummy_mapping);
		  particle_handler.insert_particles(points_not_found);
		  dealii::Particles::DataOut<dim, dim> particle_output;
		  particle_output.build_patches(particle_handler);
		  std::ofstream filestream("points_not_found.vtu");
		  particle_output.write_vtu(filestream);
		}
	}
	std::cout << "rpe points not found: " << n_points_not_found_rpe << "\n";


	std::vector<bool> marked_vertices(tria.n_vertices(), true);
	dealii::Utilities::MPI::RemotePointEvaluation<dim> rpe2 (tol, false, 0, [marked_vertices](){
	    	                      return marked_vertices;
	                            });

	rpe2.reinit(points, tria, mapping);

	unsigned int n_points_not_found_rpe2 = 0;
	if(!rpe2.all_points_found())
	{
	  // get vector of points not found
	  std::vector<dealii::Point<dim>> points_not_found;
	  points_not_found.reserve(points.size());

	  for(unsigned int i = 0; i < points.size(); ++i)
	  {
		if(!rpe2.point_found(i))
		{
		  n_points_not_found_rpe2 += 1;
		  points_not_found.push_back(points[i]);
		}
	  }
		{
          dealii::Particles::ParticleHandler<dim, dim> particle_handler(particle_dummy_tria, particle_dummy_mapping);
		  particle_handler.insert_particles(points_not_found);
		  dealii::Particles::DataOut<dim, dim> particle_output;
		  particle_output.build_patches(particle_handler);
		  std::ofstream filestream("points_not_found2.vtu");
		  particle_output.write_vtu(filestream);
		}
	}
	std::cout << "rpe2 points not found: " << n_points_not_found_rpe2 << "\n";

}

int
main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deallog << std::setprecision(8) << std::fixed;

  test<3>();
}
