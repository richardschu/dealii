// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2007 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#ifndef dealii_parsed_function_h
#define dealii_parsed_function_h

#include <deal.II/base/config.h>

#include <deal.II/base/auto_derivative_function.h>
#include <deal.II/base/function_parser.h>

DEAL_II_NAMESPACE_OPEN

#ifndef DOXYGEN
class ParameterHandler;
#endif

namespace Functions
{
  /**
   * Friendly interface to the FunctionParser class. This class is meant as a
   * wrapper for the FunctionParser class. It is used in the step-34 and step-86
   * tutorial programs.
   *
   * It provides two methods to declare and parse a ParameterHandler object
   * and creates the Function object declared in the parameter file. This
   * class is derived from the AutoDerivativeFunction class, so you don't need
   * to specify derivatives. An example of usage of this class is as follows:
   *
   *   @code
   *   // A parameter handler
   *   ParameterHandler prm;
   *
   *   // Declare a section for the function we need
   *   prm.enter_subsection("My vector function");
   *   ParsedFunction<dim>::declare_parameters(prm, dim);
   *   prm.leave_subsection();
   *
   *   // Create a ParsedFunction
   *   ParsedFunction<dim> my_vector_function(dim);
   *
   *   // Parse an input file.
   *   prm.parse_input(some_input_file);
   *
   *   // Initialize the ParsedFunction object with the given file
   *   prm.enter_subsection("My vector function");
   *   my_vector_function.parse_parameters(prm);
   *   prm.leave_subsection();
   *
   *   @endcode
   *
   * And here is an example of how the input parameter could look like (see
   * the documentation of the FunctionParser class for a detailed description
   * of the syntax of the function definition):
   *
   *   @code
   *
   *   # A test two dimensional vector function, depending on time
   *   subsection My vector function
   *   set Function constants  = kappa=.1, lambda=2.
   *   set Function expression = if(y>.5, kappa*x*(1-x),0); t^2*cos(lambda*pi*x)
   *   set Variable names      = x,y,t
   *   end
   *
   *   @endcode
   *
   * @ingroup functions
   */
  template <int dim>
  class ParsedFunction : public AutoDerivativeFunction<dim>
  {
  public:
    /**
     * Construct a vector function. The vector function which is generated has
     * @p n_components components (defaults to 1). The parameter @p h is used
     * to initialize the AutoDerivativeFunction class from which this class is
     * derived.
     */
    ParsedFunction(const unsigned int n_components = 1, const double h = 1e-8);

    /**
     * Declare parameters needed by this class. The parameter @p
     * n_components is used to generate the right code according to the number
     * of components of the function that will parse this ParameterHandler. If
     * the number of components which is parsed does not match the number of
     * components of this object, an assertion is thrown and the program is
     * aborted. The additional parameter @p expr is used to declare the default
     * expression used by the function. The default behavior for this class is
     * to declare the following entries:
     *
     *  @code
     *
     *  set Function constants  =
     *  set Function expression = 0
     *  set Variable names      = x,y,t
     *
     *  @endcode
     */
    static void
    declare_parameters(ParameterHandler  &prm,
                       const unsigned int n_components = 1,
                       const std::string &input_expr   = "");

    /**
     * Parse parameters needed by this class.  If the number of components
     * which is parsed does not match the number of components of this object,
     * an assertion is thrown and the program is aborted.  In order for the
     * class to function properly, we follow the same conventions declared in
     * the FunctionParser class (look there for a detailed description of the
     * syntax for function declarations).
     *
     * The three variables that can be parsed from a parameter file are the
     * following:
     *
     *  @code
     *
     *  set Function constants  =
     *  set Function expression =
     *  set Variable names      =
     *
     *  @endcode
     *
     * %Function constants is a collection of pairs in the form name=value,
     * separated by commas, for example:
     *
     *  @code
     *
     *  set Function constants = lambda=1., alpha=2., gamma=3.
     *
     *  @endcode
     *
     * These constants can be used in the declaration of the function
     * expression, which follows the convention of the FunctionParser class.
     * In order to specify vector functions, semicolons have to be used to
     * separate the different components, e.g.:
     *
     *  @code
     *
     *  set Function expression = cos(pi*x); cos(pi*y)
     *
     *  @endcode
     *
     * The variable names entry can be used to customize the name of the
     * variables used in the Function. It defaults to
     *
     *  @code
     *
     *  set Variable names      = x,t
     *
     *  @endcode
     *
     * for one dimensional problems,
     *
     *  @code
     *
     *  set Variable names      = x,y,t
     *
     *  @endcode
     *
     * for two dimensional problems and
     *
     *  @code
     *
     *  set Variable names      = x,y,z,t
     *
     *  @endcode
     *
     * for three dimensional problems.
     *
     * The time variable can be set according to specifications in the
     * FunctionTime base class.
     */
    void
    parse_parameters(ParameterHandler &prm);

    /**
     * Return all components of a vector-valued function at the given point @p
     * p.
     */
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override;

    /**
     * Return the value of the function at the given point. Unless there is
     * only one component (i.e. the function is scalar), you should state the
     * component you want to have evaluated; it defaults to zero, i.e. the
     * first component.
     */
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    /**
     * Set the time to a specific value for time-dependent functions.
     *
     * We need to overwrite this to set the time also in the accessor
     * FunctionParser<dim>.
     */
    virtual void
    set_time(const double newtime) override;

  private:
    /**
     * The object with which we do computations.
     */
    FunctionParser<dim> function_object;
  };
} // namespace Functions

DEAL_II_NAMESPACE_CLOSE

#endif
