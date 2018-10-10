/*

Copyright (C) 2018-2018 Gene Harvey

This file is part of Octave.

Octave is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Octave is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Octave; see the file COPYING.  If not, see
<https://www.gnu.org/licenses/>.

*/

#include <octave/builtin-defun-decls.h>
#include <octave/oct-string.h>
#include <octave/oct.h>

static void
print_error (const std::string& tag, const octave_value& msg)
{
  octave_value_list args (2);
  args(0) = octave_value (tag);
  args(1) = msg;
  Ferror (args);
}

static void
print_error (const char* tag, const char* msg)
{
  error_with_id (tag, msg);
}

static void
print_error (const char* tag, const std::string& msg)
{
  error_with_id (tag, msg.c_str ());
}

static void
print_error (const std::string& tag, const std::string& msg)
{
  error_with_id (tag.c_str (), msg.c_str ());
}

static void
print_error (const std::string& msg)
{
  error (msg.c_str ());
}

static octave_value
as_vector (const octave_value& ov)
{
  return ov.reshape (dim_vector (ov.numel (), 1));
}

static bool
has_any (const octave_value& ov)
{
  return ((ov.any ()).bool_matrix_value ())(0);
}

static bool
has_all (const octave_value& ov)
{
  return ((ov.all ()).bool_matrix_value ())(0);
}

static bool
is_valid_idx (const octave_value& idx)
{
  return idx.isnumeric () && idx.numel () == 1 && idx.scalar_value () > 0
         && idx.scalar_value () == (idx.fix ()).scalar_value ();
}

static bool
chk_class (const octave_value& ov_A, Array<std::string> cls)
{
  octave_idx_type i;

  std::string A_class = ov_A.class_name ();

  for (i = 0; i < cls.numel (); i++)
    {
      if (A_class == cls(i) || (cls(i) == "float" && ov_A.isfloat ())
          || (cls(i) == "integer" && ov_A.isinteger ())
          || (cls(i) == "numeric" && ov_A.isnumeric ())
          || ov_A.is_instance_of (cls(i)))
        {
          return true;
        }
    }
  return false;
}

static void
cls_error (const std::string& err_ini, Array<std::string> cls,
           const std::string& A_class)
{
  size_t                          i;
  octave_idx_type                 j;
  std::string                     err_str;
  std::set<std::string>           classes;
  std::set<std::string>::iterator classes_iter;

  for (j = 0; j < cls.numel (); j++)
    {
      if (cls(j) == "integer")
        {
          classes.insert ("int8");
          classes.insert ("int16");
          classes.insert ("int32");
          classes.insert ("int64");
          classes.insert ("uint8");
          classes.insert ("uint16");
          classes.insert ("uint32");
          classes.insert ("uint64");
        }
      else if (cls(j) == "float")
        {
          classes.insert ("single");
          classes.insert ("double");
        }
      else if (cls(j) == "numeric")
        {
          classes.insert ("int8");
          classes.insert ("int16");
          classes.insert ("int32");
          classes.insert ("int64");
          classes.insert ("uint8");
          classes.insert ("uint16");
          classes.insert ("uint32");
          classes.insert ("uint64");
          classes.insert ("single");
          classes.insert ("double");
        }
      else
        {
          classes.insert (cls(j));
        }
    }

  err_str = err_ini + " must be of class:\n\n ";

  classes_iter = classes.begin ();
  for (i = 0; i < classes.size (); i++, classes_iter++)
    {
      err_str += " " + *classes_iter;
    }
  err_str += "\n\nbut was of class " + A_class;

  print_error ("Octave:invalid-type", err_str);
}

static void
err_attr (const std::string& err_id, const std::string& err_ini, const std::string& attr_name)
{
  print_error (err_id, err_ini + " must be " + attr_name);
}

static void
err_attr (const std::string& attr_name)
{
  print_error ("Octave:invalid-input-arg",
               "validateattributes: unknown ATTRIBUTE " + attr_name);
}

static bool
chk_size (const dim_vector& A_dims, octave_idx_type A_ndims, const octave_value& attr_val)
{

  int i;
  Matrix dims_as_mat;
  boolMatrix attr_isnan;

  octave_idx_type attr_numel = attr_val.numel ();

  if (attr_numel < A_ndims)
    return false;

  attr_isnan = (attr_val.isnan ()).bool_matrix_value ();

  dims_as_mat = Matrix (attr_numel, 1);
  for (i = 0; i < attr_numel; i++)
    {
      if(i < A_ndims)
        {
          dims_as_mat(i) = A_dims(i);
        }
      else if (! attr_isnan(i))
        {
          return false;
        }
      else
        {
          dims_as_mat(i) = 0;
        }
    }
  return has_all (op_el_or(dims_as_mat == attr_val, attr_isnan));
}

static void
err_size (const octave_value& ov_A, const octave_value& attr_val,
          const std::string& err_ini)
{
  octave_value_list args (3);
  args (0) = octave_value ("%dx");
  args (1) = Fsize (ov_A)(0);
  std::string A_dims_str = Fsprintf (args.slice (0, 2))(0).string_value ();
  A_dims_str = A_dims_str.substr (0, A_dims_str.length () - 1);

  args (0) = octave_value ("%ix");
  args (1) = attr_val;

  args (0) = Fsprintf (args.slice (0, 2))(0);
  args (1) = octave_value ("NaN");
  args (2) = octave_value ("N");
  std::string attr_dims_str = (Fstrrep(args)(0).string_value ());
  attr_dims_str = attr_dims_str.substr (0, attr_dims_str.length () - 1);

  print_error ("Octave:incorrect-size", err_ini + " must be of size "
               += attr_dims_str += " but was " + A_dims_str);
}

template<typename O>
static bool
chk_monotone (const octave_value& A_vec, O op)
{
  octave_value A_diff = Fdiff (A_vec)(0);
  bool A_isnan = has_any (A_vec.isnan ());
  bool A_ismono = has_all (op (A_diff, 0)); // ex. greater than => check all A_diff > 0
  return ! A_isnan && A_ismono;
}

static bool
chk_even (const octave_value& A_vec)
{
  octave_value_list args (2);
  args(0) = A_vec;
  args(1) = octave_value (2);
  return has_all (Frem (args)(0) == 0);
}

static bool
chk_odd (const octave_value& A_vec)
{
  octave_value_list args (2);
  args(0) = A_vec;
  args(1) = octave_value (2);
  return has_all (Fmod (args)(0) == 1);
}

template<typename O>
static bool
chk_compare (const octave_value& A_vec, const octave_value& attr_val, O op)
{
  return has_all (op (A_vec, attr_val).all ());
}

static void
err_compare (const std::string& tag, const std::string& cmp_str, const std::string& err_ini,
             const octave_value& attr_val)
{
  octave_value_list args (3);
  args(0) = octave_value ("%s must be " + cmp_str + " %f");
  args(1) = octave_value (err_ini);
  args(2) = octave_value (attr_val);
  print_error (tag, Fsprintf (args)(0));
}

static bool
chk_diag (const octave_value& ov_A)
{
  if (ov_A.is_diag_matrix ())
    return true;
  else if ((ov_A.isnumeric () || ov_A.islogical ()) && ov_A.ndims () == 2)
    {
      octave_value_list dim_vecs = Ffind (ov_A, 2);
      return has_all (dim_vecs(0) == dim_vecs(1));
    }
  else
    return false;
}

static void
chk_attributes (const octave_value& ov_A, const Cell& attr, const std::string& err_ini)
{

  size_t          len;
  std::string     name;
  octave_value    attr_val;
  octave_idx_type i;

  dim_vector      A_dims  = ov_A.dims ();
  octave_value    A_vec   = as_vector(ov_A);
  octave_idx_type A_ndims = ov_A.ndims ();

  i = 0;
  while (i < attr.numel ())
    {
      name = attr (i++).string_value ();
      len = name.length ();

      if (len < 1)
        err_attr (name);

      switch (std::tolower (name[0]))
        {
          case '2': // 2d
            {
              if (len == 2 && std::tolower (name[1]) == 'd')
                {
                  if (A_ndims != 2)
                    err_attr ("Octave:expected-2d", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case '3': // 3d
            {
              if (len == 2 && std::tolower (name[1]) == 'd')
                {
                  if (A_ndims > 3)
                    err_attr ("Octave:expected-3d", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'c': // column
            {
              if (octave::string::strcmpi (name, "column"))
                {
                  if (A_ndims != 2 || A_dims(1) != 1)
                    err_attr ("Octave:expected-column", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'r': // row, real
            {
              if (octave::string::strcmpi (name, "row")) // row
                {
                  if (A_ndims != 2 || A_dims(0) != 1)
                    err_attr ("Octave:expected-row", err_ini, name);
                }
              else if (octave::string::strcmpi (name, "real")) // real
                {
                  if (! ov_A.isreal ())
                    err_attr ("Octave:expected-real", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 's': // scalar, square, size,
            {
              if (octave::string::strcmpi (name, "scalar")) // scalar
                {
                  if (ov_A.numel () != 1)
                    err_attr ("Octave:expected-scalar", err_ini, name);
                }
              else if (octave::string::strcmpi (name, "square")) // square
                {
                  if (A_ndims != 2 || A_dims(0) != A_dims(1))
                    err_attr ("Octave:expected-square", err_ini, name);
                }
              else if (octave::string::strcmpi (name, "size")) // size
                {
                  if (i >= attr.numel ())
                    print_error ("Incorrect number of attribute cell arguments");
                  attr_val = attr (i++);
                  if (! chk_size (A_dims, A_ndims, as_vector(attr_val)))
                    {
                      err_size (ov_A, attr_val, err_ini);
                    }
                }
              else
                err_attr (name);
              break;
            }
          case 'v': // vector
            {
              if (octave::string::strcmpi (name, "vector"))
                {
                  if (A_ndims != 2 || (A_dims(0) != 1 && A_dims(1) != 1))
                    err_attr ("Octave:expected-vector", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'd': // diag, decreasing
            {
              if (octave::string::strcmpi (name, "diag")) // diag
                {
                  if (! chk_diag (ov_A))
                    err_attr ("Octave:expected-diag", err_ini, name);
                }
              else if (octave::string::strcmpi (name,
                                                "decreasing")) // decreasing
                {
                  if (! chk_monotone (A_vec, op_lt))
                    err_attr ("Octave:expected-decreasing", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'n': // nonempty, nonsparse, nonnan, nonnegative, nonzero,
            // nondecreasing, nonincreasing, numel, ncols, nrows, ndims
            {

              if (len < 2)
                err_attr (name);

              switch (std::tolower (name[1]))
                {
                  case 'o': // nonempty, nonsparse, nonnan, nonnegative,
                    // nonzero, nondecreasing, nonincreasing
                    {

                      if (len < 4)
                        err_attr (name);

                      switch (std::tolower (name[3]))
                        {
                          case 'e':  // nonempty
                            {
                              if (octave::string::strcmpi (name,
                                                           "nonempty")) // nonempty
                                {
                                  if (ov_A.isempty ())
                                    err_attr ("Octave:expected-nonempty",
                                              err_ini, name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          case 's': // nonsparse
                            {
                              if (octave::string::strcmpi (
                                  name, "nonsparse")) // nonsparse
                                {
                                  if (ov_A.issparse ())
                                    err_attr ("Octave:expected-nonsparse",
                                              err_ini, name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          case 'n': // nonnan, nonnegative
                            {
                              if (octave::string::strcmpi (name,
                                                           "nonnan")) // nonnan
                                {
                                  if (! ov_A.isinteger ()
                                      && has_any(A_vec.isnan ()))
                                    err_attr ("Octave:expected-nonnan", err_ini,
                                              name);
                                }
                              else if (octave::string::strcmpi (
                                  name, "nonnegative")) // nonnegative
                                {
                                  if (has_any(A_vec < 0))
                                    err_attr ("Octave:expected-nonnegative",
                                              err_ini, name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          case 'z': // nonzero
                            {
                              if (octave::string::strcmpi (name,
                                                           "nonzero")) // nonzero
                                {
                                  if (has_any(A_vec == 0))
                                    err_attr ("Octave:expected-nonzero", err_ini,
                                              name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          case 'd': // nondecreasing
                            {
                              if (octave::string::strcmpi (
                                  name, "nondecreasing")) // nondecreasing
                                {
                                  if (! chk_monotone (A_vec, op_ge))
                                    err_attr ("Octave:expected-nondecreasing",
                                              err_ini, name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          case 'i': // nonincreasing
                            {
                              if (octave::string::strcmpi (
                                  name, "nonincreasing")) // nonincreasing
                                {
                                  if (! chk_monotone (A_vec, op_le))
                                    err_attr ("Octave:expected-nonincreasing",
                                              err_ini, name);
                                }
                              else
                                err_attr (name);
                              break;
                            }
                          default:
                            err_attr (name);
                        }
                      break;
                    }
                  case 'u': // numel
                    {
                      if (octave::string::strcmpi (name, "numel"))
                        {
                          if (i >= attr.numel ())
                            print_error ("Incorrect number of attribute cell "
                                         "arguments");
                          attr_val = attr (i++);
                          if (ov_A.numel () != attr_val.idx_type_value ())
                            {
                              print_error (
                                  "Octave:incorrect-numel",
                                  err_ini + " must have "
                                  + std::to_string (
                                      attr_val.idx_type_value ())
                                  + " elements");
                            }
                        }
                      else
                        err_attr (name);
                      break;
                    }
                  case 'c': // ncols
                    {
                      if (octave::string::strcmpi (name, "ncols"))
                        {
                          if (i >= attr.numel ())
                            print_error ("Incorrect number of attribute cell "
                                         "arguments");
                          attr_val = attr (i++);
                          if (A_ndims < 2
                              || A_dims(1) != attr_val.idx_type_value ())
                            {
                              print_error (
                                  "Octave:incorrect-numcols",
                                  err_ini + " must have "
                                  + std::to_string (
                                      attr_val.idx_type_value ())
                                  + " columns");
                            }
                        }
                      else
                        err_attr (name);
                      break;
                    }
                  case 'r': // nrows
                    {
                      if (octave::string::strcmpi (name, "nrows"))
                        {
                          if (i >= attr.numel ())
                            print_error ("Incorrect number of attribute cell "
                                         "arguments");
                          attr_val = attr (i++);
                          if (A_ndims < 1
                              || A_dims(0) != attr_val.idx_type_value ())
                            {
                              print_error (
                                  "Octave:incorrect-numrows",
                                  err_ini + " must have "
                                  + std::to_string (
                                      attr_val.idx_type_value ())
                                  + " rows");
                            }
                        }
                      else
                        err_attr (name);
                      break;
                    }
                  case 'd': // ndims
                    {
                      if (octave::string::strcmpi (name, "ndims"))
                        {
                          if (i >= attr.numel ())
                            print_error ("Incorrect number of attribute cell "
                                         "arguments");
                          attr_val = attr (i++);
                          if (A_ndims != attr_val.idx_type_value ())
                            {
                              print_error ("Octave:incorrect-numdims", err_ini
                                           + " must have " + std::to_string (
                                           attr_val.idx_type_value ())
                                           + " dimensions");
                            }
                        }
                      else
                        err_attr (name);
                      break;
                    }
                  default:
                    err_attr (name);
                }
              break;
            }
          case 'b': // binary
            {
              if (octave::string::strcmpi (name, "binary"))
                {
                  if (! ov_A.islogical ()
                      && (has_any (op_el_and ((A_vec != 1), (A_vec != 0)))))
                    err_attr ("Octave:expected-binary", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'e': // even
            {
              if (octave::string::strcmpi (name, "even"))
                {
                  if (! chk_even (A_vec))
                    err_attr ("Octave:expected-even", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'o': // odd
            {
              if (octave::string::strcmpi (name, "odd"))
                {
                  if (! chk_odd (A_vec))
                    err_attr ("Octave:expected-odd", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'i': // integer, increasing
            {
              if (octave::string::strcmpi (name, "integer")) // integer
                {
                  if (! ov_A.isinteger ()
                      && (has_any (A_vec.ceil () != A_vec)))
                    err_attr ("Octave:expected-integer", err_ini, name);
                }
              else if (octave::string::strcmpi (name,
                                                "increasing")) // increasing
                {
                  if (! chk_monotone (A_vec, op_gt))
                    err_attr ("Octave:expected-increasing", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'f': // finite
            {
              if (octave::string::strcmpi (name, "finite"))
                {
                  if (! ov_A.isinteger ()
                      && ! has_all (A_vec.isfinite ()))
                    err_attr ("Octave:expected-finite", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case 'p': // positive
            {
              if (octave::string::strcmpi (name, "positive"))
                {
                  if (has_any (A_vec <= 0))
                    err_attr ("Octave:expected-positive", err_ini, name);
                }
              else
                err_attr (name);
              break;
            }
          case '>': // >, >=
            {
              if (len == 1) // >
                {
                  if (i >= attr.numel ())
                    print_error ("Incorrect number of attribute cell arguments");
                  attr_val = attr (i++);
                  if (! chk_compare (A_vec, attr_val, op_gt))
                    err_compare ("Octave:expected-greater", "greater than",
                                 err_ini, attr_val);
                }
              else if (len == 2 && std::tolower (name[1]) == '=') // >=
                {
                  if (i >= attr.numel ())
                    print_error ("Incorrect number of attribute cell arguments");
                  attr_val = attr (i++);
                  if (! chk_compare (A_vec, attr_val, op_ge))
                    err_compare ("Octave:expected-greater-equal",
                                 "greater than or equal to", err_ini,
                                 attr_val);
                }
              else
                err_attr (name);
              break;
            }
          case '<': // <, <=
            {
              if (len == 1) // <
                {
                  if (i >= attr.numel ())
                    print_error ("Incorrect number of attribute cell arguments");
                  attr_val = attr (i++);
                  if (! chk_compare (A_vec, attr_val, op_lt))
                    err_compare ("Octave:expected-less", "less than",
                                 err_ini, attr_val);
                }
              else if (len == 2 && std::tolower (name[1]) == '=') // <=
                {
                  if (i >= attr.numel ())
                    print_error ("Incorrect number of attribute cell arguments");
                  attr_val = attr (i++);
                  if (! chk_compare (A_vec, attr_val, op_le))
                    err_compare ("Octave:expected-less-equal",
                                 "less than or equal to", err_ini, attr_val);
                }
              else
                err_attr (name);
              break;
            }
          default:
            err_attr (name);
        }
    }
}

DEFUN_DLD (validateattributes, args, nargout, "-*- texinfo -*-\n\
@deftypefn  {} {} validateattributes (@var{A}, @var{classes}, @var{attributes})\n\
@deftypefnx {} {} validateattributes (@var{A}, @var{classes}, @var{attributes}, @var{arg_idx})\n\
@deftypefnx {} {} validateattributes (@var{A}, @var{classes}, @var{attributes}, @var{func_name})\n\
@deftypefnx {} {} validateattributes (@var{A}, @var{classes}, @var{attributes}, @var{func_name}, @var{arg_name})\n\
@deftypefnx {} {} validateattributes (@var{A}, @var{classes}, @var{attributes}, @var{func_name}, @var{arg_name}, @var{arg_idx})\n\
Check validity of input argument.\n\
\n\
Confirms that the argument @var{A} is valid by belonging to one of\n\
@var{classes}, and holding all of the @var{attributes}.  If it does not,\n\
an error is thrown, with a message formatted accordingly.  The error\n\
message can be made further complete by the function name @var{fun_name},\n\
the argument name @var{arg_name}, and its position in the input\n\
@var{arg_idx}.\n\
\n\
@var{classes} must be a cell array of strings (an empty cell array is\n\
allowed) with the name of classes (remember that a class name is case\n\
sensitive).  In addition to the class name, the following categories\n\
names are also valid:\n\
\n\
@table @asis\n\
@item @qcode{\"float\"}\n\
Floating point value comprising classes @qcode{\"double\"} and\n\
@qcode{\"single\"}.\n\
\n\
@item @qcode{\"integer\"}\n\
Integer value comprising classes (u)int8, (u)int16, (u)int32, (u)int64.\n\
\n\
@item @qcode{\"numeric\"}\n\
Numeric value comprising either a floating point or integer value.\n\
\n\
@end table\n\
\n\
@var{attributes} must be a cell array with names of checks for @var{A}.\n\
Some of them require an additional value to be supplied right after the\n\
name (see details for each below).\n\
\n\
@table @asis\n\
@item @qcode{\"<=\"}\n\
All values are less than or equal to the following value in\n\
@var{attributes}.\n\
\n\
@item @qcode{\"<\"}\n\
All values are less than the following value in @var{attributes}.\n\
\n\
@item @qcode{\">=\"}\n\
All values are greater than or equal to the following value in\n\
@var{attributes}.\n\
\n\
@item @qcode{\">\"}\n\
All values are greater than the following value in @var{attributes}.\n\
\n\
@item @qcode{\"2d\"}\n\
A 2-dimensional matrix.  Note that vectors and empty matrices have\n\
2 dimensions, one of them being of length 1, or both length 0.\n\
\n\
@item @qcode{\"3d\"}\n\
Has no more than 3 dimensions.  A 2-dimensional matrix is a 3-D matrix\n\
whose 3rd dimension is of length 1.\n\
\n\
@item @qcode{\"binary\"}\n\
All values are either 1 or 0.\n\
\n\
@item @qcode{\"column\"}\n\
Values are arranged in a single column.\n\
\n\
@item @qcode{\"decreasing\"}\n\
No value is @var{NaN}, and each is less than the preceding one.\n\
\n\
@item @qcode{\"diag\"}\n\
Value is a diagonal matrix.\n\
\n\
@item @qcode{\"even\"}\n\
All values are even numbers.\n\
\n\
@item @qcode{\"finite\"}\n\
All values are finite.\n\
\n\
@item @qcode{\"increasing\"}\n\
No value is @var{NaN}, and each is greater than the preceding one.\n\
\n\
@item @qcode{\"integer\"}\n\
All values are integer.  This is different than using @code{isinteger}\n\
which only checks its an integer type.  This checks that each value in\n\
@var{A} is an integer value, i.e., it has no decimal part.\n\
\n\
@item @qcode{\"ncols\"}\n\
Has exactly as many columns as the next value in @var{attributes}.\n\
\n\
@item @qcode{\"ndims\"}\n\
Has exactly as many dimensions as the next value in @var{attributes}.\n\
\n\
@item @qcode{\"nondecreasing\"}\n\
No value is @var{NaN}, and each is greater than or equal to the preceding\n\
one.\n\
\n\
@item @qcode{\"nonempty\"}\n\
It is not empty.\n\
\n\
@item @qcode{\"nonincreasing\"}\n\
No value is @var{NaN}, and each is less than or equal to the preceding one.\n\
\n\
@item @qcode{\"nonnan\"}\n\
No value is a @code{NaN}.\n\
\n\
@item @nospell{@qcode{\"nonnegative\"}}\n\
All values are non negative.\n\
\n\
@item @qcode{\"nonsparse\"}\n\
It is not a sparse matrix.\n\
\n\
@item @qcode{\"nonzero\"}\n\
No value is zero.\n\
\n\
@item @qcode{\"nrows\"}\n\
Has exactly as many rows as the next value in @var{attributes}.\n\
\n\
@item @qcode{\"numel\"}\n\
Has exactly as many elements as the next value in @var{attributes}.\n\
\n\
@item @qcode{\"odd\"}\n\
All values are odd numbers.\n\
\n\
@item @qcode{\"positive\"}\n\
All values are positive.\n\
\n\
@item @qcode{\"real\"}\n\
It is a non-complex matrix.\n\
\n\
@item @qcode{\"row\"}\n\
Values are arranged in a single row.\n\
\n\
@item @qcode{\"scalar\"}\n\
It is a scalar.\n\
\n\
@item @qcode{\"size\"}\n\
Its size has length equal to the values of the next in @var{attributes}.\n\
The next value must is an array with the length for each dimension.  To\n\
ignore the check for a certain dimension, the value of @code{NaN} can be\n\
used.\n\
\n\
@item @qcode{\"square\"}\n\
Is a square matrix.\n\
\n\
@item @qcode{\"vector\"}\n\
Values are arranged in a single vector (column or vector).\n\
\n\
@end table\n\
\n\
@seealso{isa, validatestring, inputParser}\n\
@end deftypefn ")
{

  octave_value       ov_A;
  octave_value       ov_cls;
  octave_value       ov_attr;

  std::string        A_class;
  Array<std::string> cls;
  Cell               attr;

  std::string        err_ini;
  std::string        func_name;

  std::string        var_name = "input";
  octave_idx_type    nargin   = args.length ();

  if (nargin < 3 || nargin > 6)
    print_usage ();

  ov_A    = args(0);
  ov_cls  = args(1);
  ov_attr = args(2);

  if (! ov_cls.iscellstr ())
    {
      print_error (
          "Octave:invalid-type",
          "validateattributes: CLASSES must be a cell array of strings");
    }
  else if (! ov_attr.iscell ())
    {
      print_error ("Octave:invalid-type",
                   "validateattributes: ATTRIBUTES must be a cell array");
    }

  cls  = ov_cls.cellstr_value ();
  attr = ov_attr.cell_value ();

  if (nargin > 3)
    {
      if (args(3).is_string ())
        {
          func_name = args(3).string_value () + ": ";
        }
      else if (nargin == 4 && is_valid_idx (args(3)))
        {
          var_name = "input " + std::to_string (args(3).idx_type_value ());
        }
      else
        {
          print_error ("Octave:invalid-input-arg",
                       "validateattributes: 4th input argument must be "
                       "ARG_IDX or FUNC_NAME");
        }

      if (nargin > 4)
        {
          if (! args(4).is_string ())
            {
              print_error ("Octave:invalid-type",
                           "validateattributes: VAR_NAME must be a string");
            }
          var_name = args(4).string_value ();

          if (nargin > 5)
            {
              if (! is_valid_idx (args(5)))
                {
                  print_error ("Octave:invalid-input-arg",
                               "validateattributes: ARG_IDX must be a "
                               "positive integer");
                }
              var_name += " (argument #"
                          + std::to_string (args(5).idx_type_value ()) + ")";
            }
        }
    }

  err_ini = func_name + var_name;

  if (! cls.isempty () && ! chk_class (ov_A, cls))
    {
      cls_error (err_ini, cls, ov_A.class_name ());
    }

  chk_attributes (ov_A, attr, err_ini);

  return octave_value_list ();
}

/*
%!error <double> validateattributes (rand (5), {"uint8"}, {})
%!error <single> validateattributes (uint8 (rand (5)), {"float"}, {})
%!error <2d> validateattributes (rand (5, 5, 5), {}, {"2d"})
%!error <3d> validateattributes (rand (5, 5, 5, 7), {}, {"3d"})
%!error <column> validateattributes (rand (5, 5), {}, {"column"})
%!error <column> validateattributes (rand (1, 5), {}, {"column"})
%!error <row> validateattributes (rand (5, 5), {}, {"row"})
%!error <row> validateattributes (rand (5, 1), {}, {"row"})
%!error <scalar> validateattributes (rand (1, 5), {}, {"scalar"})
%!error <vector> validateattributes (rand (5), {}, {"vector"})
%!error <square> validateattributes (rand (5, 6), {}, {"square"})
%!error <nonempty> validateattributes ([], {}, {"nonempty"})
%!error <nonsparse> validateattributes (sparse(rand(5)), {}, {"nonsparse"})
%!error <binary> validateattributes ("text", {}, {"binary"})
%!error <binary> validateattributes ([0 1 0 3 0], {}, {"binary"})
%!error <even> validateattributes ([2 3 6 8], {}, {"even"})
%!error <even> validateattributes ([2 NaN], {}, {"even"})
%!error <odd> validateattributes ([3 4 7 5], {}, {"odd"})
%!error <odd> validateattributes ([5 NaN], {}, {"odd"})
%!error <integer> validateattributes ([5 5.2 5.7], {}, {"integer"})
%!error <real> validateattributes ([5i 8 9], {}, {"real"})
%!error <finite> validateattributes ([5i Inf 8], {}, {"finite"})
%!error <nonnan> validateattributes ([NaN Inf 8], {}, {"nonnan"})
%!error <nonnegative> validateattributes ([7 8 -9], {}, {"nonnegative"})
%!error <nonzero> validateattributes ([7 8 0], {}, {"nonzero"})
%!error <positive> validateattributes ([7 0 8], {}, {"positive"})
%!error <decreasing> validateattributes ([7 8 4 3 -5], {}, {"decreasing"})
%!error <decreasing> validateattributes ([7 NaN 4 3 -5], {}, {"decreasing"})
%!error <increasing> validateattributes ([7 8 4 9 20], {}, {"increasing"})
%!error <increasing> validateattributes ([7 8 NaN 9 20], {}, {"increasing"})
%!error <nonincreasing> validateattributes ([7 8 4 9 20], {}, {"nonincreasing"})
%!error <nonincreasing> validateattributes ([7 8 NaN 9 20], {}, {"nonincreasing"})
%!error <nondecreasing> validateattributes ([7 8 4 3 -5], {}, {"nondecreasing"})
%!error <nondecreasing> validateattributes ([7 NaN 4 3 -5], {}, {"nondecreasing"})
%!error <size> validateattributes (ones (5, 3, 6), {}, {"size", [5 4 7]})
%!error <size> validateattributes (ones (5, 3, 6), {}, {"size", [5 NaN 7]})
%!error <size> validateattributes (ones (5, 3, 6), {}, {"size", [5 3 6 2]})
%!error <elements> validateattributes (ones (6, 3), {}, {"numel", 12})
%!error <columns> validateattributes (ones (6, 2), {}, {"ncols", 3})
%!error <rows> validateattributes (ones (6, 2), {}, {"nrows", 3})
%!error <dimensions> validateattributes (ones (6, 2, 6, 3), {}, {"ndims", 3})
%!error <greater than> validateattributes ([6 7 8 5], {}, {">", 5})
%!error <greater than> validateattributes ([6 7 8 5], {}, {">=", 6})
%!error <less than> validateattributes ([6 7 8 5], {}, {"<", 8})
%!error <less than> validateattributes ([6 7 8 5], {}, {"<=", 7})
%!error <diag> validateattributes ([0 0 0; 0 0 0; 1 0 0], {}, {"diag"})
%!error <diag> validateattributes (repmat (eye (3), [1 1 3]), {}, {"diag"})

%!test validateattributes (rand (5), {"numeric"}, {});
%!test validateattributes (rand (5), {"float"}, {});
%!test validateattributes (rand (5), {"double"}, {});
%!test validateattributes ("text", {"char"}, {});
%!test validateattributes (rand (5), {}, {"2d"});
%!test validateattributes (rand (5), {}, {"3d"});
%!test validateattributes (rand (5, 5, 5), {}, {"3d"});
%!test validateattributes (rand (5, 1), {}, {"column"});
%!test validateattributes (rand (1, 5), {}, {"row"});
%!test validateattributes ("a", {}, {"scalar"});
%!test validateattributes (5, {}, {"scalar"});
%!test validateattributes (rand (1, 5), {}, {"vector"});
%!test validateattributes (rand (5, 1), {}, {"vector"});
%!test validateattributes (rand (5), {}, {"square"});
%!test validateattributes (rand (5), {}, {"nonempty"});
%!test validateattributes (rand (5), {}, {"nonsparse"});
%!test validateattributes ([0 1 0 1 0], {}, {"binary"});
%!test validateattributes (rand (5) > 0.5, {}, {"binary"});
%!test validateattributes ([8 4 0 6], {}, {"even"});
%!test validateattributes ([-1 3 5], {}, {"odd"});
%!test validateattributes ([8 4 0 6], {}, {"real"});
%!test validateattributes ([8 4i 0 6], {}, {"finite"});
%!test validateattributes (uint8 ([8 4]), {}, {"finite"});
%!test validateattributes ([8 Inf], {}, {"nonnan"});
%!test validateattributes ([0 7 4], {}, {"nonnegative"});
%!test validateattributes ([-8 7 4], {}, {"nonzero"});
%!test validateattributes ([8 7 4], {}, {"positive"});
%!test validateattributes ([8 7 4 -5], {}, {"decreasing"});
%!test validateattributes ([-8 -7 4 5], {}, {"increasing"});
%!test validateattributes ([8 4 4 -5], {}, {"nonincreasing"});
%!test validateattributes ([-8 -8 4 5], {}, {"nondecreasing"});
%!test validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 6 7 2]});
%!test validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 NaN 7 2]});
%!test validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 6 NaN 2 NaN]});
%!test validateattributes (rand (6, 2), {}, {"numel", 12});
%!test validateattributes (rand (6, 2), {}, {"ncols", 2});
%!test validateattributes (rand (6, 2), {}, {"nrows", 6});
%!test validateattributes (rand (6, 2, 4, 5), {}, {"ndims", 4});
%!test validateattributes ([4 5 6 7], {}, {">", 3});
%!test validateattributes ([4 5 6 7], {}, {">=", 4});
%!test validateattributes ([4 5 6 7], {}, {"<", 8});
%!test validateattributes ([4 5 6 7], {}, {"<=", 7});
%!test validateattributes (eye (3), {}, {"diag"});
%!test validateattributes ([1 0 0; 0 1 0; 0 0 1], {}, {"diag"});
%!test validateattributes (zeros (3), {}, {"diag"});
%!test validateattributes ([0 1 0 1], {"double", "uint8"}, {"binary", "size", [NaN 4], "nonnan"});

%!test
%! try validateattributes (ones(1,2,3), {"numeric"}, {"2d"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-2d");
%! end_try_catch

%!test
%! try validateattributes (ones(1,2,3,4), {"numeric"}, {"3d"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-3d");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"column"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-column");
%! end_try_catch

%!test
%! try validateattributes ([1 2].', {"numeric"}, {"row"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-row");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"scalar"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-scalar");
%! end_try_catch

%!test
%! try validateattributes (ones(3), {"numeric"}, {"vector"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-vector");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"size", [1 1]});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:incorrect-size");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"numel", 7});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:incorrect-numel");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"ncols", 7});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:incorrect-numcols");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"nrows", 7});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:incorrect-numrows");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"ndims", 5});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:incorrect-numdims");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"square"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-square");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"diag"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-diag");
%! end_try_catch

%!test
%! try validateattributes ([], {"numeric"}, {"nonempty"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonempty");
%! end_try_catch

%!test
%! try validateattributes (speye(2), {"numeric"}, {"nonsparse"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonsparse");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {">", 3});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-greater");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {">=", 3});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-greater-equal");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"<", -3});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-less");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"<=", -3});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-less-equal");
%! end_try_catch

%!test
%! try validateattributes (3, {"numeric"}, {"binary"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-binary");
%! end_try_catch

%!test
%! try validateattributes (1, {"numeric"}, {"even"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-even");
%! end_try_catch

%!test
%! try validateattributes (2, {"numeric"}, {"odd"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-odd");
%! end_try_catch

%!test
%! try validateattributes (1.1, {"numeric"}, {"integer"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-integer");
%! end_try_catch

%!test
%! try validateattributes (1+1i*2, {"numeric"}, {"real"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-real");
%! end_try_catch

%!test
%! try validateattributes (Inf, {"numeric"}, {"finite"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-finite");
%! end_try_catch

%!test
%! try validateattributes (NaN, {"numeric"}, {"nonnan"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonnan");
%! end_try_catch

%!test
%! try validateattributes (-1, {"numeric"}, {"nonnegative"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonnegative");
%! end_try_catch

%!test
%! try validateattributes (0, {"numeric"}, {"nonzero"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonzero");
%! end_try_catch

%!test
%! try validateattributes (-1, {"numeric"}, {"positive"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-positive");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"decreasing"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-decreasing");
%! end_try_catch

%!test
%! try validateattributes ([2 1], {"numeric"}, {"increasing"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-increasing");
%! end_try_catch

%!test
%! try validateattributes ([1 0], {"numeric"}, {"nondecreasing"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nondecreasing");
%! end_try_catch

%!test
%! try validateattributes ([1 2], {"numeric"}, {"nonincreasing"});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:expected-nonincreasing");
%! end_try_catch

%!test
%! try validateattributes (@sin, {"numeric"}, {});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:invalid-type");
%! end_try_catch

%!test
%! try validateattributes (@sin, 1, {});
%! catch id,
%! assert (getfield (id, "identifier"), "Octave:invalid-type");
%! end_try_catch
*/
