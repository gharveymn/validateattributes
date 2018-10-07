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

template <typename T, typename U>
static void
print_error (T tag, U msg)
{
  octave_value_list args (2);
  args(0) = octave_value (tag);
  args(1) = octave_value (msg);
  Ferror (args);
}

static void
print_error (std::string msg)
{
  Ferror (octave_value (msg));
}

static bool
IsValidIndex (octave_value a)
{
  return a.isnumeric () && a.numel () == 1 && a.scalar_value ()
         && a.scalar_value () == (a.fix ()).scalar_value ();
}

static bool
CheckClass (octave_value ov_A, Array<std::string> cls)
{
  octave_idx_type i;
  std::string A_class = ov_A.class_name ();
  for (i = 0; i < cls.numel (); i++)
    {
      if (A_class == cls (i) || (cls (i) == "float" && ov_A.isfloat ())
          || (cls (i) == "integer" && ov_A.isinteger ())
          || (cls (i) == "numeric" && ov_A.isnumeric ())
          || ov_A.is_instance_of (cls (i)))
        {
          return true;
        }
    }
  return false;
}

static void
InsertIntegerClasses (std::set<std::string> &classes)
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

static void
InsertFloatClasses (std::set<std::string> &classes)
{
  classes.insert ("single");
  classes.insert ("double");
}

static void
ClassNotFoundError (std::string err_ini, Array<std::string> cls,
                    std::string A_class)
{
  octave_idx_type i;
  std::set<std::__cxx11::basic_string<char> >::size_type k;
  std::set<std::string>::iterator classes_iter;
  std::set<std::string> classes;
  std::string err_str;

  for (i = 0; i < cls.numel (); i++)
    {
      if (cls (i) == "integer")
        {
          InsertIntegerClasses (classes);
        }
      else if (cls (i) == "float")
        {
          InsertFloatClasses (classes);
        }
      else if (cls (i) == "numeric")
        {
          InsertIntegerClasses (classes);
          InsertFloatClasses (classes);
        }
      else
        {
          classes.insert (cls (i));
        }
    }

  err_str = err_ini + " must be of class:\n\n ";

  classes_iter = classes.begin ();
  for (k = 0; k < classes.size (); k++, classes_iter++)
    {
      err_str += " " + *classes_iter;
    }
  err_str += "\n\nbut was of class " + A_class;

  print_error ("Octave:invalid-type", err_str);
}

static void
AttributeError (std::string err_id, std::string err_ini, std::string attr_name)
{
  print_error (err_id, err_ini + " must be " + attr_name);
}

static void
UnknownAttributeError (std::string attr_name)
{
  print_error ("Octave:invalid-input-arg",
               "validateattributes: unknown ATTRIBUTE " + attr_name);
}

static bool
CheckSize (dim_vector A_dims, octave_idx_type A_ndims, NDArray attr_dims,
           octave_idx_type attr_ndims)
{

  octave_idx_type i;

  if (attr_ndims < A_ndims)
    return false;

  for (i = 0; i < attr_ndims; i++)
    {
      if (!std::isnan (attr_dims(i)))
        {
          if (i >= A_ndims)
            {
              return false;
            }
          else if (!std::isnan (A_dims(i)) && A_dims(i) != attr_dims(i))
            {
              return false;
            }
        }
    }
  return true;
}

template <typename T>
static void
WriteDimsString (T dims, octave_idx_type ndims, std::string &str)
{
  octave_idx_type i;
  for (i = 0; i < ndims; i++)
    {
      if (std::isnan (dims (i)))
        str += "N";
      else
        str += std::to_string (dims (i));

      if (i < ndims - 1)
        str += "x";
    }
}

template <typename Op>
static bool
CheckMonotone (octave_value A_vec, Op op)
{
  bool A_isnan = (((A_vec.isnan ()).any ()).bool_matrix_value ())(0);
  octave_value A_diff = Fdiff (A_vec) (0);
  bool A_ismono = !((op (A_diff, 0).any ()).bool_matrix_value ())(0);
  return !A_isnan && A_ismono;
}

static bool
CheckEven (octave_value A_vec)
{
  octave_value_list args (2);
  args(0) = A_vec;
  args(1) = octave_value (2);
  octave_value A_rem = Frem (args)(0);
  return !(((A_rem != 0).any ()).bool_matrix_value ())(0);
}

static bool
CheckOdd (octave_value A_vec)
{
  octave_value_list args (2);
  args (0) = A_vec;
  args (1) = octave_value (2);
  octave_value A_mod = Fmod (args)(0);
  return !(((A_mod != 1).any ()).bool_matrix_value ())(0);
}

template <typename Op>
static bool
CheckCompare (octave_value A_vec, octave_value attr_val, Op op)
{
  return ((op (A_vec, attr_val).all ()).bool_matrix_value ())(0);
}

static void
ComparisonError (std::string tag, std::string cmp_str, std::string err_ini,
                 octave_value attr_val)
{
  octave_value_list args (3);
  args (0) = octave_value ("%s must be " + cmp_str + " %f");
  args (1) = octave_value (err_ini);
  args (2) = octave_value (attr_val);
  print_error (tag, Fsprintf (args) (0));
}

template <typename T>
static bool
CheckMatrixDiag (T A)
{
  // check nnz on the diagonal and compare it to overall nnz
  Array<octave_idx_type> found = A.find ();
  dim_vector A_dims = A.dims ();
  octave_idx_type i;
  for (i = 0; i < found.numel (); i++)
    {
      if (found(i) % A_dims(0) != found(i) / A_dims(0))
        return false;
    }
  return true;
}

static bool
CheckDiag (octave_value ov_A)
{
  if (ov_A.is_diag_matrix ())
    return true;
  else if ((ov_A.isnumeric () || ov_A.islogical ()) && ov_A.ndims () == 2)
    {
      if (ov_A.is_int8_type ())
        return CheckMatrixDiag (ov_A.int8_array_value ());
      else if (ov_A.is_int16_type ())
        return CheckMatrixDiag (ov_A.int16_array_value ());
      else if (ov_A.is_int32_type ())
        return CheckMatrixDiag (ov_A.int32_array_value ());
      else if (ov_A.is_int64_type ())
        return CheckMatrixDiag (ov_A.int64_array_value ());
      else if (ov_A.is_uint8_type ())
        return CheckMatrixDiag (ov_A.uint8_array_value ());
      else if (ov_A.is_uint16_type ())
        return CheckMatrixDiag (ov_A.uint16_array_value ());
      else if (ov_A.is_uint32_type ())
        return CheckMatrixDiag (ov_A.uint32_array_value ());
      else if (ov_A.is_uint64_type ())
        return CheckMatrixDiag (ov_A.uint64_array_value ());
      else if (ov_A.is_single_type ())
        return CheckMatrixDiag (ov_A.float_array_value ());
      else if (ov_A.is_double_type ())
        return CheckMatrixDiag (ov_A.array_value ());
      else if (ov_A.islogical ())
        return CheckMatrixDiag (ov_A.bool_array_value ());
      return false;
    }
  else
    return false;
}

static void
CheckAttributes (octave_value ov_A, Cell attr, std::string err_ini)
{

  octave_idx_type i;
  octave_value attr_val;
  std::string name;
  size_t len;

  octave_value A_vec = ov_A.reshape (dim_vector (ov_A.numel (), 1));
  dim_vector A_dims = ov_A.dims ();
  octave_idx_type A_ndims = ov_A.ndims ();

  i = 0;
  while (i < attr.numel ())
    {
      name = attr (i++).string_value ();
      len = name.length ();

      if (len < 1)
        UnknownAttributeError (name);

      switch (std::tolower (name[0]))
        {
        case '2': // 2d
          {
            if (len == 2 && std::tolower (name[1]) == 'd')
              {
                if (A_ndims != 2)
                  AttributeError ("Octave:expected-2d", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case '3': // 3d
          {
            if (len == 2 && std::tolower (name[1]) == 'd')
              {
                if (A_ndims > 3)
                  AttributeError ("Octave:expected-3d", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'c': // column
          {
            if (octave::string::strcmpi (name, "column"))
              {
                if (A_ndims != 2 || A_dims(1) != 1)
                  AttributeError ("Octave:expected-column", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'r': // row, real
          {
            if (octave::string::strcmpi (name, "row")) // row
              {
                if (A_ndims != 2 || A_dims(0) != 1)
                  AttributeError ("Octave:expected-row", err_ini, name);
              }
            else if (octave::string::strcmpi (name, "real")) // real
              {
                if (!ov_A.isreal ())
                  AttributeError ("Octave:expected-real", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 's': // scalar, square, size,
          {
            if (octave::string::strcmpi (name, "scalar")) // scalar
              {
                if (ov_A.numel () != 1)
                  AttributeError ("Octave:expected-scalar", err_ini, name);
              }
            else if (octave::string::strcmpi (name, "square")) // square
              {
                if (A_ndims != 2 || A_dims(0) != A_dims (1))
                  AttributeError ("Octave:expected-square", err_ini, name);
              }
            else if (octave::string::strcmpi (name, "size")) // size
              {
                if (i >= attr.numel ())
                  print_error ("Incorrect number of attribute cell arguments");
                attr_val = attr (i++);
                NDArray attr_dims = attr_val.array_value ();
                octave_idx_type attr_ndims = attr_val.numel ();
                if (!CheckSize (A_dims, A_ndims, attr_dims, attr_ndims))
                  {
                    std::string A_dims_str;
                    WriteDimsString (A_dims, A_ndims, A_dims_str);

                    std::string attr_dims_str;
                    WriteDimsString (attr_dims, attr_ndims, attr_dims_str);

                    print_error ("Octave:incorrect-size",
                                 err_ini + " must be of size " + attr_dims_str
                                     + " but was " + A_dims_str);
                  }
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'v': // vector
          {
            if (octave::string::strcmpi (name, "vector"))
              {
                if (A_ndims != 2 || (A_dims(0) != 1 && A_dims (1) != 1))
                  AttributeError ("Octave:expected-vector", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'd': // diag, decreasing
          {
            if (octave::string::strcmpi (name, "diag")) // diag
              {
                if (!CheckDiag (ov_A))
                  AttributeError ("Octave:expected-diag", err_ini, name);
              }
            else if (octave::string::strcmpi (name,
                                              "decreasing")) // decreasing
              {
                if (!CheckMonotone (A_vec, op_ge))
                  AttributeError ("Octave:expected-decreasing", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'n': // nonempty, nonsparse, nonnan, nonnegative, nonzero,
                  // nondecreasing, nonincreasing, numel, ncols, nrows, ndims,
          {
            if (len > 1)
              {
                switch (std::tolower (name[1]))
                  {
                  case 'o': // nonempty, nonsparse, nonnan, nonnegative,
                            // nonzero, nondecreasing, nonincreasing
                    {
                      if (octave::string::strcmpi (name,
                                                   "nonempty")) // nonempty
                        {
                          if (ov_A.isempty ())
                            AttributeError ("Octave:expected-nonempty",
                                            err_ini, name);
                        }
                      else if (octave::string::strcmpi (
                                   name, "nonsparse")) // nonsparse
                        {
                          if (ov_A.issparse ())
                            AttributeError ("Octave:expected-nonsparse",
                                            err_ini, name);
                        }
                      else if (octave::string::strcmpi (name,
                                                        "nonnan")) // nonnan
                        {
                          if (!ov_A.isinteger ()
                              && (((A_vec.isnan ()).any ())
                                      .bool_matrix_value ())(0))
                            AttributeError ("Octave:expected-nonnan", err_ini,
                                            name);
                        }
                      else if (octave::string::strcmpi (
                                   name, "nonnegative")) // nonnegative
                        {
                          if ((((A_vec < 0).any ()).bool_matrix_value ())(0))
                            AttributeError ("Octave:expected-nonnegative",
                                            err_ini, name);
                        }
                      else if (octave::string::strcmpi (name,
                                                        "nonzero")) // nonzero
                        {
                          if ((((A_vec == 0).any ()).bool_matrix_value ())(0))
                            AttributeError ("Octave:expected-nonzero", err_ini,
                                            name);
                        }
                      else if (octave::string::strcmpi (
                                   name, "nondecreasing")) // nondecreasing
                        {
                          if (!CheckMonotone (A_vec, op_lt))
                            AttributeError ("Octave:expected-nondecreasing",
                                            err_ini, name);
                        }
                      else if (octave::string::strcmpi (
                                   name, "nonincreasing")) // nonincreasing
                        {
                          if (!CheckMonotone (A_vec, op_gt))
                            AttributeError ("Octave:expected-nonincreasing",
                                            err_ini, name);
                        }
                      else
                        UnknownAttributeError (name);
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
                        UnknownAttributeError (name);
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
                              || A_dims (1) != attr_val.idx_type_value ())
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
                        UnknownAttributeError (name);
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
                              || A_dims (0) != attr_val.idx_type_value ())
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
                        UnknownAttributeError (name);
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
                              print_error (
                                  "Octave:incorrect-numdims",
                                  err_ini + " must have "
                                      + std::to_string (
                                            attr_val.idx_type_value ())
                                      + " dimensions");
                            }
                        }
                      else
                        UnknownAttributeError (name);
                      break;
                    }
                  default:
                    UnknownAttributeError (name);
                  }
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'b': // binary
          {
            if (octave::string::strcmpi (name, "binary"))
              {
                if (!ov_A.islogical ()
                    && (((op_el_and ((A_vec != 1), (A_vec != 0))).any ())
                            .bool_matrix_value ())(0))
                  AttributeError ("Octave:expected-binary", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'e': // even
          {
            if (octave::string::strcmpi (name, "even"))
              {

                if (!CheckEven (A_vec))
                  AttributeError ("Octave:expected-even", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'o': // odd
          {
            if (octave::string::strcmpi (name, "odd"))
              {
                if (!CheckOdd (A_vec))
                  AttributeError ("Octave:expected-odd", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'i': // integer, increasing
          {
            if (octave::string::strcmpi (name, "integer")) // integer
              {
                if (!ov_A.isinteger ()
                    && (((A_vec.ceil () != A_vec).any ())
                            .bool_matrix_value ()) (0))
                  AttributeError ("Octave:expected-integer", err_ini, name);
              }
            else if (octave::string::strcmpi (name,
                                              "increasing")) // increasing
              {
                if (!CheckMonotone (A_vec, op_le))
                  AttributeError ("Octave:expected-increasing", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'f': // finite
          {
            if (octave::string::strcmpi (name, "finite"))
              {
                if (!ov_A.isinteger ()
                    && !(((A_vec.isfinite ()).all ()).bool_matrix_value ()) (
                           0))
                  AttributeError ("Octave:expected-finite", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case 'p': // positive
          {
            if (octave::string::strcmpi (name, "positive"))
              {
                if ((((A_vec <= 0).any ()).bool_matrix_value ())(0))
                  AttributeError ("Octave:expected-positive", err_ini, name);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case '>': // >, >=
          {
            if (len == 1) // >
              {
                if (i >= attr.numel ())
                  print_error ("Incorrect number of attribute cell arguments");
                attr_val = attr (i++);
                if (!CheckCompare (A_vec, attr_val, op_gt))
                  ComparisonError ("Octave:expected-greater", "greater than",
                                   err_ini, attr_val);
              }
            else if (len == 2 && std::tolower (name[1]) == '=') // >=
              {
                if (i >= attr.numel ())
                  print_error ("Incorrect number of attribute cell arguments");
                attr_val = attr (i++);
                if (!CheckCompare (A_vec, attr_val, op_ge))
                  ComparisonError ("Octave:expected-greater-equal",
                                   "greater than or equal to", err_ini,
                                   attr_val);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        case '<': // <, <=
          {
            if (len == 1) // <
              {
                if (i >= attr.numel ())
                  print_error ("Incorrect number of attribute cell arguments");
                attr_val = attr (i++);
                if (!CheckCompare (A_vec, attr_val, op_lt))
                  ComparisonError ("Octave:expected-less", "less than",
                                   err_ini, attr_val);
              }
            else if (len == 2 && std::tolower (name[1]) == '=') // <=
              {
                if (i >= attr.numel ())
                  print_error ("Incorrect number of attribute cell arguments");
                attr_val = attr (i++);
                if (!CheckCompare (A_vec, attr_val, op_le))
                  ComparisonError ("Octave:expected-less-equal",
                                   "less than or equal to", err_ini, attr_val);
              }
            else
              UnknownAttributeError (name);
            break;
          }
        default:
          UnknownAttributeError (name);
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

  // octave_idx_type i;

  octave_value ov_A;
  octave_value ov_cls;
  octave_value ov_attr;

  std::string A_class;
  Array<std::string> cls;
  Cell attr;

  std::string err_ini;
  std::string func_name;
  std::string var_name = "input";

  octave_idx_type nargin = args.length ();
  if (nargin < 3 || nargin > 6)
    {
      print_usage ();
      return octave_value ();
    }

  ov_A    = args(0);
  ov_cls  = args(1);
  ov_attr = args(2);

  if (!ov_cls.iscellstr ())
    {
      print_error (
          "Octave:invalid-type",
          "validateattributes: CLASSES must be a cell array of strings");
    }
  else if (!ov_attr.iscell ())
    {
      print_error ("Octave:invalid-type",
                   "validateattributes: ATTRIBUTES must be a cell array");
    }

  cls = ov_cls.cellstr_value ();
  attr = ov_attr.cell_value ();

  if (nargin > 3)
    {
      if (args(3).is_string ())
        {
          func_name = args(3).string_value () + ": ";
        }
      else if (nargin == 4 && IsValidIndex (args(3)))
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
          if (!args(4).is_string ())
            {
              print_error ("Octave:invalid-type",
                           "validateattributes: VAR_NAME must be a string");
            }
          var_name = args(4).string_value ();

          if (nargin > 5)
            {
              if (!IsValidIndex (args(5)))
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

  if (!cls.isempty () && !CheckClass (ov_A, cls))
    {
      ClassNotFoundError (err_ini, cls, ov_A.class_name ());
    }

  CheckAttributes (ov_A, attr, err_ini);

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
%!error <nonincreasing> validateattributes ([7 8 4 9 20], {},
{"nonincreasing"})
%!error <nonincreasing> validateattributes ([7 8 NaN 9 20], {},
{"nonincreasing"})
%!error <nondecreasing> validateattributes ([7 8 4 3 -5], {},
{"nondecreasing"})
%!error <nondecreasing> validateattributes ([7 NaN 4 3 -5], {},
{"nondecreasing"})
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

%!test
%! validateattributes (rand (5), {"numeric"}, {});
%! validateattributes (rand (5), {"float"}, {});
%! validateattributes (rand (5), {"double"}, {});
%! validateattributes ("text", {"char"}, {});
%! validateattributes (rand (5), {}, {"2d"});
%! validateattributes (rand (5), {}, {"3d"});
%! validateattributes (rand (5, 5, 5), {}, {"3d"});
%! validateattributes (rand (5, 1), {}, {"column"});
%! validateattributes (rand (1, 5), {}, {"row"});
%! validateattributes ("a", {}, {"scalar"});
%! validateattributes (5, {}, {"scalar"});
%! validateattributes (rand (1, 5), {}, {"vector"});
%! validateattributes (rand (5, 1), {}, {"vector"});
%! validateattributes (rand (5), {}, {"square"});
%! validateattributes (rand (5), {}, {"nonempty"});
%! validateattributes (rand (5), {}, {"nonsparse"});
%! validateattributes ([0 1 0 1 0], {}, {"binary"});
%! validateattributes (rand (5) > 0.5, {}, {"binary"});
%! validateattributes ([8 4 0 6], {}, {"even"});
%! validateattributes ([-1 3 5], {}, {"odd"});
%! validateattributes ([8 4 0 6], {}, {"real"});
%! validateattributes ([8 4i 0 6], {}, {"finite"});
%! validateattributes (uint8 ([8 4]), {}, {"finite"});
%! validateattributes ([8 Inf], {}, {"nonnan"});
%! validateattributes ([0 7 4], {}, {"nonnegative"});
%! validateattributes ([-8 7 4], {}, {"nonzero"});
%! validateattributes ([8 7 4], {}, {"positive"});
%! validateattributes ([8 7 4 -5], {}, {"decreasing"});
%! validateattributes ([-8 -7 4 5], {}, {"increasing"});
%! validateattributes ([8 4 4 -5], {}, {"nonincreasing"});
%! validateattributes ([-8 -8 4 5], {}, {"nondecreasing"});
%! validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 6 7 2]});
%! validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 NaN 7 2]});
%! validateattributes (rand (4, 6, 7, 2), {}, {"size", [4 6 NaN 2 NaN]});
%! validateattributes (rand (6, 2), {}, {"numel", 12});
%! validateattributes (rand (6, 2), {}, {"ncols", 2});
%! validateattributes (rand (6, 2), {}, {"nrows", 6});
%! validateattributes (rand (6, 2, 4, 5), {}, {"ndims", 4});
%! validateattributes ([4 5 6 7], {}, {">", 3});
%! validateattributes ([4 5 6 7], {}, {">=", 4});
%! validateattributes ([4 5 6 7], {}, {"<", 8});
%! validateattributes ([4 5 6 7], {}, {"<=", 7});
%! validateattributes (eye (3), {}, {"diag"});
%! validateattributes ([1 0 0; 0 1 0; 0 0 1], {}, {"diag"});
%! validateattributes (zeros (3), {}, {"diag"});

%!test
%! validateattributes ([0 1 0 1], {"double", "uint8"}, {"binary", "size", [NaN
4], "nonnan"});

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
