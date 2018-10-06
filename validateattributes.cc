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

#include <octave/oct.h>
#include <octave/oct-string.h>

static bool IsValidIndex (octave_value a)
{
  return a.isnumeric () 
         && a.numel () == 1 
         && a.scalar_value () 
         && a.scalar_value () == (a.fix ()).scalar_value ();
}

static bool CheckClass (octave_value ov_A, Array<std::string> cls)
{
  octave_idx_type i;
  std::string A_class = ov_A.class_name ();
  for (i = 0; i < cls.numel (); i++)
  {
    if (A_class == cls(i)
        || (cls(i) == "float" && ov_A.isfloat ())
        || (cls(i) == "integer" && ov_A.isinteger ())
        || (cls(i) == "numeric" && ov_A.isnumeric ())
        || ov_A.is_instance_of (cls(i)))
    {
      return true;
    }
  }
  return false;
}

static void InsertIntegerClasses (std::set<std::string>& classes)
{
  classes.insert("int8");
  classes.insert("int16");
  classes.insert("int32");
  classes.insert("int64");
  classes.insert("uint8");
  classes.insert("uint16");
  classes.insert("uint32");
  classes.insert("uint64");
}

static void InsertFloatClasses (std::set<std::string>& classes)
{
  classes.insert("single");
  classes.insert("double");
}

static void ClassNotFoundError (std::string err_ini, Array<std::string> cls, std::string A_class)
{
  octave_idx_type i;
  std::set<std::__cxx11::basic_string<char> >::size_type k;
  std::set<std::string>::iterator classes_iter;
  std::set<std::string> classes;
  std::string err_str;
  
  for (i = 0; i < cls.numel (); i++)
  {
    if (cls(i) == "integer")
    {
      InsertIntegerClasses (classes);
    }
    else if (cls(i) == "float")
    {
      InsertFloatClasses (classes);
    }
    else if (cls(i) == "numeric")
    {
      InsertIntegerClasses (classes);
      InsertFloatClasses (classes);
    }
    else
    {
      classes.insert (cls(i));
    }
  }
  
  err_str = err_ini + "must be of class:\n\n ";
  
  classes_iter = classes.begin ();
  for (k = 0; k < classes.size (); k++, classes_iter++)
  {
    err_str += " " + *classes_iter;
  }
  err_str += "\n\nbut was of class " + A_class;
  
  error ("Octave:invalid-type", err_str.c_str ());
  
}

static void AttributeNameError (std::string err_id, std::string err_ini, std::string attr_name)
{
  error (err_id, (err_ini + " must be " + attr_name).c_str());
}

static void UnknownAttributeError (std::string attr_name)
{
  error ("Octave:invalid-input-arg", ("validateattributes: unknown ATTRIBUTE " + attr_name).c_str());
}

static void CheckAttributes (octave_value ov_A, Cell attr, std::string err_ini)
{
  octave_idx_type i;
  std::string attr_name;
  for (i = 0; i < attr.numel (); i++)
  {
    attr_name = attr(i).string_value ();
    
    if (attr_name.length () < 1)
      UnknownAttributeError(attr_name);
    
    switch (attr_name[0])
    {
      case '2':
      {
        
      }
      case '3':
      {
        
      }
      case 'c':
      {
        
      }
      case 'r':
      {
        
      }
      case 's':
      {
        
      }
      case 'v':
      {
        
      }
      case 'd':
      {
        
      }
      case 'n':
      {
        
      }
      case 'b':
      {
        
      }
      case 'e':
      {
        
      }
      case 'o':
      {
        
      }
      case 'i':
      {
        
      }
      case 'r':
      {
        
      }
      case 'f':
      {
        
      }
      case 'p':
      {
        
      }
      case '>':
      {
        
      }
      case '<':
      {
        
      }
      default:
        UnknownAttributeError(attr_name);
    }
  }
}

DEFUN_DLD (validateattributes, args, nargout, 
"-*- texinfo -*-\n\
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
    print_usage();
    return octave_value();
  }
  
  ov_A    = args(0);
  ov_cls  = args(1);
  ov_attr = args(2);
  
  if(! ov_cls.iscellstr ())
  {
    error ("Octave:invalid-type",
           "validateattributes: CLASSES must be a cell array of strings");
  }
  else if(! ov_attr.iscell ())
  {
    error ("Octave:invalid-type",
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
    else if (nargin == 4 && IsValidIndex(args(3)))
    {
      var_name = "input " + std::to_string(args(3).idx_type_value ());
    }
    else
    {
      error ("Octave:invalid-input-arg",
             "validateattributes: 4th input argument must be ARG_IDX or FUNC_NAME");
    }
    
    if (nargin > 4)
    {
      if (! args(4).is_string ())
      {
        error ("Octave:invalid-type",
               "validateattributes: VAR_NAME must be a string");
      }
      var_name = args(4).string_value ();
      
      if (nargin > 5)
      {
        if (! IsValidIndex (args (5)))
        {
          error ("Octave:invalid-input-arg",
                 "validateattributes: ARG_IDX must be a positive integer");
        }
        var_name += " (argument #" + std::to_string(args (5).idx_type_value ()) + ")";
      }
    }
  }
  
  err_ini = func_name + var_name;
  
  if (! cls.isempty () && ! CheckClass(ov_A, cls))
  {
    ClassNotFoundError (err_ini, cls, ov_A.class_name ());
  }
  
  return octave_value ();
  
}

