## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Append to a variable
#   var = var + value
macro(append var value)
  set(${var} "${${var}} ${value}")
endmacro()

# Set variable depending on condition:
#   var = cond ? val_if_true : val_if_false
macro(set_ternary var condition val_if_true val_if_false)
  if (${condition})
    set(${var} "${val_if_true}")
  else()
    set(${var} "${val_if_false}")
  endif()
endmacro()

# Conditionally set a variable
#   if (cond) var = value
macro(set_if condition var value)
  if (${condition})
    set(${var} "${value}")
  endif()
endmacro()

# Conditionally append
#   if (cond) var = var + value
macro(append_if condition var value)
  if (${condition})
    append(${var} "${value}")
  endif()
endmacro()

# Generates C++ files from the specified binary blobs
find_package(PythonInterp REQUIRED)
function(generate_cpp_from_blob out_sources namespace)
  set(${out_sources})
  foreach(in_file ${ARGN})
    get_filename_component(in_file_we ${in_file} NAME_WE)
    get_filename_component(in_dir ${in_file} PATH)
    get_filename_component(in_path ${in_file} ABSOLUTE)
    set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/${in_dir})
    set(out_cpp_path ${out_dir}/${in_file_we}.cpp)
    set(out_hpp_path ${out_dir}/${in_file_we}.h)
    list(APPEND ${out_sources} ${out_cpp_path} ${out_hpp_path})
    add_custom_command(
      OUTPUT ${out_cpp_path} ${out_hpp_path}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
      COMMAND ${PYTHON_EXECUTABLE}
      ARGS ${PROJECT_SOURCE_DIR}/scripts/blob_to_cpp.py ${in_path} -o ${out_cpp_path} -H ${out_hpp_path} -n ${namespace}
      DEPENDS ${in_path}
      COMMENT "Generating CXX source files from blob ${in_path}"
      VERBATIM)
  endforeach()
  set_source_files_properties(${${out_sources}} PROPERTIES GENERATED TRUE)
  set(${out_sources} ${${out_sources}} PARENT_SCOPE)
endfunction()
