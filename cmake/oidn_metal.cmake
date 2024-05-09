## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Builds a Metal library from the given Metal shader sources and
# adds C++ sources generated from the Metal library blob to the specified target
function(metallib_target_add_sources target metallib)
	set(options INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)
  cmake_parse_arguments(PARSE_ARGV 2 METAL "" "" "${options}")

  set(include_dirs "")
  foreach(inc ${METAL_INCLUDE_DIRECTORIES})
    file(TO_NATIVE_PATH "${inc}" inc_native_path)
    list(APPEND include_dirs "-I${inc_native_path}")
  endforeach()

  set(compile_defs "")
  foreach(def ${METAL_COMPILE_DEFINITIONS})
    list(APPEND compile_defs "-D${def}")
  endforeach()

  # Compile each Metal shader to an AIR (Apple Intermediate Representation) file
  set(air_files "")

  foreach(src ${METAL_UNPARSED_ARGUMENTS})
    get_filename_component(src_file ${src} ABSOLUTE)
    get_filename_component(src_name ${src} NAME_WE)
    get_filename_component(src_dir  ${src_file} DIRECTORY)

    oidn_get_build_path(out_dir ${src_dir})
    set(air_file ${out_dir}/CMakeFiles/${target}.dir/${src_name}.air)
    file(RELATIVE_PATH air_file_rel ${CMAKE_BINARY_DIR} ${air_file})

    add_custom_command(
      OUTPUT ${air_file}
      COMMAND xcrun -sdk iphoneos metal
                -c ${src_file}
                ${include_dirs}
                ${compile_defs}
                ${METAL_COMPILE_OPTIONS}
                -MD -MT ${air_file} -MF ${air_file}.d
                -o ${air_file}
      DEPENDS ${src_file}
      DEPFILE ${air_file}.d
      COMMENT "Building Metal AIR file ${air_file_rel}"
    )

    list(APPEND air_files ${air_file})
  endforeach()

  # Create the Metal library by linking the AIR files
  set(metallib_file ${out_dir}/${metallib}.metallib)
  file(RELATIVE_PATH metallib_file_rel ${CMAKE_BINARY_DIR} ${metallib_file})

  add_custom_command(
    OUTPUT ${metallib_file}
    COMMAND xcrun -sdk iphoneos metallib ${air_files} -o ${metallib_file}
    DEPENDS ${air_files}
    COMMENT "Linking Metal library ${metallib_file_rel}"
  )

  # Generate C++ code from the Metal library blob
  oidn_generate_cpp_from_blob(cpp_files "${OIDN_NAMESPACE}::blobs" ${metallib_file})

  # Add the generated C++ files to the target
  get_property(target_sources TARGET ${target} PROPERTY SOURCES)
  list(APPEND target_sources ${cpp_files})
  set_target_properties(${target} PROPERTIES SOURCES "${target_sources}")
endfunction()