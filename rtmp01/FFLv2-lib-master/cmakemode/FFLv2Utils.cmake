#分组源文件 
FUNCTION(group_by_dir src_dir)
  foreach(FILE ${ARGN})
    # 获取文件绝对路径
    get_filename_component(FULL_NAME "${FILE}" ABSOLUTE)

    # 获取文件父路径
    get_filename_component(PARENT_DIR "${FULL_NAME}" PATH)

    # 移除父路径中的源码根路径
    string(REPLACE "${ARGV0}" "" GROUP "${PARENT_DIR}")

    # 确保路径使用windows路径符号
    string(REPLACE "/" "\\" GROUP "${GROUP}")

    # 将文件归组到 "Source Files" 和 "Header Files"
    if("${FILE}" MATCHES ".*\\.h")
      set(GROUP "Header Files${GROUP}")
    else()
      set(GROUP "Source Files${GROUP}")
    endif()

    source_group("${GROUP}" FILES "${FILE}")
  endforeach()
endfunction(group_by_dir)

#打印参数
FUNCTION(printf arg)
   message("message:" ${ARGV0})
ENDFUNCTION()


# mt编译
FUNCTION(MSVC_mt)
if (MSVC)
    add_definitions("/wd4819")
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif(MSVC)
ENDFUNCTION()

#group_by_dir("${CMAKE_CURRENT_SOURCE_DIR}" ${INC_LIST})
#group_by_dir("${CMAKE_CURRENT_SOURCE_DIR}" ${SRC_LIST})