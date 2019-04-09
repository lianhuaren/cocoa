#指定FFLv2库源码位置
SET( FFLV2_SOURCE_DIR ${CMAKE_MODULE_PATH}/../FFLv2-lib)
#message(message ${CMAKE_MODULE_PATH})
#message(message ${FFLV2_SOURCE_DIR})

if( IS_DIRECTORY ${FFLV2_SOURCE_DIR} )
  #关闭example的编译
  SET( BUILD_FFLV2_EXAMPLE "OFF")  
  #include目录指向源文件
  SET (FFLV2_LIB_INCLUDE_PATH  ${FFLV2_SOURCE_DIR}/include)
  SET (FFLV2_LIB_LIB_PATH  "")
  
  # 第二个cassdk.out参数用于指定外部文件夹在输出文件夹中的位置
  add_subdirectory( ${FFLV2_SOURCE_DIR} FFLv2.out )  

else()
  # 目录指向安装目录
  SET (FFLV2_LIB_INCLUDE_PATH  ${CMAKE_SOURCE_DIR}/../install/FFLv2-lib/include)
  SET (FFLV2_LIB_LIB_PATH  ${CMAKE_SOURCE_DIR}/../install/FFLv2-lib/include)
endif()

function(group_by_dir src_dir)
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

group_by_dir("${CMAKE_CURRENT_SOURCE_DIR}" ${INC_LIST})
group_by_dir("${CMAKE_CURRENT_SOURCE_DIR}" ${SRC_LIST})