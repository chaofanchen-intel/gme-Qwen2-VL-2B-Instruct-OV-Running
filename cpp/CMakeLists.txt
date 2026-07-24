cmake_minimum_required(VERSION 3.16)
project(gme_qwen2vl_openvino CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Locate the openvino-tokenizers extension next to the OpenVINO runtime libs so we
# can bake its path in as a default (overridable at runtime via GME_TOKENIZERS_LIB).
get_target_property(_ov_loc openvino::runtime LOCATION)
get_filename_component(_ov_libdir "${_ov_loc}" DIRECTORY)
find_library(GME_OV_TOKENIZERS_LIB
  NAMES openvino_tokenizers
  HINTS "${_ov_libdir}"
  PATHS "${_ov_libdir}")
if(NOT GME_OV_TOKENIZERS_LIB)
  message(WARNING "libopenvino_tokenizers not found near ${_ov_libdir}; "
                  "set GME_TOKENIZERS_LIB at runtime.")
endif()

add_library(gme_qwen2vl STATIC
  src/gme_qwen2vl.cpp
  src/image_preprocess.cpp)
target_include_directories(gme_qwen2vl PUBLIC include PRIVATE src)
target_link_libraries(gme_qwen2vl PUBLIC openvino::runtime
  PRIVATE ${OpenCV_LIBS})
if(GME_OV_TOKENIZERS_LIB)
  target_compile_definitions(gme_qwen2vl PRIVATE
    GME_OV_TOKENIZERS_LIB="${GME_OV_TOKENIZERS_LIB}")
endif()

add_executable(gme_embed src/main.cpp)
target_link_libraries(gme_embed PRIVATE gme_qwen2vl)
