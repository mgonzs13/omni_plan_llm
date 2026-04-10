cmake_minimum_required(VERSION 3.5)
project(omni_plan_llm)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(pluginlib REQUIRED)
find_package(omni_plan REQUIRED)
find_package(llama_msgs REQUIRED)

# Include directories
include_directories(include)

set(DEPENDENCIES
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  pluginlib::pluginlib
  omni_plan::omni_plan
  ${llama_msgs_TARGETS}
)

# Register the pluginlib description file
pluginlib_export_plugin_description_file(omni_plan plugins.xml)

# Build the planner plugin shared library
add_library(llm_planner SHARED
  src/omni_plan_llm/llm_planner.cpp
)
target_include_directories(llm_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(llm_planner PUBLIC ${DEPENDENCIES})

install(TARGETS llm_planner
  LIBRARY DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include/
)

# Testing
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_llm_planner test/test_llm_planner.cpp)
  target_include_directories(test_llm_planner PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(test_llm_planner
    llm_planner
    ${DEPENDENCIES}
  )
endif()

ament_package()
