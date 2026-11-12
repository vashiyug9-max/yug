cmake_minimum_required(VERSION 3.8)
project(roboteq LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  #here serial has been addes
  serial
  #also this
  diagnostic_updater
)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.

find_package(rclcpp REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(diagnostic_msgs REQUIRED)
#added
find_package(diagnostic_updater REQUIRED)
#until here
find_package(yaml-cpp REQUIRED)
find_package(joint_limits REQUIRED)
find_package(urdf REQUIRED)
find_package(urdfdom REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED
                COMPONENTS
                    chrono
                    system
                    thread
)
#replaced
#find_package(pluginlib REQUIRED)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../serial ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/serial)
#replaced with
find_package(pluginlib REQUIRED)

find_package(pluginlib REQUIRED)
find_package(serial REQUIRED)
#target_link_libraries(roboteq
#  ${serial_LIBRARIES}
#  yaml-cpp
#  urdfdom::urdfdom_model
#  ${Boost_LIBRARIES}
#)
#target_include_directories(roboteq PUBLIC
#  ${Boost_INCLUDE_DIRS}
#  ${serial_INCLUDE_DIRS}
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>
#)
#until here
add_library(roboteq SHARED src/roboteq.cpp src/serial_controller.cpp src/motor.cpp src/configurator/gpio_encoder.cpp src/configurator/gpio_analog.cpp src/configurator/gpio_pulse.cpp src/configurator/motor_param.cpp src/configurator/motor_pid.cpp)
#replaced
#target_link_libraries(roboteq serial yaml-cpp urdfdom::urdfdom_model ${Boost_LIBRARIES})
#replaced with 
target_link_libraries(roboteq
  ${serial_LIBRARIES}
  yaml-cpp
  urdfdom::urdfdom_model
  ${Boost_LIBRARIES}
)
#until here
ament_target_dependencies(roboteq rclcpp hardware_interface diagnostic_msgs  diagnostic_updater urdf joint_limits urdfdom std_msgs pluginlib)

target_compile_features(roboteq PUBLIC cxx_std_17)  # Require C99 and C++17
#replaced 
#target_include_directories(roboteq PUBLIC ${Boost_INCLUDE_DIRS}
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#  $<INSTALL_INTERFACE:include>)
#replaced with
target_include_directories(roboteq PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${serial_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
#until here

# Export hardware plugins
pluginlib_export_plugin_description_file(hardware_interface roboteq.xml)

install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS roboteq
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()
