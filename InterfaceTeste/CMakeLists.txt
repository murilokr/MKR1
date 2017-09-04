cmake_minimum_required(VERSION 3.5)
project(opengl)
add_executable(opengl opengl.cpp)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} )

target_link_libraries(opengl ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
