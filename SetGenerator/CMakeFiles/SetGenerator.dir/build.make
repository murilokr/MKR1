# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/murilo/Documents/TCC/SetGenerator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/murilo/Documents/TCC/SetGenerator

# Include any dependencies generated for this target.
include CMakeFiles/SetGenerator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/SetGenerator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SetGenerator.dir/flags.make

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o: CMakeFiles/SetGenerator.dir/flags.make
CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o: SetGenerator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/murilo/Documents/TCC/SetGenerator/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o -c /home/murilo/Documents/TCC/SetGenerator/SetGenerator.cpp

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SetGenerator.dir/SetGenerator.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/murilo/Documents/TCC/SetGenerator/SetGenerator.cpp > CMakeFiles/SetGenerator.dir/SetGenerator.cpp.i

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SetGenerator.dir/SetGenerator.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/murilo/Documents/TCC/SetGenerator/SetGenerator.cpp -o CMakeFiles/SetGenerator.dir/SetGenerator.cpp.s

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.requires:

.PHONY : CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.requires

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.provides: CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.requires
	$(MAKE) -f CMakeFiles/SetGenerator.dir/build.make CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.provides.build
.PHONY : CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.provides

CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.provides.build: CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o


# Object files for target SetGenerator
SetGenerator_OBJECTS = \
"CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o"

# External object files for target SetGenerator
SetGenerator_EXTERNAL_OBJECTS =

SetGenerator: CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o
SetGenerator: CMakeFiles/SetGenerator.dir/build.make
SetGenerator: /usr/local/lib/libopencv_ml.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_objdetect.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_shape.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_stitching.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_superres.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_videostab.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_calib3d.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_features2d.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_flann.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_highgui.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_photo.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_video.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_videoio.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_imgproc.so.3.2.0
SetGenerator: /usr/local/lib/libopencv_core.so.3.2.0
SetGenerator: CMakeFiles/SetGenerator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/murilo/Documents/TCC/SetGenerator/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SetGenerator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SetGenerator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SetGenerator.dir/build: SetGenerator

.PHONY : CMakeFiles/SetGenerator.dir/build

CMakeFiles/SetGenerator.dir/requires: CMakeFiles/SetGenerator.dir/SetGenerator.cpp.o.requires

.PHONY : CMakeFiles/SetGenerator.dir/requires

CMakeFiles/SetGenerator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SetGenerator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SetGenerator.dir/clean

CMakeFiles/SetGenerator.dir/depend:
	cd /home/murilo/Documents/TCC/SetGenerator && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/murilo/Documents/TCC/SetGenerator /home/murilo/Documents/TCC/SetGenerator /home/murilo/Documents/TCC/SetGenerator /home/murilo/Documents/TCC/SetGenerator /home/murilo/Documents/TCC/SetGenerator/CMakeFiles/SetGenerator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SetGenerator.dir/depend

