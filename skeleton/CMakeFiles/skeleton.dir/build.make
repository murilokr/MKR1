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
CMAKE_SOURCE_DIR = /home/murilo/Documents/TCC/Applications/skeleton

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/murilo/Documents/TCC/Applications/skeleton

# Include any dependencies generated for this target.
include CMakeFiles/skeleton.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/skeleton.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/skeleton.dir/flags.make

CMakeFiles/skeleton.dir/skeleton.cpp.o: CMakeFiles/skeleton.dir/flags.make
CMakeFiles/skeleton.dir/skeleton.cpp.o: skeleton.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/murilo/Documents/TCC/Applications/skeleton/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/skeleton.dir/skeleton.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/skeleton.dir/skeleton.cpp.o -c /home/murilo/Documents/TCC/Applications/skeleton/skeleton.cpp

CMakeFiles/skeleton.dir/skeleton.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/skeleton.dir/skeleton.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/murilo/Documents/TCC/Applications/skeleton/skeleton.cpp > CMakeFiles/skeleton.dir/skeleton.cpp.i

CMakeFiles/skeleton.dir/skeleton.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/skeleton.dir/skeleton.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/murilo/Documents/TCC/Applications/skeleton/skeleton.cpp -o CMakeFiles/skeleton.dir/skeleton.cpp.s

CMakeFiles/skeleton.dir/skeleton.cpp.o.requires:

.PHONY : CMakeFiles/skeleton.dir/skeleton.cpp.o.requires

CMakeFiles/skeleton.dir/skeleton.cpp.o.provides: CMakeFiles/skeleton.dir/skeleton.cpp.o.requires
	$(MAKE) -f CMakeFiles/skeleton.dir/build.make CMakeFiles/skeleton.dir/skeleton.cpp.o.provides.build
.PHONY : CMakeFiles/skeleton.dir/skeleton.cpp.o.provides

CMakeFiles/skeleton.dir/skeleton.cpp.o.provides.build: CMakeFiles/skeleton.dir/skeleton.cpp.o


# Object files for target skeleton
skeleton_OBJECTS = \
"CMakeFiles/skeleton.dir/skeleton.cpp.o"

# External object files for target skeleton
skeleton_EXTERNAL_OBJECTS =

skeleton: CMakeFiles/skeleton.dir/skeleton.cpp.o
skeleton: CMakeFiles/skeleton.dir/build.make
skeleton: /usr/lib/libOpenNI.so
skeleton: /usr/local/lib/libopencv_ml.so.3.2.0
skeleton: /usr/local/lib/libopencv_objdetect.so.3.2.0
skeleton: /usr/local/lib/libopencv_shape.so.3.2.0
skeleton: /usr/local/lib/libopencv_stitching.so.3.2.0
skeleton: /usr/local/lib/libopencv_superres.so.3.2.0
skeleton: /usr/local/lib/libopencv_videostab.so.3.2.0
skeleton: /usr/local/lib/libopencv_calib3d.so.3.2.0
skeleton: /usr/local/lib/libopencv_features2d.so.3.2.0
skeleton: /usr/local/lib/libopencv_flann.so.3.2.0
skeleton: /usr/local/lib/libopencv_highgui.so.3.2.0
skeleton: /usr/local/lib/libopencv_photo.so.3.2.0
skeleton: /usr/local/lib/libopencv_video.so.3.2.0
skeleton: /usr/local/lib/libopencv_videoio.so.3.2.0
skeleton: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
skeleton: /usr/local/lib/libopencv_imgproc.so.3.2.0
skeleton: /usr/local/lib/libopencv_core.so.3.2.0
skeleton: CMakeFiles/skeleton.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/murilo/Documents/TCC/Applications/skeleton/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable skeleton"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/skeleton.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/skeleton.dir/build: skeleton

.PHONY : CMakeFiles/skeleton.dir/build

CMakeFiles/skeleton.dir/requires: CMakeFiles/skeleton.dir/skeleton.cpp.o.requires

.PHONY : CMakeFiles/skeleton.dir/requires

CMakeFiles/skeleton.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/skeleton.dir/cmake_clean.cmake
.PHONY : CMakeFiles/skeleton.dir/clean

CMakeFiles/skeleton.dir/depend:
	cd /home/murilo/Documents/TCC/Applications/skeleton && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/murilo/Documents/TCC/Applications/skeleton /home/murilo/Documents/TCC/Applications/skeleton /home/murilo/Documents/TCC/Applications/skeleton /home/murilo/Documents/TCC/Applications/skeleton /home/murilo/Documents/TCC/Applications/skeleton/CMakeFiles/skeleton.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/skeleton.dir/depend

