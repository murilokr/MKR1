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
CMAKE_SOURCE_DIR = /home/murilo/Documents/TCC/Applications/DatasetRecorder

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/murilo/Documents/TCC/Applications/DatasetRecorder

# Include any dependencies generated for this target.
include CMakeFiles/recorder.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/recorder.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/recorder.dir/flags.make

CMakeFiles/recorder.dir/datasetrecorder.cpp.o: CMakeFiles/recorder.dir/flags.make
CMakeFiles/recorder.dir/datasetrecorder.cpp.o: datasetrecorder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/murilo/Documents/TCC/Applications/DatasetRecorder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/recorder.dir/datasetrecorder.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/recorder.dir/datasetrecorder.cpp.o -c /home/murilo/Documents/TCC/Applications/DatasetRecorder/datasetrecorder.cpp

CMakeFiles/recorder.dir/datasetrecorder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/recorder.dir/datasetrecorder.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/murilo/Documents/TCC/Applications/DatasetRecorder/datasetrecorder.cpp > CMakeFiles/recorder.dir/datasetrecorder.cpp.i

CMakeFiles/recorder.dir/datasetrecorder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/recorder.dir/datasetrecorder.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/murilo/Documents/TCC/Applications/DatasetRecorder/datasetrecorder.cpp -o CMakeFiles/recorder.dir/datasetrecorder.cpp.s

CMakeFiles/recorder.dir/datasetrecorder.cpp.o.requires:

.PHONY : CMakeFiles/recorder.dir/datasetrecorder.cpp.o.requires

CMakeFiles/recorder.dir/datasetrecorder.cpp.o.provides: CMakeFiles/recorder.dir/datasetrecorder.cpp.o.requires
	$(MAKE) -f CMakeFiles/recorder.dir/build.make CMakeFiles/recorder.dir/datasetrecorder.cpp.o.provides.build
.PHONY : CMakeFiles/recorder.dir/datasetrecorder.cpp.o.provides

CMakeFiles/recorder.dir/datasetrecorder.cpp.o.provides.build: CMakeFiles/recorder.dir/datasetrecorder.cpp.o


# Object files for target recorder
recorder_OBJECTS = \
"CMakeFiles/recorder.dir/datasetrecorder.cpp.o"

# External object files for target recorder
recorder_EXTERNAL_OBJECTS =

recorder: CMakeFiles/recorder.dir/datasetrecorder.cpp.o
recorder: CMakeFiles/recorder.dir/build.make
recorder: /usr/lib/libOpenNI.so
recorder: /usr/local/lib/libopencv_ml.so.3.2.0
recorder: /usr/local/lib/libopencv_objdetect.so.3.2.0
recorder: /usr/local/lib/libopencv_shape.so.3.2.0
recorder: /usr/local/lib/libopencv_stitching.so.3.2.0
recorder: /usr/local/lib/libopencv_superres.so.3.2.0
recorder: /usr/local/lib/libopencv_videostab.so.3.2.0
recorder: /usr/local/lib/libopencv_calib3d.so.3.2.0
recorder: /usr/local/lib/libopencv_features2d.so.3.2.0
recorder: /usr/local/lib/libopencv_flann.so.3.2.0
recorder: /usr/local/lib/libopencv_highgui.so.3.2.0
recorder: /usr/local/lib/libopencv_photo.so.3.2.0
recorder: /usr/local/lib/libopencv_video.so.3.2.0
recorder: /usr/local/lib/libopencv_videoio.so.3.2.0
recorder: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
recorder: /usr/local/lib/libopencv_imgproc.so.3.2.0
recorder: /usr/local/lib/libopencv_core.so.3.2.0
recorder: CMakeFiles/recorder.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/murilo/Documents/TCC/Applications/DatasetRecorder/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable recorder"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/recorder.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/recorder.dir/build: recorder

.PHONY : CMakeFiles/recorder.dir/build

CMakeFiles/recorder.dir/requires: CMakeFiles/recorder.dir/datasetrecorder.cpp.o.requires

.PHONY : CMakeFiles/recorder.dir/requires

CMakeFiles/recorder.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/recorder.dir/cmake_clean.cmake
.PHONY : CMakeFiles/recorder.dir/clean

CMakeFiles/recorder.dir/depend:
	cd /home/murilo/Documents/TCC/Applications/DatasetRecorder && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/murilo/Documents/TCC/Applications/DatasetRecorder /home/murilo/Documents/TCC/Applications/DatasetRecorder /home/murilo/Documents/TCC/Applications/DatasetRecorder /home/murilo/Documents/TCC/Applications/DatasetRecorder /home/murilo/Documents/TCC/Applications/DatasetRecorder/CMakeFiles/recorder.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/recorder.dir/depend

