#    Copyright 2025 Paul Bergmann
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

find_package(PkgConfig REQUIRED)
pkg_check_modules(DC1394 libdc1394-2)

if (WITH_DC1394 AND DC1394_FOUND)
    add_definitions(-D DC1394)
    message(STATUS "libdc1394-2 found, activating libdc1394-2 support.")
endif()
