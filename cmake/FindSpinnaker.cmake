unset(SPINNAKER_INCLUDE_DIRS)
unset(SPINNAKER_LIBS)

if (WITH_SPINNAKER)

find_path(SPINNAKER_INCLUDE_DIRS NAMES
        Spinnaker.h
        HINTS
        /opt/spinnaker/include/
        /usr/include/spinnaker/
        /usr/local/include/spinnaker/)

find_library(SPINNAKER_LIBS NAMES Spinnaker
        HINTS
        /opt/spinnaker/lib
        /usr/lib
        /usr/local/lib)

if (SPINNAKER_INCLUDE_DIRS AND SPINNAKER_LIBS)
    add_definitions( -DSPINNAKER )
    message(STATUS "Spinnaker found, activating Spinnaker support.")
endif (SPINNAKER_INCLUDE_DIRS AND SPINNAKER_LIBS)

endif (WITH_SPINNAKER)