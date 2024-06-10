unset(mvIMPACT_INCLUDE_DIRS)
unset(mvIMPACT_LIBS)

if (WITH_MVIMPACT)
find_path(mvIMPACT_INCLUDE_DIRS NAMES
        mvIMPACT_CPP/mvIMPACT_acquire.h
        HINTS
        /opt/mvIMPACT_Acquire/)

find_library(mvIMPACT_LIBS NAMES
        mvDeviceManager
        HINTS
        /opt/mvIMPACT_Acquire/lib/x86_64)

if (mvIMPACT_LIBS AND mvIMPACT_INCLUDE_DIRS)
    add_definitions( -DMVIMPACT )
    message(STATUS "mvIMPACT found, activating mvIMPACT support.")
endif (mvIMPACT_LIBS AND mvIMPACT_INCLUDE_DIRS)

endif (WITH_MVIMPACT)
