# include

Reserved for project-wide C/C++ header files (`.h`) shared across the source
files in `../src`. Currently empty (no extra headers needed); the firmware in
`src/main.cpp` only includes library headers at this time.

PlatformIO automatically adds this directory to the include path, so any
header placed here can be included as `#include "myheader.h"`.