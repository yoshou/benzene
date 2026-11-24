#include "host/command_line.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace benzene::host {

bool parse_command_line(int argc, char ** argv, int positional, const Option * options, int option_count,
                        const char * usage) {
    bool valid = argc >= 2 + positional;
    for (int i = 2 + positional; valid && i < argc; i++) {
        const Option * option = nullptr;
        for (int o = 0; o < option_count; o++) {
            if (strcmp(argv[i], options[o].name) == 0) option = &options[o];
        }
        if (option == nullptr) {
            valid = false;
        } else if (option->value != nullptr) {
            valid = i + 1 < argc;
            if (valid) *option->value = argv[++i];
        } else {
            *option->present = true;
        }
    }
    if (!valid) fprintf(stderr, "usage: %s %s\n", argv[1], usage);
    return valid;
}

bool parse_integer(const char * text, const char * what, int minimum, int maximum, int * value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        fprintf(stderr, "bad %s %s (%d .. %d)\n", what, text, minimum, maximum);
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

}  // namespace benzene::host
