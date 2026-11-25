#pragma once

namespace benzene::host {

struct Option {
    const char * name;         // "--name"
    const char ** value;       // an option with a value: set to it; nullptr for a flag
    bool * present = nullptr;  // a flag: set when given
};

// The arguments after the command's name (argv[1]): exactly `positional` positional arguments, then the options.
// Prints "usage: <command> <usage>" and returns false on anything else.
bool parse_command_line(int argc, char ** argv, int positional, const Option * options, int option_count,
                        const char * usage);

// A whole decimal integer in [minimum, maximum] (false and a message otherwise).
bool parse_integer(const char * text, const char * what, int minimum, int maximum, int * value);

// A finite decimal number in [minimum, maximum] (false and a message otherwise).
bool parse_float(const char * text, const char * what, float minimum, float maximum, float * value);

}  // namespace benzene::host
