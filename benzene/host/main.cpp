#include <cstdio>
#include <cstring>

#include "host/commands.h"
#include "host/fastrpc.h"

namespace {

using namespace benzene::host;

struct Command {
    const char * name;
    int (*run)(int argc, char ** argv);
};

const Command kCommands[] = {
    {"superpoint_lightglue", run_superpoint_lightglue},
    {"xfeat_lightglue", run_xfeat_lightglue},
    {"aliked_lightglue", run_aliked_lightglue},
    {"loma", run_loma},
    {"vggt", run_vggt},
    {"cosmos3_edge_llm_prefill", run_cosmos3_edge_llm_prefill},
    {"cosmos3_edge_llm_decode", run_cosmos3_edge_llm_decode},
    {"cosmos3_edge_vision", run_cosmos3_edge_vision},
    {"cosmos3_edge_generate", run_cosmos3_edge_generate},
};

const Command * find_command(const char * name) {
    for (const Command & command : kCommands) {
        if (strcmp(name, command.name) == 0) return &command;
    }
    return nullptr;
}

bool print_info() {
    uint32_t architecture;
    uint32_t vtcm_bytes;
    uint32_t hvx_units;
    uint32_t hmx_units;
    uint32_t pcycle_khz;
    if (!rpc_succeeded(benzene_info(session(), &architecture, &vtcm_bytes, &hvx_units, &hmx_units, &pcycle_khz),
                       "info")) {
        return false;
    }
    printf("arch v%u  vtcm %u KiB  hvx(128B) %u  hmx %u  pclk %.0f MHz\n", architecture, vtcm_bytes >> 10, hvx_units,
           hmx_units, pcycle_khz / 1000.0);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const Command * command = argc >= 2 ? find_command(argv[1]) : nullptr;
    if (command == nullptr) {
        fprintf(stderr, "usage: %s <command> <arguments>, the commands:", argv[0]);
        for (const Command & known : kCommands) fprintf(stderr, " %s", known.name);
        fprintf(stderr, "\n");
        return 1;
    }

    install_signal_guard();
    if (!open_session()) return 1;
    const int status = print_info() ? command->run(argc, argv) : 1;
    close_session();
    if (status != 0) fprintf(stderr, "%s failed\n", command->name);
    return status;
}
