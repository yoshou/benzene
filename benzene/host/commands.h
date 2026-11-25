#pragma once

namespace benzene::host {

// The commands, named as the checks in python/benzene/models that run them: argv[1] is the command's name, argv[2] ..
// its arguments; they return the process exit status.
int run_superpoint_lightglue(int argc, char ** argv);
int run_xfeat_lightglue(int argc, char ** argv);
int run_aliked_lightglue(int argc, char ** argv);
int run_loma(int argc, char ** argv);
int run_vggt(int argc, char ** argv);
int run_anysplat(int argc, char ** argv);
int run_cosmos3_edge_llm_prefill(int argc, char ** argv);
int run_cosmos3_edge_llm_decode(int argc, char ** argv);
int run_cosmos3_edge_vision(int argc, char ** argv);
int run_cosmos3_edge_generate(int argc, char ** argv);

}  // namespace benzene::host
