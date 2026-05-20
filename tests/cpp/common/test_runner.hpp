#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace PtoTestRunner {

struct CaseSpec {
    const char *name;
    std::function<bool(const std::string &)> run;
};

inline void PrintUsage(const char *prog)
{
    std::printf("Usage: %s [--all] [--case <name>]\n", prog);
    std::printf("  (default) run smoke-test cases only\n");
    std::printf("  --all     run smoke + full regression cases\n");
    std::printf("  --case    run a single named case\n");
}

inline bool RunCaseList(const std::vector<CaseSpec> &cases, const std::string &dataRoot, const char *filterName)
{
    int failed = 0;
    int passed = 0;
    int skipped = 0;

    for (const auto &spec : cases) {
        if (filterName != nullptr && std::string(filterName) != spec.name) {
            continue;
        }
        if (spec.run == nullptr) {
            skipped++;
            continue;
        }

        const std::string dataDir = dataRoot + "/" + spec.name;
        std::printf("[INFO]  Running %s\n", spec.name);
        if (spec.run(dataDir)) {
            std::printf("[INFO]  PASS: %s\n", spec.name);
            passed++;
        } else {
            std::printf("[ERROR] FAIL: %s\n", spec.name);
            failed++;
        }
    }

    std::printf("[INFO]  Summary: %d passed, %d failed", passed, failed);
    if (skipped > 0) {
        std::printf(", %d skipped (not in this build)", skipped);
    }
    std::printf("\n");
    return failed == 0 && passed > 0;
}

inline int MainImpl(int argc, char **argv, const std::vector<CaseSpec> &smokeCases,
                    const std::vector<CaseSpec> &fullCases = {})
{
    bool runAll = false;
    const char *caseFilter = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--all") {
            runAll = true;
        } else if (arg == "--case" && i + 1 < argc) {
            caseFilter = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::printf("[ERROR] Unknown argument: %s\n", arg.c_str());
            PrintUsage(argv[0]);
            return 1;
        }
    }

    const std::string dataRoot = "cases";

    if (caseFilter != nullptr) {
        if (RunCaseList(smokeCases, dataRoot, caseFilter)) {
            return 0;
        }
        if (!fullCases.empty() && RunCaseList(fullCases, dataRoot, caseFilter)) {
            return 0;
        }
#ifndef PTOISA_FULL_TEST
        if (!fullCases.empty()) {
            std::printf("[ERROR] Case '%s' requires full build: ./build.sh --full\n", caseFilter);
        }
#endif
        return 1;
    }

    bool ok = RunCaseList(smokeCases, dataRoot, nullptr);
    if (runAll) {
#ifndef PTOISA_FULL_TEST
        std::printf("[ERROR] --all requires a full build: ./build.sh --full\n");
        return 1;
#else
        ok = RunCaseList(fullCases, dataRoot, nullptr) && ok;
#endif
    }
    return ok ? 0 : 1;
}

} // namespace PtoTestRunner
