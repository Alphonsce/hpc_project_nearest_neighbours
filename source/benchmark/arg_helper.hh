#ifndef SOURCE_BENCHMARK_ARG_HELPER_HH_
#define SOURCE_BENCHMARK_ARG_HELPER_HH_

#include <string_view>
#include <span>

static inline std::string_view getArg(std::span<char *> aArgs,
	std::string_view aFlag, std::string_view aDefault = "")
{
	for (size_t i = 0; i + 1 < aArgs.size(); ++i)
		if (aArgs[i] == aFlag)
			return aArgs[i + 1];
	return aDefault;
}

static inline bool hasFlag(std::span<char *> aArgs, std::string_view aFlag)
{
	for (auto arg : aArgs)
		if (arg == aFlag)
			return true;
	return false;
}

#endif /* SOURCE_BENCHMARK_ARG_HELPER_HH_ */
