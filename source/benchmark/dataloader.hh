#ifndef SOURCE_BENCHMARK_DATALOADER_HH_
#define SOURCE_BENCHMARK_DATALOADER_HH_

#include "generic/types.hh"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

// Minimal NPY reader for float32 2-D C-contiguous arrays.
//
// Supports NPY format v1.0 and v2.0 produced by numpy.save().
// Dtype must be '<f4' (little-endian float32) or '=f4' (native, x86).
// Fortran-order arrays are rejected.
class DataLoader {
public:
	DataLoader(std::filesystem::path aPath) : pointsPath(std::move(aPath))
	{
		if (!std::filesystem::exists(pointsPath))
			throw std::invalid_argument(
				"DataLoader: file not found: " + pointsPath.string());
	}

	PointsPtr get()
	{
		std::ifstream f(pointsPath, std::ios::binary);
		if (!f)
			throw std::runtime_error(
				"DataLoader: cannot open " + pointsPath.string());

		// --- magic (\x93NUMPY) + version (major, minor) ---
		char magic[6];
		f.read(magic, 6);
		if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
			throw std::runtime_error(
				"DataLoader: not a .npy file: " + pointsPath.string());

		uint8_t major = 0, minor = 0;
		f.read(reinterpret_cast<char *>(&major), 1);
		f.read(reinterpret_cast<char *>(&minor), 1);
		(void)minor;

		// --- header length (uint16 for v1, uint32 for v2) ---
		uint32_t headerLen = 0;
		if (major == 1) {
			uint16_t hl = 0;
			f.read(reinterpret_cast<char *>(&hl), 2);
			headerLen = hl;
		} else if (major == 2) {
			f.read(reinterpret_cast<char *>(&headerLen), 4);
		} else {
			throw std::runtime_error("DataLoader: unsupported NPY major version "
				+ std::to_string(major));
		}

		// --- parse header dict ---
		std::string header(headerLen, '\0');
		f.read(header.data(), headerLen);
		if (!f)
			throw std::runtime_error("DataLoader: truncated NPY header");

		if (header.find("'F'") != std::string::npos
			|| header.find("\"F\"") != std::string::npos)
			throw std::runtime_error(
				"DataLoader: Fortran-order arrays are not supported");

		checkDtype(header);
		auto [n, dim] = parseShape(header);

		// --- read raw float32 payload ---
		auto pts = std::make_shared<Points>(n, Point(dim));
		for (size_t i = 0; i < n; ++i) {
			f.read(reinterpret_cast<char *>(pts->at(i).data()),
				static_cast<std::streamsize>(dim * sizeof(float)));
		}
		if (!f)
			throw std::runtime_error(
				"DataLoader: truncated data in " + pointsPath.string());

		return pts;
	}

private:
	std::filesystem::path pointsPath;

	// Returns (n, dim) from the 'shape': (n, dim) entry.
	static std::pair<size_t, size_t> parseShape(const std::string &aHdr)
	{
		auto pos = aHdr.find("'shape'");
		if (pos == std::string::npos) pos = aHdr.find("\"shape\"");
		if (pos == std::string::npos)
			throw std::runtime_error("DataLoader: 'shape' key not found");

		auto lp = aHdr.find('(', pos);
		auto rp = aHdr.find(')', lp);
		if (lp == std::string::npos || rp == std::string::npos)
			throw std::runtime_error("DataLoader: malformed shape tuple");

		std::string tuple = aHdr.substr(lp + 1, rp - lp - 1);
		auto comma		  = tuple.find(',');
		if (comma == std::string::npos)
			throw std::runtime_error(
				"DataLoader: shape must be 2-D, got 1-D tuple");

		size_t n   = std::stoul(tuple.substr(0, comma));
		size_t dim = std::stoul(tuple.substr(comma + 1));
		if (n == 0 || dim == 0)
			throw std::runtime_error("DataLoader: shape has zero dimension");

		return {n, dim};
	}

	// Accepts '<f4' (little-endian float32) and '=f4' (native, ≡ '<f4' on x86).
	static void checkDtype(const std::string &aHdr)
	{
		bool ok = aHdr.find("'<f4'") != std::string::npos
			|| aHdr.find("\"<f4\"") != std::string::npos
			|| aHdr.find("'=f4'") != std::string::npos
			|| aHdr.find("\"=f4\"") != std::string::npos;

		if (!ok)
			throw std::runtime_error(
				"DataLoader: only float32 ('<f4') .npy files are supported");
	}
};

#endif /* SOURCE_BENCHMARK_DATALOADER_HH_ */
