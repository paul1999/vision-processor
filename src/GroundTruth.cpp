#include "GroundTruth.h"

#include <yaml-cpp/yaml.h>


static void parseColor(const YAML::Node& list, std::vector<I2>& target) {
	for (const auto &item : list) {
		target.push_back({
			item["x"].as<int>(),
			item["y"].as<int>()
		});
	}
}

GroundTruth::GroundTruth(const std::string &source) {
	YAML::Node yaml = YAML::LoadFile(source);
	parseColor(yaml["yellow"], yellow);
	parseColor(yaml["blue"], blue);
	parseColor(yaml["orange"], orange);
	parseColor(yaml["green"], green);
	parseColor(yaml["pink"], pink);
}
